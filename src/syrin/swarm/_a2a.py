"""A2A messaging — agent-to-agent communication."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

_T = TypeVar("_T")

from syrin.enums import A2AChannel, Hook
from syrin.events import EventContext, Events
from syrin.swarm._agent_ref import AgentRef, _aid

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class A2ABudgetExceededError(Exception):
    """Raised when a sender exceeds budget or message-count limits in the A2A router.

    Attributes:
        sender_id: The agent ID that exceeded the limit.
        limit: The configured limit (message count or total spend in USD).
        actual: The actual value that caused the violation.
    """

    def __init__(self, sender_id: str, limit: float, actual: float) -> None:
        """Initialise A2ABudgetExceededError.

        Args:
            sender_id: The agent ID that exceeded the limit.
            limit: The configured limit (message count or total spend).
            actual: The actual value at the point of violation.
        """
        super().__init__(f"Sender '{sender_id}' exceeded limit: actual={actual}, limit={limit}")
        self.sender_id = sender_id
        self.limit = limit
        self.actual = actual


class A2ATimeoutError(Exception):
    """Raised when an ack does not arrive within the configured timeout.

    Attributes:
        message_id: ID of the message that timed out.
        timeout: Configured timeout in seconds.
    """

    def __init__(self, message_id: str, timeout: float) -> None:
        """Initialise A2ATimeoutError.

        Args:
            message_id: ID of the unacknowledged message.
            timeout: The timeout value that was exceeded.
        """
        super().__init__(f"Ack not received for message '{message_id}' within {timeout}s")
        self.message_id = message_id
        self.timeout = timeout


class A2AMessageTooLarge(Exception):
    """Raised when a message payload exceeds the configured size limit.

    Attributes:
        size_bytes: Actual serialized size.
        max_bytes: Configured limit.
    """

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        """Initialise A2AMessageTooLarge.

        Args:
            size_bytes: Actual message size in bytes.
            max_bytes: Configured maximum.
        """
        super().__init__(f"Message size {size_bytes} bytes exceeds limit of {max_bytes} bytes")
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class A2AConfig(BaseModel):  # type: ignore[explicit-any]
    """Configuration for the :class:`A2ARouter`.

    Attributes:
        max_message_size: Maximum allowed serialized message size in bytes.
            ``0`` means unlimited (default).
        max_queue_depth: Maximum number of messages in any single agent's inbox
            before overflow messages are dropped.  ``0`` means unlimited.
        audit_all: When ``True``, every sent message is appended to an in-memory
            audit log accessible via :meth:`~A2ARouter.audit_log`.
        budget_per_message: Per-message budget cap in USD.  ``0.0`` means
            unlimited (default).  Used with ``max_messages_per_sender`` to
            compute the total per-sender spend cap.
        max_messages_per_sender: Maximum number of messages a single sender may
            send.  ``0`` means unlimited (default).  When exceeded,
            :meth:`~A2ARouter.send` raises :class:`A2ABudgetExceededError`.
    """

    max_message_size: int = 0
    max_queue_depth: int = 0
    audit_all: bool = False
    budget_per_message: float = 0.0
    max_messages_per_sender: int = 0


# ---------------------------------------------------------------------------
# Envelope & Audit
# ---------------------------------------------------------------------------


class A2AMessageEnvelope(BaseModel):  # type: ignore[explicit-any]
    """Wraps a typed message with routing metadata.

    Attributes:
        message_id: Unique identifier for this envelope.
        from_agent: Sender agent ID.
        to_agent: Target agent ID, topic name, or ``"broadcast"``.
        channel: :class:`~syrin.enums.A2AChannel` routing mode.
        message_type: ``__name__`` of the payload class.
        payload: The original typed message (Pydantic model or ``@structured`` dataclass).
        timestamp: UTC timestamp when the envelope was created.
        requires_ack: Whether the sender is waiting for an acknowledgement.
    """

    message_id: str
    from_agent: str
    to_agent: str
    channel: A2AChannel
    message_type: str
    # `payload` holds a Pydantic BaseModel or @structured dataclass. It is
    # typed as `object` because the router is generic — callers narrow the
    # type using `get_typed_payload()` or a manual isinstance() check.
    payload: object
    timestamp: datetime
    requires_ack: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def get_typed_payload(self, payload_type: type[_T]) -> _T:
        """Return the payload narrowed to *payload_type*.

        Prefer this over raw ``envelope.payload`` so that mypy can infer
        the concrete type of the message inside an inbox handler.

        Args:
            payload_type: The expected type of the payload (Pydantic model
                class or ``@structured`` dataclass class).

        Returns:
            The payload cast to *payload_type*.

        Raises:
            TypeError: If the payload is not an instance of *payload_type*.

        Example:
            envelope = await router.receive(agent_id="writer", timeout=5.0)
            if envelope:
                result = envelope.get_typed_payload(ResearchResult)
                print(result.summary)  # fully typed
        """
        if not isinstance(self.payload, payload_type):
            raise TypeError(
                f"Expected payload of type {payload_type.__name__!r}, "
                f"got {type(self.payload).__name__!r}"
            )
        return self.payload


class A2AAuditEntry(BaseModel):  # type: ignore[explicit-any]
    """A single entry in the A2A audit log.

    Attributes:
        from_agent: Sender agent ID.
        to_agent: Recipient agent ID or topic.
        message_type: ``__name__`` of the payload class.
        timestamp: When the message was sent.
        size_bytes: Serialized payload size in bytes.
    """

    from_agent: str
    to_agent: str
    message_type: str
    timestamp: datetime
    size_bytes: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class A2ARouter:
    """Central message router for agent-to-agent communication.

    Manages per-agent asyncio queues (inboxes).  Supports three delivery
    channels:

    - :attr:`~syrin.enums.A2AChannel.DIRECT` — point-to-point delivery.
    - :attr:`~syrin.enums.A2AChannel.BROADCAST` — delivered to all registered
      agents (excluding sender).
    - :attr:`~syrin.enums.A2AChannel.TOPIC` — delivered to all agents
      subscribed to the named topic.

    Args:
        config: :class:`A2AConfig` controlling size limits, queue depth, and
            audit behaviour.  Defaults to permissive settings.
        swarm_events: Optional :class:`~syrin.events.Events` bus for firing
            lifecycle hooks.

    Example:
        router = A2ARouter(config=A2AConfig(audit_all=True))
        router.register_agent("writer")
        router.register_agent("researcher")

        await router.send(
            from_agent="researcher",
            to_agent="writer",
            message=MyMessage(text="hello"),
        )
        envelope = await router.receive(agent_id="writer", timeout=5.0)
    """

    def __init__(
        self,
        config: A2AConfig | None = None,
        swarm_events: Events | None = None,
    ) -> None:
        """Initialise A2ARouter.

        Args:
            config: Router configuration.  Defaults to :class:`A2AConfig` defaults.
            swarm_events: Events bus for lifecycle hook emission.
        """
        self._config: A2AConfig = config or A2AConfig()
        self._events: Events | None = swarm_events
        self._inboxes: dict[str, asyncio.Queue[A2AMessageEnvelope]] = {}
        self._topics: dict[str, set[str]] = {}
        self._audit: list[A2AAuditEntry] = []
        # Pending ack futures: message_id -> Future
        self._ack_futures: dict[str, asyncio.Future[None]] = {}
        # Budget tracking: per-sender message counts and spend
        self._sent_counts: dict[str, int] = {}
        self._sender_spend: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, hook: Hook, data: dict[str, object]) -> None:
        """Emit *hook* via swarm events if connected."""
        if self._events is not None:
            ctx = EventContext(data)
            ctx.scrub()
            self._events._trigger(hook, ctx)

    def _serialize_size(self, message: object) -> int:
        """Return the serialized JSON byte-length of *message*.

        Handles both Pydantic models (``model_dump_json()``) and
        ``@structured`` dataclass instances (``json.dumps(asdict(...))``)."""
        if hasattr(message, "model_dump_json"):
            return len(message.model_dump_json().encode())
        if dataclasses.is_dataclass(message) and not isinstance(message, type):
            return len(json.dumps(dataclasses.asdict(message)).encode())
        return len(str(message).encode())

    def _make_envelope(
        self,
        from_agent: str,
        to_agent: str,
        message: object,
        channel: A2AChannel,
        requires_ack: bool = False,
    ) -> A2AMessageEnvelope:
        """Build an :class:`A2AMessageEnvelope` wrapping *message*."""
        return A2AMessageEnvelope(
            message_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            channel=channel,
            message_type=type(message).__name__,
            payload=message,
            timestamp=datetime.utcnow(),
            requires_ack=requires_ack,
        )

    async def _deliver(
        self,
        envelope: A2AMessageEnvelope,
        target_agent: str,
    ) -> bool:
        """Place *envelope* in *target_agent*'s inbox.

        Returns:
            ``True`` if delivered, ``False`` if queue was full and message dropped.
        """
        queue = self._inboxes.get(target_agent)
        if queue is None:
            return False

        max_depth = self._config.max_queue_depth
        if max_depth > 0 and queue.qsize() >= max_depth:
            self._fire(
                Hook.A2A_QUEUE_FULL,
                {
                    "agent_id": target_agent,
                    "from_agent": envelope.from_agent,
                    "message_id": envelope.message_id,
                    "queue_depth": queue.qsize(),
                },
            )
            return False

        await queue.put(envelope)
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_inbox(self, agent: AgentRef | str) -> bool:
        """Return whether *agent* has a registered inbox.

        Args:
            agent: Agent instance to check.

        Returns:
            ``True`` if the agent has been registered.
        """
        return _aid(agent) in self._inboxes

    def register_agent(self, agent: AgentRef | str) -> None:
        """Create an inbox queue for *agent*.

        Safe to call multiple times for the same agent; subsequent calls are
        no-ops.

        Args:
            agent: Agent instance to register.  The inbox key is derived from
                ``agent.agent_id`` automatically.
        """
        aid = _aid(agent)
        if aid not in self._inboxes:
            self._inboxes[aid] = asyncio.Queue()

    def subscribe(self, agent: AgentRef | str, topic: str) -> None:
        """Subscribe *agent* to *topic*.

        Messages sent via :meth:`send_topic` with matching *topic* will be
        delivered to all subscribed agents.

        Args:
            agent: Agent instance to subscribe.
            topic: Topic name (string channel label, not an agent reference).
        """
        aid = _aid(agent)
        if topic not in self._topics:
            self._topics[topic] = set()
        self._topics[topic].add(aid)

    # ------------------------------------------------------------------
    # Sending — three distinct methods replace the old channel-based send()
    # ------------------------------------------------------------------

    async def send(
        self,
        from_agent: AgentRef | str,
        to_agent: AgentRef | str,
        message: object,
        channel: A2AChannel = A2AChannel.DIRECT,
    ) -> None:
        """Send *message* from *from_agent* to *to_agent* via *channel*.

        Default channel is ``DIRECT`` (point-to-point).  Pass
        ``channel=A2AChannel.BROADCAST`` to reach every registered agent or
        ``channel=A2AChannel.TOPIC`` to fan out to topic subscribers (use the
        topic name as *to_agent*).

        Args:
            from_agent: Sending agent instance or agent ID string.
            to_agent: Receiving agent instance, agent ID string, or topic name.
            message: A Pydantic :class:`~pydantic.BaseModel` instance **or** a
                ``@structured`` dataclass instance.
            channel: Delivery channel; defaults to :attr:`~syrin.enums.A2AChannel.DIRECT`.

        Raises:
            A2AMessageTooLarge: When the serialized payload exceeds
                :attr:`A2AConfig.max_message_size`.
            A2ABudgetExceededError: When the sender has exceeded
                :attr:`A2AConfig.max_messages_per_sender`.
        """
        await self._send_internal(
            from_id=_aid(from_agent),
            to_id=_aid(to_agent),
            message=message,
            channel=channel,
        )

    async def send_topic(
        self,
        from_agent: AgentRef | str,
        topic: str,
        message: object,
    ) -> None:
        """Send *message* to all agents subscribed to *topic*.

        Agents subscribe via :meth:`subscribe`.  The *topic* string is a
        routing label — not an agent reference.

        Args:
            from_agent: Sending agent instance.
            topic: Topic name used to select subscribers.
            message: A Pydantic model or ``@structured`` dataclass payload.

        Raises:
            A2AMessageTooLarge: When the serialized payload exceeds the limit.
            A2ABudgetExceededError: When the sender has exceeded the message limit.
        """
        await self._send_internal(
            from_id=_aid(from_agent),
            to_id=topic,
            message=message,
            channel=A2AChannel.TOPIC,
        )

    async def send_broadcast(
        self,
        from_agent: AgentRef | str,
        message: object,
    ) -> None:
        """Broadcast *message* to every registered agent except the sender.

        Args:
            from_agent: Sending agent instance.
            message: A Pydantic model or ``@structured`` dataclass payload.

        Raises:
            A2AMessageTooLarge: When the serialized payload exceeds the limit.
            A2ABudgetExceededError: When the sender has exceeded the message limit.
        """
        await self._send_internal(
            from_id=_aid(from_agent),
            to_id="__broadcast__",
            message=message,
            channel=A2AChannel.BROADCAST,
        )

    async def _send_internal(
        self,
        from_id: str,
        to_id: str,
        message: object,
        channel: A2AChannel,
    ) -> None:
        """Core delivery logic shared by send / send_topic / send_broadcast."""
        size_bytes = self._serialize_size(message)
        max_size = self._config.max_message_size
        if max_size > 0 and size_bytes > max_size:
            raise A2AMessageTooLarge(size_bytes=size_bytes, max_bytes=max_size)

        max_msgs = self._config.max_messages_per_sender
        if max_msgs > 0:
            current_count = self._sent_counts.get(from_id, 0)
            if current_count >= max_msgs:
                raise A2ABudgetExceededError(
                    sender_id=from_id,
                    limit=float(max_msgs),
                    actual=float(current_count + 1),
                )

        envelope = self._make_envelope(
            from_agent=from_id,
            to_agent=to_id,
            message=message,
            channel=channel,
        )

        self._sent_counts[from_id] = self._sent_counts.get(from_id, 0) + 1
        self._sender_spend[from_id] = (
            self._sender_spend.get(from_id, 0.0) + self._config.budget_per_message
        )

        if channel == A2AChannel.BROADCAST:
            targets = [a for a in self._inboxes if a != from_id]
        elif channel == A2AChannel.TOPIC:
            targets = list(self._topics.get(to_id, set()))
        else:
            targets = [to_id]

        for target in targets:
            await self._deliver(envelope, target)

        self._fire(
            Hook.A2A_MESSAGE_SENT,
            {
                "from_agent": from_id,
                "to_agent": to_id,
                "message_type": type(message).__name__,
                "size_bytes": size_bytes,
                "channel": str(channel),
                "message_id": envelope.message_id,
            },
        )

        if self._config.audit_all:
            self._audit.append(
                A2AAuditEntry(
                    from_agent=from_id,
                    to_agent=to_id,
                    message_type=type(message).__name__,
                    timestamp=envelope.timestamp,
                    size_bytes=size_bytes,
                )
            )

    async def send_with_ack(
        self,
        from_agent: AgentRef | str,
        to_agent: AgentRef | str,
        message: object,
        timeout: float = 30.0,
    ) -> None:
        """Send *message* and wait for the recipient to acknowledge.

        Args:
            from_agent: Sending agent instance.
            to_agent: Receiving agent instance.
            message: A Pydantic model or ``@structured`` dataclass payload.
            timeout: Maximum seconds to wait for an ack.

        Raises:
            A2ATimeoutError: If the ack does not arrive within *timeout* seconds.
            A2AMessageTooLarge: If the payload exceeds the size limit.
        """
        from_id = _aid(from_agent)
        to_id = _aid(to_agent)

        size_bytes = self._serialize_size(message)
        max_size = self._config.max_message_size
        if max_size > 0 and size_bytes > max_size:
            raise A2AMessageTooLarge(size_bytes=size_bytes, max_bytes=max_size)

        envelope = self._make_envelope(
            from_agent=from_id,
            to_agent=to_id,
            message=message,
            channel=A2AChannel.DIRECT,
            requires_ack=True,
        )

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._ack_futures[envelope.message_id] = fut

        await self._deliver(envelope, to_id)
        self._fire(
            Hook.A2A_MESSAGE_SENT,
            {
                "from_agent": from_id,
                "to_agent": to_id,
                "message_type": type(message).__name__,
                "size_bytes": size_bytes,
                "channel": str(A2AChannel.DIRECT),
                "message_id": envelope.message_id,
                "requires_ack": True,
            },
        )

        if self._config.audit_all:
            self._audit.append(
                A2AAuditEntry(
                    from_agent=from_id,
                    to_agent=to_id,
                    message_type=type(message).__name__,
                    timestamp=envelope.timestamp,
                    size_bytes=size_bytes,
                )
            )

        try:
            await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as err:
            self._ack_futures.pop(envelope.message_id, None)
            self._fire(
                Hook.A2A_MESSAGE_TIMEOUT,
                {
                    "message_id": envelope.message_id,
                    "from_agent": from_id,
                    "to_agent": to_id,
                    "timeout": timeout,
                },
            )
            raise A2ATimeoutError(message_id=envelope.message_id, timeout=timeout) from err

    async def receive(
        self,
        agent_id: AgentRef | str,
        timeout: float | None = None,
    ) -> A2AMessageEnvelope | None:
        """Consume the next message from *agent_id*'s inbox.

        Fires :attr:`~syrin.enums.Hook.A2A_MESSAGE_RECEIVED` when a message is
        returned.

        Args:
            agent_id: Agent instance or agent ID string whose inbox to read.
            timeout: Maximum seconds to wait.  ``None`` blocks indefinitely.

        Returns:
            The next :class:`A2AMessageEnvelope`, or ``None`` if the timeout
            expires before a message arrives.
        """
        aid = _aid(agent_id)
        queue = self._inboxes.get(aid)
        if queue is None:
            return None

        try:
            if timeout is not None:
                envelope = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                envelope = await queue.get()
        except TimeoutError:
            return None

        self._fire(
            Hook.A2A_MESSAGE_RECEIVED,
            {
                "agent_id": aid,
                "from_agent": envelope.from_agent,
                "message_type": envelope.message_type,
                "message_id": envelope.message_id,
            },
        )
        return envelope

    async def ack(self, agent_id: AgentRef | str, message_id: str) -> None:
        """Acknowledge receipt of *message_id*.

        Resolves the pending future for :meth:`send_with_ack` and fires
        :attr:`~syrin.enums.Hook.A2A_MESSAGE_ACKED`.

        Args:
            agent_id: Agent instance or agent ID string acknowledging the message.
            message_id: ID of the message to acknowledge.
        """
        aid = _aid(agent_id)
        fut = self._ack_futures.pop(message_id, None)
        if fut is not None and not fut.done():
            fut.set_result(None)

        self._fire(
            Hook.A2A_MESSAGE_ACKED,
            {
                "agent_id": aid,
                "message_id": message_id,
            },
        )

    def audit_log(self) -> list[A2AAuditEntry]:
        """Return a copy of the audit log.

        Returns:
            List of :class:`A2AAuditEntry` objects in insertion order.
            Empty list when ``audit_all=False`` (default).
        """
        return list(self._audit)


__all__ = [
    "A2AAuditEntry",
    "A2ABudgetExceededError",
    "A2AConfig",
    "A2AMessageEnvelope",
    "A2AMessageTooLarge",
    "A2ARouter",
    "A2ATimeoutError",
]
