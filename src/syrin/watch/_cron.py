"""CronProtocol — cron-scheduled trigger using croniter."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from syrin.watch._trigger import TriggerEvent

_log = logging.getLogger(__name__)


class CronProtocol:
    """Cron-schedule trigger source.

    Fires the agent on a cron schedule using ``croniter`` for expression parsing.
    No external cron daemon required — runs entirely in-process.

    Args:
        schedule: Standard 5-field POSIX cron expression
            (``"minute hour day month weekday"``). E.g. ``"0 9 * * 1-5"`` for
            9 AM on weekdays.
        input: Fixed input string passed to ``agent.run()`` on each tick. Default: ``""``.
        timezone: Timezone name for schedule evaluation. Default: ``"UTC"``.
        run_on_start: If ``True``, run once immediately when ``start()`` is called
            before the first scheduled tick. Default: ``False``.

    Example::

        from syrin.watch import CronProtocol

        agent.watch(
            protocol=CronProtocol(
                schedule="0 9 * * 1-5",
                input="Run the daily morning report",
                timezone="America/New_York",
            )
        )
    """

    def __init__(
        self,
        schedule: str = "* * * * *",
        input: str = "",  # noqa: A002
        timezone: str = "UTC",
        run_on_start: bool = False,
    ) -> None:
        self.schedule = schedule
        self.input = input
        self.timezone = timezone
        self.run_on_start = run_on_start
        self._running = False

    def next_run_time(self) -> float:
        """Return the next scheduled run time as a Unix timestamp.

        Uses ``croniter`` to compute the next occurrence after now.

        Returns:
            Unix timestamp (float) of the next scheduled run.

        Raises:
            ValueError: If the cron expression is invalid.
        """
        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "croniter is required for CronProtocol. Install it with: pip install croniter"
            ) from exc

        now = datetime.now(UTC)
        cron = croniter(self.schedule, now)
        result = cron.get_next(float)
        return float(result)

    async def start(
        self,
        handler: Callable[[TriggerEvent], Awaitable[None]],
    ) -> None:
        """Start the cron loop. Blocks until ``stop()`` is called.

        Args:
            handler: Async function called on each scheduled tick.
        """
        self._running = True

        if self.run_on_start:
            await self._fire(handler)

        while self._running:
            try:
                next_ts = self.next_run_time()
            except (ImportError, ValueError) as exc:
                _log.error(f"CronProtocol schedule error: {exc}")
                break

            delay = next_ts - time.time()
            if delay > 0:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break

            if self._running:
                await self._fire(handler)

    async def _fire(
        self,
        handler: Callable[[TriggerEvent], Awaitable[None]],
    ) -> None:
        event = TriggerEvent(
            input=self.input,
            source="cron",
            metadata={"schedule": self.schedule, "timezone": self.timezone},
            trigger_id=str(uuid.uuid4()),
        )
        try:
            await handler(event)
        except Exception as exc:
            _log.error(f"CronProtocol handler error: {exc}")

    async def stop(self) -> None:
        """Stop the cron loop."""
        self._running = False
