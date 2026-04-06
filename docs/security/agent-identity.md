---
title: Agent Identity
description: Ed25519 cryptographic identity and message signing for agents
weight: 52
---

`AgentIdentity` provides Ed25519 cryptographic identity for agents. Each identity has a unique keypair — private key material is **never** serialized, logged, or included in `repr()`.

## Requirements

```bash
pip install cryptography
```

## Quick start

```python
from syrin.security import AgentIdentity

# Generate a fresh identity
identity = AgentIdentity.generate()
print(identity.agent_id)          # UUID string
print(len(identity.public_key_bytes))  # 32

# Sign a message
message = b"approved: transfer $100 to account 12345"
signature = identity.sign(message)
print(len(signature))  # 64

# Verify the signature (pass public key only — never private)
valid = AgentIdentity.verify(message, signature, identity.public_key_bytes)
assert valid  # True

# Tampered message fails
tampered = b"approved: transfer $9999 to account 99999"
assert not AgentIdentity.verify(tampered, signature, identity.public_key_bytes)
```

## Custom agent ID

```python
identity = AgentIdentity.generate(agent_id="orchestrator-v1")
print(identity.agent_id)  # "orchestrator-v1"
```

## Hook firing

```python
from syrin.enums import Hook
from syrin.security import AgentIdentity

def on_event(hook: Hook, ctx: dict) -> None:
    print(f"[{hook}] {ctx}")

identity = AgentIdentity.generate()
sig = identity.sign(b"hello")

AgentIdentity.verify(b"hello", sig, identity.public_key_bytes, fire_event_fn=on_event)
# [identity.verified] {'agent_public_key_size': 32}

AgentIdentity.verify(b"bad", sig, identity.public_key_bytes, fire_event_fn=on_event)
# [signature.invalid] {'reason': 'Signature verification failed'}
```

## Safety guarantees

- `repr(identity)` and `str(identity)` show only `agent_id` and public key size — never private key bytes.
- `identity.to_dict()` contains only `agent_id` and `public_key_bytes` (hex-encoded) — no private key.
- The `_private_key` dataclass field has `repr=False` and is excluded from equality comparison.
- Ed25519 signing is deterministic: same key + message always produces the same signature.

## Hooks reference

Two hooks are emitted during identity operations. `Hook.IDENTITY_VERIFIED` fires when `verify()` succeeds. `Hook.SIGNATURE_INVALID` fires when `verify()` fails due to a wrong message, tampered signature, or wrong key.
