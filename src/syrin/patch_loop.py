import re

with open('loop.py', 'r') as f:
    content = f.read()

# For LLM_REQUEST_END:
content = re.sub(
    r'(Hook\.LLM_REQUEST_END,\s*EventContext\([\s\S]*?)(model=ctx\.model_id,)(\s*\),)',
    r'\1\2\n                    metadata=getattr(response, "metadata", {}) or {},\3',
    content
)

# For Hook.AGENT_RUN_END in SingleShotLoop AND ReactLoop AND HumanInTheLoop:
# Wait, they end with stop_reason=...,
content = re.sub(
    r'(Hook\.AGENT_RUN_END,\s*EventContext\([\s\S]*?)(stop_reason=[^,]+,)(\s*\),)',
    r'\1\2\n                metadata=getattr(response, "metadata", {}) or {},\3',
    content
)

# Wait, check if some loops use variables other than `response` for completion result.
# Let's inspect the results before writing!
print(content[8000:9000])

with open('loop_patched.py', 'w') as f:
    f.write(content)
