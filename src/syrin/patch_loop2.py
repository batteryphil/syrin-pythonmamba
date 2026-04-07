import re

with open('loop.py', 'r') as f:
    content = f.read()

# For LLM_REQUEST_END
content = re.sub(
    r'(Hook\.LLM_REQUEST_END,\s*EventContext\([\s\S]*?)(model=[^,]+,)(\s*\),)',
    r'\1\2\n                    metadata=getattr(response, "metadata", {}) or {},\3',
    content
)

# For AGENT_RUN_END
content = re.sub(
    r'(Hook\.AGENT_RUN_END,\s*EventContext\([\s\S]*?)(stop_reason=[\s\S]*?,)(\s*\),)',
    r'\1\2\n                metadata=getattr(response, "metadata", {}) or {},\3',
    content
)

with open('loop_patched.py', 'w') as f:
    f.write(content)
