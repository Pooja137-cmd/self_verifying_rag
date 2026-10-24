from pathlib import Path

p = Path("src/agents/nli.py")
t = p.read_text(encoding="utf-8")
a = 'DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"'
assert a in t and "import numpy as np\n" in t, "nli.py differs from expected"
t = t.replace("import numpy as np\n", "import os\n\nimport numpy as np\n", 1)
t = t.replace(a, 'DEFAULT_NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")')
p.write_text(t, encoding="utf-8")
print("patched OK")
