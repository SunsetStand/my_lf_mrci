# MR_LF learning environment

Use WSL2 (`ubuntu-d`) and open `/mnt/d/code/MR_LF` in VS Code.
The existing `.venv` belongs to Windows Python 3.9.10; leave it untouched.
Linux interpreter: `/home/enovo/.venvs/mr_lf/bin/python` (Python 3.12.3).
`requirements.txt` records the installed direct and transitive package versions.

```bash
cd /mnt/d/code/MR_LF
source /home/enovo/.venvs/mr_lf/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
ruff check --no-cache tests/test_environment.py
ruff check --no-cache src tests
```

In VS Code, run `Python: Select Interpreter` and select the Linux interpreter
above if a previous workspace selection overrides the default setting.
The student's initial `src/my_direct_ep.py` contains an unused NumPy import;
the full Ruff check may report F401 until that import is used.

Tests cover environment imports, cistring enumeration/link semantics and a
headless plot. They do not validate any Hubbard–Holstein model implementation.
The private `_unpack_nelec` helper is probed only for compatibility.
