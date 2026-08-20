# Wheel leak guard (build-time boundary)

**What**: A build-time and CI guarantee that the closed-source
recommendation engine (`server/` and `engine_core`) never ships inside the
PyPI wheel of `robot-data-audit`.

**Why it matters**: RDA's Path B commercial model is "audit logic is
open-source and visible in the wheel; recommendation logic is closed-source
and lives on the server." If `server/` accidentally gets packaged, the
closed-source boundary collapses and the model is voided. This is a
single-line-of-config mistake that's easy to make and catastrophic to
recover from (you can't un-publish a PyPI release, only yank it).

## The two layers of defense

### 1. Packaging config (static)

[`pyproject.toml`](../pyproject.toml):

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["rda*"]
exclude = ["server*"]
```

Only `rda/` and its subpackages are included. `server/` is excluded by
name. This is the primary boundary.

### 2. CI assertion (runtime guard)

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs a post-build
check on every push:

```python
import glob, zipfile
wheel = [w for w in glob.glob('dist/*.whl') if 'robot_data_audit' in w][0]
names = zipfile.ZipFile(wheel).namelist()
leaked = [n for n in names if n.startswith('server/') or 'engine_core' in n]
assert not leaked, leaked
```

If anyone ever adds `server` to the `include` list (or renames it in a way
that matches), the CI goes red. The assertion also catches the subtler
case: a stray `import` that pulls `engine_core` into the wheel as a
dependency artifact.

## Verification (v0.5.3 wheel)

Build: `python -m build` → `dist/robot_data_audit-0.5.3-py3-none-any.whl`

| Check | Result |
|---|---|
| Total files in wheel | 53 |
| `server/` paths in wheel | 0 |
| `engine_core` in any path | 0 |
| `rda/` module files | 47 |

```bash
# reproduce locally
python -m build
python -c "
import glob, zipfile
w = [x for x in glob.glob('dist/*.whl') if 'robot_data_audit' in x][0]
n = zipfile.ZipFile(w).namelist()
print('server/ leaked:', [x for x in n if x.startswith('server/')])
print('engine_core leaked:', [x for x in n if 'engine_core' in x])
"
```

## What this does NOT prove

- It proves the *packaged artifact* doesn't contain the closed source. It
  does **not** prove the open-source `rda/` code can't *call* a closed
  endpoint — it can, that's the design (`api_client.py` is open, the server
  it calls is closed).
- It doesn't prove the server-side engine matches what's described in
  [`server_deploy_verify.md`](./server_deploy_verify.md). The server is
  self-hosted by the author; users who self-deploy build from their own
  `server/` checkout.

## Backing artifact

- Config: [`pyproject.toml`](../pyproject.toml)
- CI job: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — the
  "wheel leak guard" step.
- The check is also runnable locally (command above) and was run ad-hoc
  during every release from v0.4.x onward.
