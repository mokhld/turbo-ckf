# Publishing

This project publishes to TestPyPI and PyPI via GitHub Actions using trusted publishing (OIDC).

## Security Rules

- Do not store API tokens in this repository.
- Do not commit `.pypirc`.
- Publish only from tagged commits on `main`.
- Keep GitHub environments `testpypi` and `pypi` protected with reviewer approval.

## One-Time Setup

1. In TestPyPI, configure a trusted publisher for:
   - Owner: `mokhld`
   - Repository: `turbo-ckf`
   - Workflow: `release.yml`
   - Environment: `testpypi`
2. In PyPI, configure a trusted publisher for:
   - Owner: `mokhld`
   - Repository: `turbo-ckf`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In GitHub repo settings, create environments:
   - `testpypi`
   - `pypi`
4. Add required reviewers to `pypi` environment before first production publish.

## Release Flow

1. Bump version in:
   - `pyproject.toml`
   - `Cargo.toml`
2. Commit and push.
3. Create and push a tag:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
4. Run release workflow:
   - For TestPyPI dry run: `workflow_dispatch` with `repository=testpypi`
   - For PyPI publish: create GitHub release from the tag, or run `workflow_dispatch` with `repository=pypi`

## Local Preflight (Optional)

```bash
python -m pip install -U pip maturin twine
python -m maturin build --release -o dist
python -m twine check dist/*
```
