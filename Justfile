[unix]
set shell := ["bash", "-euo", "pipefail", "-c"]

[windows]
set shell := ["cmd.exe", "/c"]

set dotenv-load := true

[default]
@_:
    just --list

[doc("Update the `detect-secrets` baseline in place (keeps audit results)")]
[group("DEV")]
secrets-baseline:
    @echo NOTE: re-run after intentionally adding secrets, e.g. test fixtures, then audit new results.
    uv run detect-secrets scan --baseline .secrets.baseline

[doc("Setup development environment")]
[group("DEV")]
setup: && secrets-baseline
    uv sync --all-extras
    uv run -m prek install

[doc("Create `coverage` report")]
[group("DEV")]
coverage *FLAGS:
    @uv run coverage run -m pytest {{FLAGS}}
    @uv run coverage report --show-missing --skip-covered --fail-under=75
    @uv run coverage xml
    @uv run coverage html

[doc("Git prune (aggressive)")]
[group("DEV")]
git-prune:
    git gc --prune=now --aggressive

[doc("Symlink `AGENTS.md` -> `CLAUDE.md`")]
[group("DEV")]
symlink-agents:
    @uv run python -c "import pathlib; p=pathlib.Path('CLAUDE.md'); p.unlink(missing_ok=True); p.symlink_to('AGENTS.md')"

[doc("Build Sphinx HTML documentation (warnings treated as errors)")]
[group("DOCS")]
docs:
    uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html

[doc("Serve documentation locally with live-reload on changes")]
[group("DOCS")]
docs-serve:
    uv run --extra docs sphinx-autobuild docs/source docs/build/html --open-browser

[doc("Remove built documentation artifacts")]
[group("DOCS")]
docs-clean:
    @uv run python -c "import shutil; shutil.rmtree('docs/build', ignore_errors=True)"
