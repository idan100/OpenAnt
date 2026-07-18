"""Shared fixtures for OpenAnt tests."""
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so imports like `from utilities...` work
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Several test files defensively stub ``sys.modules["anthropic"]`` if the
# real SDK isn't loaded yet (a legacy guard from before anthropic became a
# hard dep). Now that core modules no longer eagerly import the SDK at
# module load (issue #65 moved provider IO behind the adapter layer), the
# first-loaded test that runs the guard would install the stub — and then
# the Anthropic adapter contract tests fail with ``Cannot spec a Mock
# object``. Claim the slot here with the real module so the guards become
# no-ops. The SDK is in requirements.txt so the import is guaranteed to
# succeed in any environment that runs the test suite.
import anthropic  # noqa: F401,E402 — see comment above

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PYTHON_REPO = FIXTURES_DIR / "sample_python_repo"
SAMPLE_JS_REPO = FIXTURES_DIR / "sample_js_repo"


@pytest.fixture
def sample_python_repo():
    """Path to the sample Python repository fixture."""
    return str(SAMPLE_PYTHON_REPO)


@pytest.fixture
def sample_js_repo():
    """Path to the sample JavaScript repository fixture."""
    return str(SAMPLE_JS_REPO)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for parser results."""
    return str(tmp_path / "output")


@pytest.fixture(autouse=True)
def _isolate_probe_cache(tmp_path, monkeypatch):
    """Never let any test read/write the real ~/.config/openant/probe_cache.json.

    PhaseRegistry.validate() consults this cache (see
    utilities/llm/probe_cache.py) to skip re-probing a recently-validated
    (adapter, model) pair. Without this fixture, any test that calls
    validate() with a real (unmocked) cache path pollutes the actual
    user's on-disk cache with fake test provider/model names — and,
    worse, cross-contaminates OTHER tests within the same pytest run:
    whichever test happens to run first writes e.g. "anthropic:m" to
    the real file, and every later test using that same fake pair then
    sees "recently validated" and silently skips its own validate()
    call, breaking assertions that expected it to actually run.
    """
    from utilities.llm import probe_cache

    fake_path = tmp_path / "probe_cache.json"
    monkeypatch.setattr(probe_cache, "_cache_path", lambda: fake_path)
