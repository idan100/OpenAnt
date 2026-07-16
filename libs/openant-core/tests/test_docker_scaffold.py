"""Regression tests for Dockerfile scaffold pre-staging.

The dynamic-test scaffold must stage the vulnerable source file into the
Docker build context BEFORE asking the LLM to write the Dockerfile, so
`COPY VulnerablePythonScript.py .` works on the first try.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CORE_ROOT))

if "anthropic" not in sys.modules:
    _stub = types.ModuleType("anthropic")
    _stub.Anthropic = MagicMock()
    _stub.RateLimitError = type("RateLimitError", (Exception,), {})
    _stub.AuthenticationError = type("AuthenticationError", (Exception,), {})
    sys.modules["anthropic"] = _stub


def _fake_registry():
    """Build a PhaseRegistry whose adapter never probes the network.

    The orchestrator tests below mock ``generate_test`` and
    ``run_single_container`` so the adapter is never actually called.
    But ``run_dynamic_tests`` still builds a registry when none is
    passed in, which probes Anthropic at startup. Pre-issue-#65 the
    test relied on an ``ANTHROPIC_API_KEY`` happening to be in env;
    that's no longer reliable. Injecting a fake registry removes the
    env dependency entirely.
    """
    from utilities.llm import PhaseBinding, PhaseRegistry

    class _NoopAdapter:
        name = "anthropic"
        supports_tools = True

        def complete(self, **kwargs):  # pragma: no cover - mocked away
            raise AssertionError("orchestrator tests should not reach the adapter")

        def validate(self, model):
            pass

    adapter = _NoopAdapter()
    bindings = {
        phase: PhaseBinding(
            phase=phase,
            adapter=adapter,
            model="test-model",
            provider_name="anthropic",
        )
        for phase in ("analyze", "enhance", "verify", "report", "dynamic_test", "llm_reach", "app_context")
    }
    return PhaseRegistry(bindings=bindings, config_name="docker-test-config")


def test_write_test_files_stages_source(tmp_path):
    """_write_test_files must copy the vulnerable source into the work dir."""
    from utilities.dynamic_tester.docker_executor import _write_test_files

    # Create a fake source file to stage
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    source = repo_dir / "app.py"
    source.write_text("def vuln(): pass")

    generation = {
        "dockerfile": "FROM python:3.11\nCOPY app.py .\nCMD python app.py",
        "test_script": "print('test')",
        "test_filename": "test_exploit.py",
        "requirements": "flask",
    }

    finding = {
        "location": {"file": "app.py", "function": "app.py:vuln"},
    }

    work_dir = str(tmp_path / "work")
    os.makedirs(work_dir)

    _write_test_files(work_dir, generation, source_file=str(source))

    staged = os.path.join(work_dir, "app.py")
    assert os.path.exists(staged), "source file must be staged into work_dir"
    assert open(staged).read() == "def vuln(): pass"


def test_write_test_files_works_without_source(tmp_path):
    """Backward compat: _write_test_files must not fail when no source_file is given."""
    from utilities.dynamic_tester.docker_executor import _write_test_files

    generation = {
        "dockerfile": "FROM python:3.11\nCMD echo hi",
        "test_script": "print('test')",
        "test_filename": "test_exploit.py",
    }

    work_dir = str(tmp_path / "work")
    os.makedirs(work_dir)

    # Must not raise
    _write_test_files(work_dir, generation)


# ---------------------------------------------------------------------------
# Link 3: orchestrator resolves source_file and passes it to run_single_container
# ---------------------------------------------------------------------------

def test_orchestrator_passes_source_file(tmp_path, monkeypatch):
    """run_dynamic_tests must resolve source_file from repo_path + finding.location.file
    and pass it through to run_single_container."""
    import json

    # Create a fake repo with a source file
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def vuln(): pass")

    # Create a minimal pipeline_output.json
    po = {
        "repository": {"name": "test", "language": "python"},
        "application_type": "web_app",
        "findings": [{
            "id": "VULN-001",
            "name": "test vuln",
            "short_name": "vuln",
            "location": {"file": "app.py", "function": "app.py:vuln"},
            "cwe_id": 79,
            "cwe_name": "XSS",
            "stage1_verdict": "vulnerable",
            "stage2_verdict": "confirmed",
        }],
    }
    po_path = tmp_path / "pipeline_output.json"
    po_path.write_text(json.dumps(po))

    # Track what run_single_container receives
    captured_kwargs = {}

    def mock_generate_test(finding, repo_info, binding, tracker):
        return {
            "dockerfile": "FROM python:3.11\nCMD echo hi",
            "test_script": "print('ok')",
            "test_filename": "test_exploit.py",
        }

    def mock_run_single_container(generation, finding_id, source_file=None, **kwargs):
        captured_kwargs["source_file"] = source_file
        from utilities.dynamic_tester.docker_executor import DockerExecutionResult
        result = DockerExecutionResult()
        result.stdout = '{"status": "CONFIRMED", "details": "test", "evidence": []}'
        result.exit_code = 0
        return result

    monkeypatch.setattr("utilities.dynamic_tester.generate_test", mock_generate_test)
    monkeypatch.setattr("utilities.dynamic_tester.run_single_container", mock_run_single_container)

    from utilities.dynamic_tester import run_dynamic_tests
    run_dynamic_tests(
        pipeline_output_path=str(po_path),
        output_dir=str(tmp_path / "out"),
        max_retries=0,
        repo_path=str(repo),
        registry=_fake_registry(),
    )

    assert captured_kwargs.get("source_file") is not None, (
        "orchestrator must pass source_file to run_single_container"
    )
    assert captured_kwargs["source_file"].endswith("app.py")
    assert os.path.isfile(captured_kwargs["source_file"])


def test_orchestrator_works_without_repo_path(tmp_path, monkeypatch):
    """Backward compat: when repo_path is None, source_file should be None."""
    import json

    po = {
        "repository": {"name": "test", "language": "python"},
        "application_type": "web_app",
        "findings": [{
            "id": "VULN-001",
            "name": "test",
            "short_name": "vuln",
            "location": {"file": "app.py", "function": "app.py:vuln"},
            "cwe_id": 79,
            "cwe_name": "XSS",
            "stage1_verdict": "vulnerable",
            "stage2_verdict": "confirmed",
        }],
    }
    po_path = tmp_path / "pipeline_output.json"
    po_path.write_text(json.dumps(po))

    captured_kwargs = {}

    def mock_generate_test(finding, repo_info, binding, tracker):
        return {
            "dockerfile": "FROM python:3.11\nCMD echo hi",
            "test_script": "print('ok')",
            "test_filename": "test_exploit.py",
        }

    def mock_run_single_container(generation, finding_id, source_file=None, **kwargs):
        captured_kwargs["source_file"] = source_file
        from utilities.dynamic_tester.docker_executor import DockerExecutionResult
        result = DockerExecutionResult()
        result.stdout = '{"status": "CONFIRMED", "details": "test", "evidence": []}'
        result.exit_code = 0
        return result

    monkeypatch.setattr("utilities.dynamic_tester.generate_test", mock_generate_test)
    monkeypatch.setattr("utilities.dynamic_tester.run_single_container", mock_run_single_container)

    from utilities.dynamic_tester import run_dynamic_tests
    run_dynamic_tests(
        pipeline_output_path=str(po_path),
        output_dir=str(tmp_path / "out"),
        max_retries=0,
        registry=_fake_registry(),
    )

    assert captured_kwargs.get("source_file") is None, (
        "without repo_path, source_file must be None (backward compat)"
    )


# ---------------------------------------------------------------------------
# Link 4 + prompt: existing tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parallel execution (workers > 1)
# ---------------------------------------------------------------------------


def _multi_finding_pipeline_output(n: int) -> dict:
    return {
        "repository": {"name": "test", "language": "python"},
        "application_type": "web_app",
        "findings": [
            {
                "id": f"VULN-{i:03d}",
                "name": f"test vuln {i}",
                "short_name": "vuln",
                "location": {"file": "app.py", "function": "app.py:vuln"},
                "cwe_id": 79,
                "cwe_name": "XSS",
                "stage1_verdict": "vulnerable",
                "stage2_verdict": "confirmed",
            }
            for i in range(n)
        ],
    }


def _install_ok_mocks(monkeypatch, seen_finding_ids):
    """Mock generate_test/run_single_container to succeed and record which
    finding_id each call was for, thread-safely (list.append is atomic
    under the GIL)."""

    def mock_generate_test(finding, repo_info, binding, tracker):
        return {
            "dockerfile": "FROM python:3.11\nCMD echo hi",
            "test_script": "print('ok')",
            "test_filename": "test_exploit.py",
        }

    def mock_run_single_container(generation, finding_id, source_file=None, **kwargs):
        seen_finding_ids.append(finding_id)
        from utilities.dynamic_tester.docker_executor import DockerExecutionResult
        result = DockerExecutionResult()
        result.stdout = '{"status": "CONFIRMED", "details": "test", "evidence": []}'
        result.exit_code = 0
        return result

    monkeypatch.setattr("utilities.dynamic_tester.generate_test", mock_generate_test)
    monkeypatch.setattr("utilities.dynamic_tester.run_single_container", mock_run_single_container)


def test_parallel_workers_process_every_finding_exactly_once(tmp_path, monkeypatch):
    """workers>1 must still test every finding exactly once, with results
    in the same order as the input findings list."""
    import json

    n = 6
    po = _multi_finding_pipeline_output(n)
    po_path = tmp_path / "pipeline_output.json"
    po_path.write_text(json.dumps(po))

    seen = []
    _install_ok_mocks(monkeypatch, seen)

    from utilities.dynamic_tester import run_dynamic_tests
    results = run_dynamic_tests(
        pipeline_output_path=str(po_path),
        output_dir=str(tmp_path / "out"),
        max_retries=0,
        registry=_fake_registry(),
        workers=4,
    )

    assert len(results) == n
    assert [r.finding_id for r in results] == [f"VULN-{i:03d}" for i in range(n)], (
        "results must preserve input order even when processed concurrently"
    )
    assert sorted(seen) == sorted(f"VULN-{i:03d}" for i in range(n)), (
        "every finding must be tested exactly once"
    )
    assert all(r.status == "CONFIRMED" for r in results)


def test_workers_one_matches_parallel_results(tmp_path, monkeypatch):
    """workers=1 (sequential) and workers>1 (parallel) must produce the
    same set of results for the same input — parallelism is an
    implementation detail, not an observable behavior change."""
    import json

    n = 4
    po = _multi_finding_pipeline_output(n)
    po_path = tmp_path / "pipeline_output.json"
    po_path.write_text(json.dumps(po))

    seen = []
    _install_ok_mocks(monkeypatch, seen)

    from utilities.dynamic_tester import run_dynamic_tests
    results = run_dynamic_tests(
        pipeline_output_path=str(po_path),
        output_dir=str(tmp_path / "out"),
        max_retries=0,
        registry=_fake_registry(),
        workers=1,
    )

    assert [r.finding_id for r in results] == [f"VULN-{i:03d}" for i in range(n)]
    assert all(r.status == "CONFIRMED" for r in results)


def test_finding_prompt_includes_source_basename():
    """_build_finding_prompt must tell the LLM the staged filename."""
    from utilities.dynamic_tester.test_generator import _build_finding_prompt

    finding = {
        "id": "VULN-001",
        "name": "Command Injection",
        "cwe_id": 78,
        "cwe_name": "Command Injection",
        "location": {"file": "VulnerablePythonScript.py", "function": "ping"},
        "stage1_verdict": "vulnerable",
        "stage2_verdict": "agreed",
        "vulnerable_code": "def ping(): ...",
    }
    repo_info = {"name": "test", "language": "python", "application_type": "web_app"}

    prompt = _build_finding_prompt(finding, repo_info)
    assert "VulnerablePythonScript.py" in prompt, (
        "prompt must mention the staged source filename so the LLM references it in COPY"
    )
