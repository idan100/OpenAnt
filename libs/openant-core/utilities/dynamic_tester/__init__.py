"""Dynamic testing module for OpenAnt.

Takes pipeline_output.json from the static analysis pipeline and dynamically
tests all detected vulnerabilities using Docker containers.

Supports checkpoint/resume: each completed finding is saved to a per-unit
checkpoint file so interrupted runs can resume automatically.

Public API:
    run_dynamic_tests(pipeline_output_path, output_dir) -> list[DynamicTestResult]
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utilities.dynamic_tester.models import DynamicTestResult
from utilities.dynamic_tester.test_generator import generate_test, regenerate_test
from utilities.dynamic_tester.docker_executor import run_single_container
from utilities.dynamic_tester.result_collector import collect_result
from utilities.dynamic_tester.reporter import generate_report
from utilities.llm_client import get_global_tracker
from utilities.llm import (
    PhaseRegistry,
    build_phase_registry,
    load_config_file,
    resolve_llm_config,
)
from utilities.file_io import read_json, write_json, open_utf8


def _process_one_finding(
    index: int,
    total: int,
    finding: dict,
    finding_id: str,
    repo_info: dict,
    repo_path: str | None,
    dynamic_test_binding,
    tracker,
    max_retries: int,
) -> DynamicTestResult:
    """Generate + run (with error-feedback retries) the dynamic test for one
    finding. No shared mutable state besides ``tracker``, which is safe for
    concurrent callers: ``start_unit_tracking()``/``get_unit_usage()`` key off
    ``threading.local()``, so each worker thread gets its own accumulator.
    """
    print(f"\n[{index + 1}/{total}] Testing {finding_id}: "
          f"{finding.get('name', 'unknown')}...", file=sys.stderr)

    tracker.start_unit_tracking()

    print("  Generating test...", file=sys.stderr)
    generation = generate_test(finding, repo_info, dynamic_test_binding, tracker)
    unit_usage = tracker.get_unit_usage()
    generation_cost = unit_usage["cost_usd"]

    if generation is None:
        print("  Test generation failed.", file=sys.stderr)
        result = collect_result(finding, None, None, generation_cost)
        result.generation_input_tokens = unit_usage["input_tokens"]
        result.generation_output_tokens = unit_usage["output_tokens"]
        return result

    print(f"  Generated (${generation_cost:.4f}). Running in Docker...",
          file=sys.stderr)

    # Resolve the vulnerable source file for pre-staging.
    source_file = None
    if repo_path:
        rel_path = finding.get("location", {}).get("file", "")
        if rel_path:
            candidate = os.path.join(repo_path, rel_path)
            if os.path.isfile(candidate):
                source_file = candidate

    execution = run_single_container(generation, finding_id, source_file=source_file)
    result = collect_result(finding, generation, execution, generation_cost)
    retry_count = 0

    while result.status == "ERROR" and retry_count < max_retries:
        if execution.build_error:
            error_msg = execution.build_error
            error_type = "Build"
        elif execution.exit_code != 0 and execution.stderr:
            error_msg = execution.stderr
            error_type = "Runtime"
        else:
            error_msg = result.details
            error_type = "Application"

        if execution.timed_out:
            print(f"  Timed out — not retrying.", file=sys.stderr)
            break

        retry_count += 1
        print(f"  {error_type} error. Retry {retry_count}/{max_retries} "
              f"with error feedback...", file=sys.stderr)

        retry_gen = regenerate_test(
            finding, repo_info, generation,
            error_msg, dynamic_test_binding, tracker,
        )
        unit_usage = tracker.get_unit_usage()
        generation_cost = unit_usage["cost_usd"]

        if retry_gen is None:
            print(f"  Retry generation failed.", file=sys.stderr)
            break

        generation = retry_gen
        execution = run_single_container(generation, finding_id, source_file=source_file)
        result = collect_result(finding, generation, execution, generation_cost)
        print(f"  Retry {retry_count} result: {result.status} "
              f"(${generation_cost:.4f})", file=sys.stderr)

    result.retry_count = retry_count
    result.generation_input_tokens = unit_usage["input_tokens"]
    result.generation_output_tokens = unit_usage["output_tokens"]

    print(f"  Result: {result.status} ({result.elapsed_seconds:.1f}s)",
          file=sys.stderr)
    return result


def run_dynamic_tests(
    pipeline_output_path: str,
    output_dir: str | None = None,
    max_retries: int = 3,
    checkpoint_path: str | None = None,
    repo_path: str | None = None,
    registry: PhaseRegistry | None = None,
    llm_config_name: str | None = None,
    workers: int = 4,
) -> list[DynamicTestResult]:
    """Run dynamic tests for all findings in a pipeline output file.

    Args:
        pipeline_output_path: Path to pipeline_output.json
        output_dir: Directory for output files. Defaults to same directory
                    as pipeline_output_path.
        max_retries: Max retries per finding on error (default 3).
        checkpoint_path: Path to checkpoint directory for resume support.
        repo_path: Path to the repository root. When given, the vulnerable
            source file is pre-staged into the Docker build context so
            ``COPY <filename> .`` works on the first try.
        workers: Number of findings to test in parallel (default 4). Each
            finding gets its own uniquely-named Docker image/network/work
            dir (see docker_executor.run_single_container), so concurrent
            runs don't collide. Lower than the 8 used for pure-API phases
            (enhance/analyze/verify) since Docker build+run is bound by the
            host's CPU/disk, not just network latency — set to 1 for the
            old strictly-sequential behavior.

    Returns:
        List of DynamicTestResult objects
    """
    # Resolve the dynamic_test phase binding from the registry once and
    # reuse it across every finding. Standalone-invocation path
    # validates upfront; scanner-driven calls trust the scanner's probe.
    if registry is None:
        from utilities.llm import probe_registry_or_raise

        cf = load_config_file()
        registry = build_phase_registry(cf, resolve_llm_config(cf, llm_config_name))
        probe_registry_or_raise(registry)
    dynamic_test_binding = registry.get("dynamic_test")

    # Load pipeline output
    pipeline = read_json(pipeline_output_path)
    findings = pipeline.get("findings", [])
    repo_info = {
        "name": pipeline.get("repository", {}).get("name", "unknown"),
        "language": pipeline.get("repository", {}).get("language", "Python"),
        "application_type": pipeline.get("application_type", "unknown"),
    }

    if not findings:
        print("No findings to test.", file=sys.stderr)
        return []

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pipeline_output_path))
    os.makedirs(output_dir, exist_ok=True)

    # Set up checkpoint support
    checkpoint = None
    checkpointed = {}
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "dynamic_test_checkpoints")

    from core.checkpoint import StepCheckpoint
    checkpoint = StepCheckpoint("dynamic_test", output_dir)
    checkpoint.dir = checkpoint_path
    if checkpoint.exists:
        checkpointed = checkpoint.load()

    # Count successful vs errored checkpoints. Errored ones are NOT "already
    # done" — they'll be retried with fresh test generation on resume.
    successful_ids = {fid for fid, cp in checkpointed.items()
                      if cp.get("status") != "ERROR"}
    errored_ids = {fid for fid in checkpointed.keys() if fid not in successful_ids}

    if successful_ids:
        print(f"Restored {len(successful_ids)} already-tested findings from checkpoints",
              file=sys.stderr, flush=True)
    if errored_ids:
        print(f"Retrying {len(errored_ids)} previously errored findings",
              file=sys.stderr, flush=True)

    # Use the global tracker so step_context captures dynamic-test cost in
    # dynamic-test.report.json (same as enhance/analyze/verify).
    tracker = get_global_tracker()

    # Inject prior usage from ALL existing checkpoints (both successful and
    # errored) so the report shows total cost across runs. The errored
    # entries will be retried — their initial attempt cost is preserved,
    # and the retry API calls get added on top.
    _prior_input = 0
    _prior_output = 0
    _prior_cost = 0.0
    for _cp in checkpointed.values():
        _prior_cost += _cp.get("generation_cost_usd", 0) or 0
        _prior_input += _cp.get("generation_input_tokens", 0) or 0
        _prior_output += _cp.get("generation_output_tokens", 0) or 0
    if _prior_cost > 0 or _prior_input > 0 or _prior_output > 0:
        tracker.add_prior_usage(_prior_input, _prior_output, _prior_cost)

    total = len(findings)
    restored = len(successful_ids)
    remaining = total - restored

    # Pre-populate results from checkpoints (restored, in original order);
    # collect the rest as (index, finding, finding_id) work items.
    results: list = [None] * total
    to_process: list[tuple[int, dict, str]] = []
    for i, finding in enumerate(findings):
        finding_id = finding.get("id", f"FINDING-{i + 1}")
        cp_data = checkpointed.get(finding_id)
        if cp_data and cp_data.get("status") != "ERROR":
            results[i] = DynamicTestResult(
                finding_id=finding_id,
                status=cp_data.get("status", "ERROR"),
                details=cp_data.get("details", ""),
                elapsed_seconds=cp_data.get("elapsed_seconds", 0),
                generation_cost_usd=cp_data.get("generation_cost_usd", 0),
                generation_input_tokens=cp_data.get("generation_input_tokens", 0),
                generation_output_tokens=cp_data.get("generation_output_tokens", 0),
                retry_count=cp_data.get("retry_count", 0),
                test_code=cp_data.get("test_code", ""),
                dockerfile=cp_data.get("dockerfile", ""),
                docker_compose=cp_data.get("docker_compose", ""),
            )
        else:
            to_process.append((i, finding, finding_id))

    _completed = restored
    _errors = 0
    _summary_lock = threading.Lock()

    # Write initial summary so Go CLI can show accurate counts
    checkpoint.ensure_dir()
    checkpoint.write_summary(total, _completed, _errors, {}, phase="in_progress")

    print(f"Dynamic testing {total} findings from {repo_info['name']} "
          f"({restored} already done, {remaining} remaining)",
          file=sys.stderr)

    def _record(index: int, finding_id: str, result: DynamicTestResult) -> None:
        """Store a finished finding's result, save its checkpoint, and
        update/publish the running summary. Safe to call from any worker
        thread — the only shared mutable state (_completed/_errors) is
        guarded by _summary_lock."""
        nonlocal _completed, _errors
        results[index] = result
        with _summary_lock:
            checkpoint.save(finding_id, result.to_dict())
            _completed += 1
            if result.status == "ERROR":
                _errors += 1
            checkpoint.write_summary(total, _completed, _errors, {}, phase="in_progress")

    try:
        if workers <= 1:
            for i, finding, finding_id in to_process:
                result = _process_one_finding(
                    i, total, finding, finding_id, repo_info, repo_path,
                    dynamic_test_binding, tracker, max_retries,
                )
                _record(i, finding_id, result)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            futures = {
                executor.submit(
                    _process_one_finding,
                    i, total, finding, finding_id, repo_info, repo_path,
                    dynamic_test_binding, tracker, max_retries,
                ): (i, finding_id)
                for i, finding, finding_id in to_process
            }
            try:
                for future in as_completed(futures):
                    i, finding_id = futures[future]
                    _record(i, finding_id, future.result())
            except KeyboardInterrupt:
                print("\n[Dynamic Test] Interrupted — cancelling pending work...",
                      file=sys.stderr, flush=True)
                executor.shutdown(wait=False, cancel_futures=True)
                print("[Dynamic Test] Progress saved to checkpoints",
                      file=sys.stderr, flush=True)
                return [r for r in results if r is not None]
            executor.shutdown(wait=False)
    except KeyboardInterrupt:
        print("\n[Dynamic Test] Interrupted — progress saved to checkpoints",
              file=sys.stderr, flush=True)
        return [r for r in results if r is not None]

    # Generate report
    total_cost = tracker.total_cost_usd
    report_md = generate_report(results, repo_info["name"], total_cost)

    report_path = os.path.join(output_dir, "DYNAMIC_TEST_RESULTS.md")
    with open_utf8(report_path, "w") as f:
        f.write(report_md)
    print(f"\nReport written to {report_path}", file=sys.stderr)

    # Save structured results JSON
    results_path = os.path.join(output_dir, "dynamic_test_results.json")
    with open_utf8(results_path, "w") as f:
        json.dump({
            "repository": repo_info["name"],
            "total_findings": len(findings),
            "total_cost_usd": round(total_cost, 6),
            "results": [r.to_dict() for r in results],
        }, f, indent=2, ensure_ascii=False)
    print(f"Results JSON written to {results_path}", file=sys.stderr)

    # Mark done. Checkpoints are preserved as a permanent artifact alongside
    # results — allows retroactive retry of errored findings after fixes.
    checkpoint.write_summary(total, _completed, _errors, {}, phase="done")

    return results
