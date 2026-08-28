#!/usr/bin/env python3
"""Fail-closed release gate for a paid central GT smoke dispatch.

This is deliberately narrower than the repository test suite.  It checks the
exact workflow that GitHub executes, invokes the census exactly as documented,
and exercises all 17 feature identities through the real agent loop with no
provider or task-container cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TESTS = (
    "tests/test_persistent_execution_state.py",
    "tests/test_gt_central_agent.py::test_persistent_state_bootstraps_once_then_runs_at_every_live_boundary",
    "tests/test_gt_central_agent.py::test_persistent_state_is_one_switch_and_off_audit_cannot_enable_it",
    "tests/test_central_release_gate.py::test_release_gate_rejects_bootstrap_only_or_silently_missing_living_state",
    "tests/test_central_release_gate.py::test_release_gate_rejects_missing_or_unwired_initial_hybrid_retrieval",
    "tests/test_gt_central_runtime.py::test_effect_timing_consumes_evidence_before_the_next_action",
    "tests/test_gt_central_runtime.py::test_documented_direct_census_entrypoint_is_executable",
    "tests/test_gt_central_agent.py::test_paid_engine_workflow_receives_exact_harbor_budget_without_new_limit",
    "tests/test_gt_central_agent.py::test_executable_completion_certificate_auto_submits_before_next_model_call",
    "tests/test_gt_central_agent.py::test_partial_completion_plan_executes_no_private_predicates",
    "tests/test_gt_central_agent.py::test_custom_probe_failure_is_not_reframed_as_model_guidance",
    "tests/test_gt_central_agent.py::test_context_transform_preserves_oversized_read_before_budget_pressure",
    "tests/test_gt_central_agent.py::test_context_soft_character_limit_starts_one_safe_compaction_epoch",
    "tests/test_gt_central_agent.py::test_execution_budget_reserve_exits_before_outer_timeout",
    "tests/test_gt_central_agent.py::test_provider_budget_failure_stops_before_model_query_and_is_receipted",
    "tests/test_gt_central_agent.py::test_over_budget_next_request_does_not_confirm_pending_guidance",
    "tests/test_gt_central_agent.py::test_redirected_declared_validator_receives_bounded_timeout_extension",
    "tests/test_gt_central_agent.py::test_failed_reader_does_not_consume_anchor_or_create_fallback_stall",
    "tests/test_gt_completion.py",
    "tests/test_gt_progress.py",
    "tests/test_harbor_budget.py",
    "tests/test_central_readiness.py",
    "tests/test_gt_central_agent.py::test_actual_agent_loop_routes_all_17_features_with_nonpredictive_effects",
    "tests/test_gt_feature_applicability_repair.py",
    "tests/test_gt_repository_intelligence.py",
    "tests/test_gt_repository_mirror.py",
    "tests/test_gt_intelligence_layer.py",
    "tests/test_gt_uplift_policy.py",
    "tests/test_gt_on_experiment.py",
    "tests/test_gt_language_resolution.py",
    "tests/test_gt_benchmark_language_contract.py",
    "tests/test_gt_central_agent.py::test_context_frontier_advances_repository_intelligence_without_feature_advisory",
    "tests/test_gt_central_agent.py::test_preemptive_hybrid_retrieval_reaches_exact_first_provider_request",
    "tests/test_gt_central_agent.py::test_action_conditioned_missing_evidence_returns_before_mutation_once",
    "tests/test_gt_central_agent.py::test_action_conditioned_decision_is_observation_only_in_shadow",
    "tests/test_gt_central_runtime.py::test_workspace_manifest_prunes_known_derived_trees_before_entry_bound",
    "tests/test_gt_central_runtime.py::test_workspace_manifest_command_is_shell_parseable_and_prunes_derived_trees",
    "tests/test_gt_central_runtime.py::test_sensor_recovery_from_unhealthy_snapshot_rehashes_all_source",
    "tests/test_gt_central_consumer_proof.py::test_disabled_task_start_reslot_closes_its_semantic_claim",
    "tests/test_decision_sufficiency.py",
    "tests/test_central_release_gate.py",
    "tests/test_gt_delivery_audit.py",
    "tests/test_preemptive_retrieval_frame.py",
    "tests/test_hybrid_retrieval.py",
    "tests/test_hybrid_repository.py",
    "tests/test_snowflake_onnx_backend.py",
    "tests/test_gt_central_agent.py::test_source_less_task_is_denominator_excluded_not_graph_invalid",
    "tests/test_gt_central_agent.py::test_task_graph_failure_degrades_but_preserves_provider_loop",
    "tests/test_gt_central_agent.py::test_paid_environment_path_transfers_only_selected_source_files",
    "tests/test_gt_central_agent.py::test_strict_graph_gate_allows_current_certified_graph",
    "tests/test_gt_central_agent.py::test_frontier_fact_is_one_shot_and_task_budget_is_receipted",
    "tests/test_gt_central_agent.py::test_proven_read_only_action_reuses_workspace_snapshot_without_rescan",
    "tests/test_gt_central_agent.py::test_grounded_failure_warns_before_submit_without_holding_it",
    "tests/test_gt_central_agent.py::test_syntax_failure_does_not_interrupt_multi_action_batch",
    "tests/test_gt_preflight.py::test_validation_classification_applies_only_to_runner_segment",
    "tests/test_gt_preflight.py::test_wrapped_go_test_uses_normalized_executable_arguments",
    "tests/test_gt_preflight.py::test_multiline_shell_commands_are_separate_top_level_segments",
    "tests/test_gt_preflight.py::test_heredoc_body_is_opaque_and_only_shell_operands_become_targets",
    "tests/test_gt_preflight.py::test_inline_interpreter_program_is_opaque_to_target_extraction",
    "tests/test_gt_preflight.py::test_sed_range_does_not_attach_across_non_pipeline_connector",
    "tests/test_gt_preflight.py::test_attached_output_redirection_is_classified_as_edit",
    "tests/test_gt_preflight.py::test_descriptor_redirect_does_not_hide_declared_validation_or_fake_mutation",
    "tests/test_gt_preflight.py::test_validator_with_file_redirect_retains_validation_and_records_side_effect",
    "tests/test_gt_preflight.py::test_input_redirection_is_a_typed_read_without_workspace_mutation",
    "tests/test_gt_preflight.py::test_spaced_descriptor_remains_an_argument_but_attached_descriptor_does_not",
    "tests/test_gt_preflight.py::test_absent_output_target_is_expected_creation_and_preflight_passes",
    "tests/test_gt_preflight.py::test_absent_in_place_edit_target_fails_open_to_shell_postflight",
    "tests/test_gt_trajectory_utilization.py",
    "tests/test_provider_view.py::test_compiler_proves_existing_read_fact_at_exact_provider_message",
    "tests/test_gt_central_runtime.py::test_declared_verifier_text_in_a_read_is_not_a_validation_result",
    "tests/test_gt_central_runtime.py::test_literal_timeout_wrapper_preserves_declared_check_authority",
    "tests/test_gt_central_runtime.py::test_declared_validator_identity_ignores_descriptor_redirection",
    "tests/test_gt_central_runtime.py::test_declared_validator_identity_ignores_file_output_redirection",
    "tests/test_gt_central_runtime.py::test_trailing_reporter_prevents_validator_status_attribution",
    "tests/test_gt_central_runtime.py::test_background_validation_is_pending_not_passing",
    "tests/test_gt_central_runtime.py::test_pipeline_reporter_does_not_borrow_validator_status_without_pipefail",
    "tests/test_gt_central_runtime.py::test_newfile_precedent_is_one_shot_per_task_not_per_created_filename",
    "tests/test_gt_central_runtime.py::test_classify_change_never_advances_source_for_artifacts",
    "tests/test_gt_central_runtime.py::test_generated_binary_data_cannot_trigger_validation_debt_after_source_edits",
    "tests/test_gt_central_runtime.py::test_task_deliverable_paths_extracts_contract_outputs",
    "tests/test_gt_central_runtime.py::test_task_deliverable_paths_uses_wrapped_output_context_and_rejects_inputs",
    "tests/test_gt_central_runtime.py::test_task_deliverable_paths_does_not_conflate_compressor_input_and_output",
    "tests/test_gt_central_consumer_proof.py::test_newfile_precedent_ranks_semantically_related_nonempty_sibling",
    "tests/test_gt_central_consumer_proof.py::test_newfile_precedent_abstains_when_only_sibling_is_empty_package_marker",
    "tests/test_central_replay.py::test_replay_does_not_require_a_certificate_for_unattributed_declared_pipeline",
    "tests/test_central_run_diff.py",
    "tests/test_provider_view.py::test_below_compaction_trigger_preserves_provider_messages_byte_for_byte",
    "tests/test_provider_view.py::test_compaction_never_removes_distinct_assistant_reasoning",
    "tests/test_provider_view.py::test_recent_oversized_observation_is_bounded_when_pressure_requires_transform",
    "tests/test_provider_view.py::test_duplicate_turn_is_represented_append_only_instead_of_deleting_history",
    "tests/test_provider_view.py::test_provider_request_budget_fails_closed_before_provider_overflow",
    "tests/test_provider_view.py::test_provider_compaction_trigger_is_based_on_measured_headroom",
    "tests/test_provider_view.py::test_provider_view_session_reuses_an_immutable_compacted_prefix",
    "tests/test_provider_view.py::test_provider_view_session_does_not_inject_unrepresented_current_failure",
    "tests/test_gt_central_runtime.py::test_semantic_source_revision_ignores_filesystem_timestamps",
    "tests/test_gt_central_runtime.py::test_semantic_source_revision_receipt_fails_closed_when_source_digest_is_missing",
    "tests/test_gt_central_runtime.py::test_sensor_caches_missing_python_capture_backend_across_source_edits",
    "tests/test_gt_central_consumer_proof.py::test_newfile_precedent_never_uses_a_model_created_sibling_as_repository_precedent",
    "tests/test_provider_evidence.py",
    "tests/test_replay_bundle.py",
    "tests/test_provider_view.py::test_compiler_canonicalizes_app_absolute_and_repository_relative_paths",
    "tests/test_gt_central_agent.py::test_disabled_task_start_advisory_never_leaks_into_call_two",
    "tests/test_gt_central_agent.py::test_receipt_hashes_the_provider_prepared_messages_not_private_extra",
    "tests/test_gt_central_agent.py::test_off_and_shadow_dispatch_identical_model_commands",
    "tests/test_gt_central_agent.py::test_integration_mode_is_one_switch_and_audit_cannot_intervene",
    "tests/test_gt_central_agent.py::test_certified_shadow_is_provider_neutral_and_cannot_run_active_controllers",
    "tests/test_gt_semantic_engine.py::test_over_budget_fact_is_omitted_whole_instead_of_truncated",
    "tests/test_gt_central_runtime.py::test_advisory_is_complete_or_quiet_never_truncated",
    "tests/test_gt_central_consumer_proof.py::test_context_compiler_accounts_every_effect_without_claiming_model_visibility",
    "tests/test_gt_deep_metrics.py::test_extract_trajectory_includes_outer_harbor_timeout_and_wall_time",
    "tests/test_gt_deep_metrics.py::test_outcome_first_gate_allows_small_per_task_variance_only_when_aggregate_wins",
    "tests/test_gt_deep_metrics.py::test_outcome_first_gate_rejects_two_large_per_task_resource_regressions",
    "tests/test_gt_deep_metrics.py::test_efficiency_aggregate_uses_only_common_uncensored_solves_and_separates_controller_work",
    "tests/test_gt_deep_metrics.py::test_outcome_gate_rejects_repository_intelligence_failure_even_if_resources_win",
    "tests/test_gt_deep_metrics.py::test_extract_trajectory_reports_provider_action_batching",
    "tests/test_gt_deep_metrics.py::test_efficiency_gate_rejects_positive_assistant_steps_and_effective_actions",
    "tests/test_gt_intelligence_layer.py::test_task_path_alone_does_not_expose_unrequested_generic_definition",
    "tests/test_gt_intelligence_layer.py::test_malformed_structural_symbol_is_rejected_before_provider_delivery",
    "tests/test_gt_host_execution.py",
    "tests/test_central_efficiency_replay.py",
)


def run(label: str, *command: str) -> bool:
    print(f"== {label} ==")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        print(f"FAILED: {label} (exit {completed.returncode})")
        # CI can truncate a large successful-test stream before the useful
        # failure summary.  Re-run only the failing gate with captured output
        # so the exact environment/collection error is replayable.
        if label == "strict agent lifecycle tests":
            diagnostic = subprocess.run(
                (*command, "--maxfail=1", "--tb=short"),
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
            tail = ((diagnostic.stdout or "") + (diagnostic.stderr or ""))[-12000:]
            if tail:
                print("STRICT_TEST_DIAGNOSTIC_BEGIN")
                print(tail)
                print("STRICT_TEST_DIAGNOSTIC_END")
        return False
    print(f"PASSED: {label}")
    return True


def exact_commit_is_pushed() -> bool:
    """Fail closed unless tracked files match the pushed workflow commit."""

    print("== exact pushed commit ==")
    dirty = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    local = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dispatch_sha = os.environ.get("GITHUB_SHA", "").strip()
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        approved = (
            dirty.returncode == 0
            and not dirty.stdout.strip()
            and bool(dispatch_sha)
            and bool(local)
            and local == dispatch_sha
        )
        print(f"detached_ci=true local={local[:12]} dispatch={dispatch_sha[:12]}")
        print("PASSED: exact pushed commit" if approved else "FAILED: exact pushed commit")
        return approved
    remote = subprocess.run(
        ("git", "rev-parse", f"origin/{branch}"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    approved = (
        dirty.returncode == 0
        and not dirty.stdout.strip()
        and bool(branch)
        and bool(local)
        and local == remote
    )
    print(f"branch={branch} local={local[:12]} origin={remote[:12]}")
    print("PASSED: exact pushed commit" if approved else "FAILED: exact pushed commit")
    return approved


def main() -> int:
    checks = (
        exact_commit_is_pushed(),
        run("strict agent lifecycle tests", sys.executable, "-m", "pytest", *RELEASE_TESTS, "-q"),
        run("documented direct census", sys.executable, "scripts/central_feature_census.py"),
        run("module census", sys.executable, "-m", "scripts.central_feature_census"),
        run(
            "repository intelligence substrate",
            sys.executable,
            "scripts/verify_gt_index_runtime.py",
        ),
        run(
            "pinned benchmark language contract",
            sys.executable,
            "scripts/verify_tb2_language_contract.py",
        ),
        run("workflow/readiness audit", sys.executable, "scripts/central_readiness_audit.py"),
    )
    if all(checks):
        print("SMOKE_APPROVED")
        return 0
    print("SMOKE_BLOCKED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
