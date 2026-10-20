"""Structured recovery, trajectory and repair diagnostics for paper analysis.

Structured events identify local admission, environment execution, and final
task success as distinct stages.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import (digest, first_divergence, origin_metadata,
                       response_usage, _natural_attempts)

GROUP = ("benchmark", "model_id", "interface", "policy", "analysis_configuration_hash", "prompts_hash")
IDENTITY = GROUP + ("configuration_hash", "task_id", "episode_id", "replicate_id")
RECOVERY_CATEGORIES = ("semantic_choice_changed", "corrected_admitted", "same_error_repeated",
                       "different_failure", "termination_no_next_attempt", "unresolved")
PHASES = ("ordinary", "commitment", "realization", "correction", "fallback", "unclassified")


def _calls(bundle: Mapping[str, Any]) -> dict[str, dict]:
    calls = bundle.get("_cost_calls", bundle.get("model_call_rows", []))
    indexed = {c["call_id"]: c for c in calls}
    if len(indexed) != len(calls):
        raise ValueError("duplicate cost-ledger call")
    return indexed


def attempt_usage(bundle: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]) -> dict:
    """Count unique logical calls and retain unknown call assignments."""
    ledger, assigned, missing = _calls(bundle), [], 0
    all_references = [call_id for row in bundle.get("attempt_rows", []) for call_id in row.get("model_call_ids", [])]
    if len(all_references) != len(set(all_references)):
        raise ValueError("a logical call is assigned to multiple attempts")
    coordinates = {call_id: row["attempt_id"] for row in bundle.get("attempt_rows", [])
                   for call_id in row.get("model_call_ids", [])}
    coordinates.update({c["call_id"]: c["attempt_id"] for c in bundle.get("model_call_rows", []) if "attempt_id" in c})
    coordinates.update({key: call["attempt_id"] for key, call in ledger.items() if "attempt_id" in call})
    for row in attempts:
        if "model_call_ids" not in row:
            missing += 1
        else:
            assigned.extend(row["model_call_ids"])
    if len(assigned) != len(set(assigned)):
        raise ValueError("a logical call is assigned to multiple attempts")
    selected_attempts = {a["attempt_id"] for a in attempts if "attempt_id" in a}
    assigned = sorted(set(assigned) | {c for c, attempt_id in coordinates.items() if attempt_id in selected_attempts})
    unassigned_ledger = set(ledger) - coordinates.keys()
    absent = set(assigned) - ledger.keys()
    usage = response_usage([ledger[c] for c in assigned if c in ledger])
    usage["unassigned_attempts"] = missing
    usage["missing_call_records"] = len(absent)
    usage["unassigned_ledger_calls"] = len(unassigned_ledger)
    if missing or absent or unassigned_ledger:
        usage.update(response_tokens=None, input_tokens=None, output_tokens=None,
                     model_calls=None, usage_coverage=None)
    return usage


def recovery_roots(bundle: Mapping[str, Any], *, final_success: bool | None) -> list[dict]:
    sequence = _natural_attempts(bundle)
    by_decision = defaultdict(list)
    for row in sequence:
        by_decision[row["decision_id"]].append(row)
    result = []
    for decision, attempts in sorted(by_decision.items()):
        root, failure = attempts[0], attempts[0].get("failure") or {}
        if root.get("retry_index") != 0 or not failure:
            continue
        eligible = (failure.get("retryable") is True and failure.get("obligation_owner") == "policy_output"
                    and failure.get("transition_status") == "unchanged")
        if not eligible:
            continue
        successor = attempts[1] if len(attempts) > 1 else None
        changed = (root["semantic_key"] != successor["semantic_key"]
                   if successor and root["semantic_key"] is not None and successor["semantic_key"] is not None else None)
        next_failure = (successor.get("failure") or {}) if successor else {}
        category = ("termination_no_next_attempt" if successor is None else
                    "semantic_choice_changed" if changed else
                    "corrected_admitted" if successor.get("accepted") is True else
                    "same_error_repeated" if next_failure.get("error_id") and next_failure["error_id"] == failure.get("error_id") else
                    "different_failure" if next_failure.get("first_failed_stage") else "unresolved")
        usage = attempt_usage(bundle, [successor] if successor else [])
        result.append({**origin_metadata(bundle), "decision_id": decision, "root_attempt_id": root["attempt_id"],
            "successor_attempt_id": successor["attempt_id"] if successor else None,
            "category": category, "root_error_id": failure.get("error_id"), "next_error_id": next_failure.get("error_id"),
            "semantic_choice_changed": changed,
            "intent_preserved": not changed if changed is not None else None,
            "immediate_admitted": successor.get("accepted") if successor else False,
            "later_task_success": final_success, "retry_response_tokens": usage["response_tokens"],
            "retry_model_calls": usage["model_calls"], "retry_usage": usage,
            "interpretation": "immediate same-decision successor; final success is an association, not a repair effect"})
    return result


def propagation(mcp: Mapping[str, Any], cli: Mapping[str, Any], *,
                mcp_success: bool | None, cli_success: bool | None) -> dict:
    result = first_divergence(mcp, cli)
    observed = result["status"] == "divergence_observed"
    index = int(result["first_divergence_attempt"]) - 1 if observed else None
    sequences = {"mcp": _natural_attempts(mcp), "cli": _natural_attempts(cli)}
    for label, offset in (("remaining", 0), ("subsequent", 1)):
        for interface, bundle in (("mcp", mcp), ("cli", cli)):
            tail = sequences[interface][index + offset:] if observed else []
            usage = attempt_usage(bundle, tail)
            for name, value in {"attempts": len(tail), "model_calls": usage["model_calls"],
                                "response_tokens": usage["response_tokens"],
                                "failures": sum(bool(r.get("failure")) for r in tail),
                                "retries": sum(r.get("retry_index", 0) > 0 for r in tail)}.items():
                result[f"{label}_{name}_{interface}"] = value if observed else None
        for name in ("attempts", "model_calls", "response_tokens", "failures", "retries"):
            a, b = (result[f"{label}_{name}_{side}"] for side in ("mcp", "cli"))
            result[f"{label}_{name}_delta"] = a - b if a is not None and b is not None else None
    later = list(zip(sequences["mcp"][index + 1:], sequences["cli"][index + 1:])) if observed else []
    known = [(a, b) for a, b in later if a["semantic_key"] is not None and b["semantic_key"] is not None]
    matches = sum(a["semantic_key"] == b["semantic_key"] for a, b in known)
    result.update(later_aligned_attempts=len(later) if observed else None,
                  later_jointly_decoded_attempts=len(known) if observed else None,
                  later_semantic_agreements=matches if observed else None,
                  later_semantic_agreement_rate=matches / len(known) if known else None,
                  terminal_success_delta=int(mcp_success) - int(cli_success)
                      if observed and mcp_success is not None and cli_success is not None else None,
                  suffix_definition="remaining includes the first divergence; subsequent excludes it",
                  unknown_semantics="agreement uses jointly decoded ordinal attempts only")
    return result


def dart_triggers(bundle: Mapping[str, Any]) -> list[dict]:
    """Extract trigger sessions from structured controller events."""
    events = bundle.get("_events", bundle.get("dart_events", []))
    sessions, current = [], None
    for event in events:
        kind, payload = event["kind"], event.get("payload") or {}
        if kind == "trigger":
            current = {"trigger_index": len(sessions) + 1, "trigger_failure_stage": (payload.get("public_failure") or {}).get("first_failed_stage"),
                "attempt_id": payload.get("attempt_id"), "valid_commitment": False,
                "realizations": [], "correction_dispatched": False,
                "release_reason": None, "fallback": False, "outcome": None}
            sessions.append(current)
        elif current is not None:
            if kind == "commitment":
                current["valid_commitment"] = True
            elif kind == "realization":
                current["realizations"].append(payload)
            elif kind == "dispatch" and payload.get("purpose") in {"dart_correction", "dart_reselection"}:
                current["correction_dispatched"] = True
            elif kind in {"release", "selection_rejected"}:
                current["release_reason"] = payload.get("reason") or kind
            elif kind == "release_to_baseline":
                current.update(fallback=True, release_reason=payload.get("reason"), attempt_id=payload.get("attempt_id"))
            elif kind == "repair_outcome":
                current["outcome"] = payload
    interventions = [(a["attempt_id"], a["intervention"]) for a in bundle.get("attempt_rows", []) if a.get("intervention")]
    by_attempt = dict(interventions)
    # Use attempt coordinates when present; otherwise pair by order when
    # every trigger has exactly one recorded intervention.
    ordered_link = len(interventions) == len(sessions)
    result = []
    for i, session in enumerate(sessions):
        outcome = session["outcome"] or by_attempt.get(session["attempt_id"])
        if outcome is None and ordered_link:
            session["attempt_id"], outcome = interventions[i]
        realizations = session["realizations"]
        first = next((r for r in realizations if r["realization_attempts"] == 1), None)
        correction = next((r for r in realizations if r["realization_attempts"] == 2), None)
        correction_attempted = correction is not None or session["correction_dispatched"]
        result.append({**origin_metadata(bundle), "trigger_index": session["trigger_index"],
            "attempt_id": session["attempt_id"], "trigger_failure_stage": session["trigger_failure_stage"],
            "valid_commitment": session["valid_commitment"],
            "first_realization_admitted": first.get("admitted") if first else None,
            "correction_opportunity": (not first["admitted"] if not first.get("deterministic") else False)
                if first is not None else None if session["valid_commitment"] else False,
            "correction_attempted": correction_attempted,
            "corrected_admitted": correction.get("admitted") if correction else None if correction_attempted else False,
            "preserving_corrected_admitted": (bool(correction.get("admitted") and correction.get("commitment_preserved") is True)
                                               if correction else None if correction_attempted else False),
            "reselected": correction.get("reselected") if correction else False,
            "released": outcome.get("released") if outcome else True if session["release_reason"] else None,
            "release_reason": outcome.get("release_reason") if outcome else session["release_reason"],
            "fallback": outcome.get("released_to_ordinary_retry", session["fallback"]) if outcome else session["fallback"],
            "outcome_recorded": outcome is not None,
            "interpretation": "semantic-validator admission before execution; not final task success"})
    return result


def phase_costs(bundle: Mapping[str, Any]) -> list[dict]:
    ledger = _calls(bundle)
    fallback_ids = {call_id for a in bundle.get("attempt_rows", [])
                    if (a.get("intervention") or {}).get("released_to_ordinary_retry")
                    for call_id in a.get("model_call_ids", [])
                    if ledger.get(call_id, {}).get("purpose") == "environment_action"}
    fallback_ids.update((event.get("payload") or {}).get("call_id")
                        for event in bundle.get("_events", bundle.get("dart_events", []))
                        if event["kind"] == "release_to_baseline" and (event.get("payload") or {}).get("call_id"))
    buckets = {name: [] for name in PHASES}
    phase_by_purpose = {"environment_action": "ordinary", "dart_selection": "commitment",
                        "dart_realization": "realization", "dart_correction": "correction", "dart_reselection": "correction"}
    for call_id, call in ledger.items():
        phase = "fallback" if call_id in fallback_ids else phase_by_purpose.get(call.get("purpose"), "unclassified")
        buckets[phase].append(call)
    unsettled = bundle.get("_unsettled_call_records", [])
    for call in unsettled:
        phase = "fallback" if call.get("call_id") in fallback_ids else phase_by_purpose.get(call.get("purpose"), "unclassified")
        # A dispatch without settlement has no known response usage.
        buckets[phase].append(dict(call, usage={}))
    unexplained = int(bundle.get("_unsettled_calls", 0)) - len(unsettled)
    if unexplained < 0:
        raise ValueError("inconsistent unsettled-call accounting")
    buckets["unclassified"].extend({"usage": {}} for _ in range(unexplained))
    return [origin_metadata(bundle) | {"phase": phase, **response_usage(calls),
            "phase_definition": "disjoint logical-call purposes; fallback uses explicit intervention assignment",
            "unsettled_calls": sum(c.get("call_id") in {u.get("call_id") for u in unsettled} for c in calls)
                + (unexplained if phase == "unclassified" else 0)} for phase, calls in buckets.items()]


def clustered_summary(records: Sequence[Mapping[str, Any]], dimensions: Sequence[str],
                      metrics: Mapping[str, tuple[str, str | None]], *, samples: int, seed: int,
                      task_mean: bool = False) -> list[dict]:
    """Task-cluster CIs for explicit ratios or equal-task means.

    With no denominator field, each supported row has unit mass. Ratio
    resampling preserves every root/attempt contribution of a sampled task.
    """
    origin = origin_metadata(*records)
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in dimensions)].append(row)
    output = []
    for group, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        tasks = sorted({r["task_id"] for r in rows})
        local_seed = int(digest([seed, list(dimensions), group])[:16], 16)
        rng = np.random.default_rng(local_seed)
        weights = np.vstack([np.ones(len(tasks)), rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=samples)])
        for name, (numerator, denominator) in metrics.items():
            sums = defaultdict(lambda: [0., 0.])
            supported_rows = 0
            for row in rows:
                x, y = row.get(numerator), row.get(denominator) if denominator else 1
                if x is None or y is None or y <= 0:
                    continue
                sums[row["task_id"]][0] += float(x)
                sums[row["task_id"]][1] += float(y)
                supported_rows += 1
            counts = np.array([sums[t] for t in tasks])
            if task_mean:
                for pair in counts:
                    if pair[1]:
                        pair[0], pair[1] = pair[0] / pair[1], 1
            totals = weights @ counts
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(totals[:, 1] > 0, totals[:, 0] / totals[:, 1], np.nan)
            valid = values[1:][np.isfinite(values[1:])]
            support = int((counts[:, 1] > 0).sum())
            low, high = np.quantile(valid, [.025, .975]).tolist() if len(valid) and support >= 2 else (None, None)
            output.append(dict(zip(dimensions, group)) | origin | {"metric": name,
                "estimate": float(values[0]) if np.isfinite(values[0]) else None,
                "ci95_low": low, "ci95_high": high, "tasks": len(tasks), "supported_tasks": support,
                "rows": len(rows), "supported_rows": supported_rows,
                "numerator_total": sum(sums[t][0] for t in tasks), "denominator_total": sum(sums[t][1] for t in tasks),
                "numerator": numerator, "denominator": denominator or "supported_rows",
                "aggregation": "equal_task_mean" if task_mean else "pooled_ratio_with_task_clusters",
                "bootstrap_samples": samples, "supported_bootstrap_samples": len(valid), "bootstrap_seed": local_seed,
                "ci_evidence": "mock_not_measured" if origin["data_origin"] == "synthetic" else
                    "task_bootstrap_percentile" if origin["data_origin"] == "experimental" else "origin_not_verified"})
    return output


def episode_diagnostics(rows: Sequence[Mapping[str, Any]], bundles: Mapping[str, Mapping[str, Any]], *,
                        samples: int, seed: int) -> dict[str, list[dict]]:
    roots, triggers, costs, failures, proposals = [], [], [], [], []
    for row in rows:
        bundle = bundles[row["bundle_key"]]
        common = {k: row[k] for k in IDENTITY} | origin_metadata(row)
        # Acquisition costs include infrastructure failures; efficacy does not.
        sessions = dart_triggers(bundle)
        for cost in phase_costs(bundle):
            costs.append(common | cost | {"infrastructure_failure": row["infrastructure_failure"], "triggered_failures": len(sessions)})
        for trigger in sessions:
            triggers.append(common | trigger | {"infrastructure_failure": row["infrastructure_failure"]})
        if row["infrastructure_failure"]:
            continue
        roots.extend(common | root for root in recovery_roots(bundle, final_success=row["success"]))
        attempts = _natural_attempts(bundle)
        stages = Counter((a.get("failure") or {}).get("first_failed_stage", "unknown") for a in attempts if a.get("failure"))
        failures.append(common | {"attempts": len(attempts), "failed_attempts": sum(stages.values()),
                                  "stage_counts": dict(stages)})
        for a in attempts:
            proposals.append(common | {"attempt_id": a["attempt_id"], "semantic_key": a["semantic_key"],
                "semantic_identity_level": a["semantic_identity_level"], "decoded": a["semantic_key"] is not None,
                "schema_id": (a.get("parsed_candidate") or {}).get("schema_id"), "admitted": a.get("accepted")})
    for root in roots:
        root.update({f"category_{category}": root["category"] == category for category in RECOVERY_CATEGORIES})
    root_metrics = {name: (name, None) for name in ("immediate_admitted", "later_task_success", "retry_response_tokens", "retry_model_calls")}
    root_metrics.update({f"category_{name}": (f"category_{name}", None) for name in RECOVERY_CATEGORIES})
    recovery_summary = clustered_summary(roots, GROUP, root_metrics, samples=samples, seed=seed)
    for record in failures:
        record.update({f"stage_{stage}": record["stage_counts"].get(stage, 0)
                       for stage in sorted({s for r in failures for s in r["stage_counts"]})})
    failure_metrics = {"failure_incidence": ("failed_attempts", "attempts")}
    for stage in sorted({s for r in failures for s in r["stage_counts"]}):
        failure_metrics[f"composition_{stage}"] = (f"stage_{stage}", "failed_attempts")
    failure_summary = clustered_summary(failures, GROUP, failure_metrics, samples=samples, seed=seed)
    trigger_metrics = {name: (name, None) for name in ("valid_commitment", "first_realization_admitted", "correction_opportunity",
                      "correction_attempted", "corrected_admitted", "preserving_corrected_admitted", "released", "fallback", "outcome_recorded")}
    trigger_metrics["corrected_admission_per_opportunity"] = ("corrected_admitted", "correction_opportunity")
    dart_summary = clustered_summary(triggers, GROUP + ("infrastructure_failure",), trigger_metrics, samples=samples, seed=seed)
    cost_summary = clustered_summary(costs, GROUP + ("phase", "infrastructure_failure"),
        {name: (name, None) for name in ("model_calls", "response_tokens", "known_response_token_lower_bound", "unknown_usage_calls")},
        samples=samples, seed=seed, task_mean=True)
    cost_summary += clustered_summary(costs, GROUP + ("phase", "infrastructure_failure"),
        {"response_tokens_per_trigger_in_triggered_episodes": ("response_tokens", "triggered_failures"),
         "model_calls_per_trigger_in_triggered_episodes": ("model_calls", "triggered_failures")}, samples=samples, seed=seed)
    profiles = []
    groups = defaultdict(list)
    for proposal in proposals:
        groups[tuple(proposal[k] for k in GROUP)].append(proposal)
    for group, population in sorted(groups.items()):
        decoded = [p for p in population if p["decoded"]]
        for level, field in (("typed_action", "semantic_key"), ("operation_schema", "schema_id")):
            for key, count in sorted(Counter(p[field] for p in decoded).items()):
                profiles.append(dict(zip(GROUP, group)) | origin_metadata(*population) | {
                    "profile_level": level, "semantic_choice": key, "count": count,
                    "decoded_proposals": len(decoded), "all_attempts": len(population),
                    "conditional_share": count / len(decoded), "denominator": "decoded_normalized_proposals",
                    "interpretation": "descriptive proposal mass, not independent task observations"})
    return {"recovery_roots": roots, "recovery_summary": recovery_summary,
            "failure_summary": failure_summary, "semantic_proposals": proposals, "semantic_profiles": profiles,
            "dart_triggers": triggers, "dart_summary": dart_summary, "phase_costs": costs, "phase_cost_summary": cost_summary}
