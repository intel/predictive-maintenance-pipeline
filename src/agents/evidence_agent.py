"""
Evidence Agent - Generate audit trail documenting policy decisions.

Uses policy to filter detections and generates compliance trail.
"""

from typing import Dict, Any
import json
import os
import time
from datetime import datetime, timezone
from src.agents.utility.policy_filter import filter_detections


def evidence_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate audit trail for policy execution."""
    start_time = time.time()
    print(f"[Evidence Agent] ⏱️ Started at {time.strftime('%H:%M:%S')}")
    
    meta = state.get("meta", {})
    for_evidence = meta.get("for_evidence", {})
    
    detections = for_evidence.get("detections", [])
    policy = for_evidence.get("policy", {})
    policy_metrics = for_evidence.get("policy_metrics", {})
    resources = for_evidence.get("resources", {})
    
    # Get evidence-specific LLM
    llms = resources.get("llms", {})
    llm = llms.get("evidence")
    prompts = resources.get("prompts", {})
    evidence_prompt = prompts.get("evidence_prompt", "")
    
    # Filter using utility
    filtered = filter_detections(detections, policy)

    representative_samples = _build_representative_samples(
        filtered,
        limit=3,
    )

    # Build evidence record
    timestamp = datetime.now(timezone.utc).isoformat()
    evidence_record = {
        "timestamp": timestamp,
        "policy_decisions": policy_metrics,
        "analysis_scope": {
            "total_detections": len(detections),
            "detections_kept": len(filtered)
        },
        "representative_samples": representative_samples,
    }
    
    # Generate audit trail text
    if llm and evidence_prompt:
        try:
            trail = _generate_trail_with_llm(llm, evidence_prompt, evidence_record)
        except Exception as e:
            print(f"[Evidence Agent] ⚠️  LLM failed ({e}), using fallback")
            trail = _generate_fallback_trail(evidence_record)
    else:
        trail = _generate_fallback_trail(evidence_record)
    
    evidence_record["trail"] = trail
    
    # Save outputs
    out_dir = resources.get("out_dir", "out")
    agent_dir = f"{out_dir}/agent"
    os.makedirs(agent_dir, exist_ok=True)
    with open(f"{agent_dir}/evidence.json", "w") as f:
        json.dump(evidence_record, f, indent=2)
    with open(f"{agent_dir}/evidence_trail.txt", "w") as f:
        f.write(trail)
    
    elapsed = time.time() - start_time
    print(f"[Evidence Agent] ✅ Complete: Audit trail generated ({elapsed:.2f}s)")
    
    return {"meta": {**meta, "evidence": {"trail": trail, "record": evidence_record}}}


def _confidence_sort_key(detection):
    """Return confidence as a sortable float."""
    try:
        return float(detection.get("confidence"))
    except (TypeError, ValueError):
        return float("-inf")


def _build_representative_samples(detections, limit=5):
    """Return high-confidence samples with available evidence fields."""
    evidence_fields = (
        "sample_id",
        "image_id",
        "source",
        "state",
        "label",
        "confidence",
        "image_confidence",
        "sensor_confidence",
        "audio_confidence",
        "text_confidence",
        "sensor_type",
        "text_caption",
        "text_prompt",
    )

    ranked = sorted(
        detections,
        key=_confidence_sort_key,
        reverse=True,
    )

    samples = []
    for detection in ranked[:limit]:
        sample = {
            field: detection.get(field)
            for field in evidence_fields
            if detection.get(field) is not None
        }

        for text_field in ("text_caption", "text_prompt"):
            value = sample.get(text_field)
            if isinstance(value, str) and len(value) > 300:
                sample[text_field] = value[:297] + "..."

        samples.append(sample)

    return samples


def _generate_trail_with_llm(llm, prompt: str, record: Dict) -> str:
    """Generate audit trail using LLM."""
    enhanced_prompt = f"{prompt}\n\nEvidence Record:\n{json.dumps(record, indent=2)}\n\nGenerate professional audit trail:"
    result = llm.invoke(enhanced_prompt, temperature=0.3, max_new_tokens=600)
    return result.strip()


def _generate_fallback_trail(record: Dict) -> str:
    """Generate fallback audit trail without LLM."""
    policy_metrics = record.get("policy_decisions", {})
    scope = record.get("analysis_scope", {})
    
    lines = ["="*70, "PIPELINE EXECUTION AUDIT TRAIL", "="*70, ""]
    lines.append(f"Timestamp: {record.get('timestamp', 'N/A')}")
    lines.append("Run Status: Completed")
    lines.append("Storage: SQLite + Local Files\n")
    
    lines.append("POLICY DECISIONS")
    lines.append("-"*70)
    lines.append(f"Global Threshold: {policy_metrics.get('confidence_threshold', 'N/A')}")
    lines.append(f"Detections Kept: {scope.get('detections_kept', 0)} / {scope.get('total_detections', 0)}\n")
    
    per_class = policy_metrics.get('per_class_thresholds', {})
    if per_class:
        lines.append("Per-Class Thresholds:")
        for label, thresh in per_class.items():
            lines.append(f"  {label}: {thresh}")
        lines.append("")
    
    bbox_min = policy_metrics.get('bbox_min_size', {})
    if bbox_min:
        lines.append("Bounding Box Min Size:")
        lines.append(f"  Width: {bbox_min.get('width', 0)}, Height: {bbox_min.get('height', 0)}\n")
    
    by_label = policy_metrics.get('by_label', {})
    if by_label:
        lines.append("Per-Class Breakdown (Kept):")
        for label, count in by_label.items():
            lines.append(f"  {label}: {count}")
        lines.append("")
    
    lines.append("ANALYSIS SCOPE")
    lines.append("-"*70)
    lines.append(f"Total Detections: {scope.get('total_detections', 0)}")
    lines.append(f"Detections Kept: {scope.get('detections_kept', 0)}\n")
    
    lines.append("="*70)
    return "\n".join(lines)
