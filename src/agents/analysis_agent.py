"""
Analysis Agent - Generate statistics and insights.

Uses policy to filter detections and generates analysis report.
"""

from typing import Dict, Any
from collections import Counter
import json
import os
import statistics
import time
from src.agents.utility.policy_filter import filter_detections


def analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate statistics and insights on detections."""
    start_time = time.time()
    print(f"[Analysis Agent] ⏱️ Started at {time.strftime('%H:%M:%S')}")
    
    meta = state.get("meta", {})
    for_analysis = meta.get("for_analysis", {})
    
    detections = for_analysis.get("detections", [])
    policy = for_analysis.get("policy", {})
    resources = for_analysis.get("resources", {})
    
    # Get analysis-specific LLM
    llms = resources.get("llms", {})
    llm = llms.get("analysis")
    prompts = resources.get("prompts", {})
    analysis_prompt = prompts.get("analysis_prompt", "")
    
    # Filter using utility
    filtered = filter_detections(detections, policy)
    
    if not filtered:
        print("[Analysis Agent] ⚠️  No detections after filtering")
        return {"meta": meta}
    
    # Generate statistics
    stats = _generate_stats(filtered, policy)

    # Generate text summary with LLM if available
    summary_text = ""
    if llm and analysis_prompt:
        try:
            summary_text = _generate_summary_with_llm(
                llm,
                analysis_prompt,
                stats,
            )
        except Exception as e:
            print(f"[Analysis Agent] ⚠️  LLM failed ({e}), using fallback")
            summary_text = _generate_fallback_summary(stats)
    else:
        summary_text = _generate_fallback_summary(stats)
    
    # Create output
    analysis = {
        "total_defects": len(filtered),
        "stats": stats,
        "summary_text": summary_text
    }
    
    # Save outputs
    out_dir = resources.get("out_dir", "out")
    agent_dir = f"{out_dir}/agent"
    os.makedirs(agent_dir, exist_ok=True)
    with open(f"{agent_dir}/analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2)
    with open(f"{agent_dir}/analysis_summary.txt", "w") as f:
        f.write(summary_text)
    
    elapsed = time.time() - start_time
    print(f"[Analysis Agent] ✅ Complete: {analysis['total_defects']} defects ({elapsed:.2f}s)")
    
    return {"meta": {**meta, "analysis": analysis}}


def _generate_stats(detections, policy=None):
    """Generate statistics from filtered detections."""
    label_counts = Counter([d["label"] for d in detections])
    policy = policy or {}
    
    label_confidences = {}
    for label in label_counts:
        confs = [d["confidence"] for d in detections if d["label"] == label]
        label_confidences[label] = {
            "count": len(confs),
            "mean": round(statistics.mean(confs), 3),
            "median": round(statistics.median(confs), 3),
            "min": round(min(confs), 3),
            "max": round(max(confs), 3)
        }
    
    return {
        "labels_found": list(label_counts.keys()),
        "counts_per_label": dict(label_counts),
        "confidence_stats": label_confidences,
        "fused_confidence_distribution": _numeric_stats(
            [d.get("confidence") for d in detections]
        ),
        "regression_statistics": (
            _generate_regression_stats(detections, policy)
            if policy.get("policy_type") == "regression"
            else {}
        ),
        # Image-vs-sensor distribution, modality dominance, and anomaly counts.
        "sensor_statistics": _generate_sensor_stats(detections, policy),
        "modality_confidence_statistics": (
            _generate_modality_confidence_stats(detections)
        ),
    }


def _safe_float(value):
    """Convert numeric-like values to float, returning None when unavailable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_stats(values):
    """Return compact distribution stats for numeric values."""
    nums = [_safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return {"count": 0}
    return {
        "count": len(nums),
        "mean": round(statistics.mean(nums), 3),
        "median": round(statistics.median(nums), 3),
        "min": round(min(nums), 3),
        "max": round(max(nums), 3)
    }

def _generate_modality_confidence_stats(detections):
    """Generate statistics for available modality confidence fields."""
    confidence_fields = (
        "image_confidence",
        "sensor_confidence",
        "audio_confidence",
        "text_confidence",
    )

    distributions = {}
    for field in confidence_fields:
        values = [
            det.get(field)
            for det in detections
            if _safe_float(det.get(field)) is not None
        ]
        if values:
            distributions[field] = _numeric_stats(values)

    audio_dominant = 0
    text_dominant = 0
    ties = 0
    confidence_gaps = []

    for det in detections:
        audio_conf = _safe_float(det.get("audio_confidence"))
        text_conf = _safe_float(det.get("text_confidence"))

        if audio_conf is None or text_conf is None:
            continue

        confidence_gaps.append(abs(audio_conf - text_conf))

        if audio_conf > text_conf:
            audio_dominant += 1
        elif text_conf > audio_conf:
            text_dominant += 1
        else:
            ties += 1

    result = {
        "distributions": distributions,
    }

    if confidence_gaps:
        result["audio_text_comparison"] = {
            "paired_count": len(confidence_gaps),
            "audio_dominant": audio_dominant,
            "text_dominant": text_dominant,
            "ties": ties,
            "absolute_confidence_gap": _numeric_stats(confidence_gaps),
        }

    return result


def _parse_sensor_raw(raw_value):
    """Parse sensor_raw_json into a dict when present."""
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

# Build sensor statistics for the analysis report.
def _generate_sensor_stats(detections, policy):
    """Generate sensor/image modality and raw sensor alert statistics."""
    image_values = []
    sensor_values = []
    modality_by_label = {}
    anomaly_count = 0
    anomaly_by_channel = Counter()
    thresholds = policy.get("sensor_alert_thresholds", {}) or {}

    for det in detections:
        label = det.get("label", "unknown")
        image_conf = _safe_float(det.get("image_confidence"))
        sensor_conf = _safe_float(det.get("sensor_confidence"))

        if image_conf is not None:
            image_values.append(image_conf)
        if sensor_conf is not None:
            sensor_values.append(sensor_conf)

        if image_conf is not None and sensor_conf is not None:
            label_stats = modality_by_label.setdefault(
                label,
                {"image_dominant": 0, "sensor_dominant": 0, "tie": 0}
            )
            if image_conf > sensor_conf:
                label_stats["image_dominant"] += 1
            elif sensor_conf > image_conf:
                label_stats["sensor_dominant"] += 1
            else:
                label_stats["tie"] += 1

        sensor_raw = _parse_sensor_raw(det.get("sensor_raw_json"))
        triggered = False
        for channel, threshold in thresholds.items():
            reading = _safe_float(sensor_raw.get(channel))
            threshold_value = _safe_float(threshold)
            if reading is not None and threshold_value is not None and reading > threshold_value:
                anomaly_by_channel[channel] += 1
                triggered = True
        if triggered:
            anomaly_count += 1
    
    mean_delta = None
    paired_deltas = []
    for det in detections:
        image_conf = _safe_float(det.get("image_confidence"))
        sensor_conf = _safe_float(det.get("sensor_confidence"))
        if image_conf is not None and sensor_conf is not None:
            paired_deltas.append(sensor_conf - image_conf)
    if paired_deltas:
        mean_delta = round(statistics.mean(paired_deltas), 3)

    return {
        "image_confidence_distribution": _numeric_stats(image_values),
        "sensor_confidence_distribution": _numeric_stats(sensor_values),
        "mean_sensor_minus_image_confidence": mean_delta,
        "modality_dominance_per_label": modality_by_label,
        "anomaly_flags": {
            "count": anomaly_count,
            "thresholds": thresholds,
            "by_channel": dict(anomaly_by_channel)
        }
    }


def _generate_regression_stats(detections, policy):
    """Generate O&G regression/degradation statistics."""
    threshold = _safe_float(policy.get("degradation_threshold"))
    continuous_values = [
        _safe_float(det.get("continuous_value"))
        for det in detections
    ]
    continuous_values = [v for v in continuous_values if v is not None]

    by_material = {}
    for det in detections:
        material = det.get("material_type") or "unknown"
        value = _safe_float(det.get("continuous_value"))
        if value is None:
            continue
        by_material.setdefault(material, []).append(value)

    material_stats = {
        material: _numeric_stats(values)
        for material, values in by_material.items()
    }

    high_degradation_count = 0
    if threshold is not None:
        high_degradation_count = sum(
            1 for value in continuous_values if value > threshold
        )

    return {
        "continuous_value_distribution": _numeric_stats(continuous_values),
        "degradation_threshold": threshold,
        "high_degradation_count": high_degradation_count,
        "mean_continuous_value_by_material": material_stats,
    }


def _generate_summary_with_llm(
    llm,
    prompt: str,
    stats: Dict,
) -> str:

    """Generate summary using the configured use-case prompt."""
    enhanced_prompt = (
        f"{prompt}\n\n"
        f"Statistics:\n{json.dumps(stats, indent=2)}\n\n"
        "Generate a concise analysis report:"
    )
    result = llm.invoke(enhanced_prompt, temperature=0.3, max_new_tokens=500)
    return result.strip()


def _generate_fallback_summary(stats: Dict) -> str:
    """Generate fallback summary without LLM."""
    counts = stats.get("counts_per_label", {})
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    
    lines = ["="*60, "DETECTION ANALYSIS REPORT", "="*60, ""]
    
    total = sum(counts.values())
    lines.append(f"Total Detections: {total}\n")
    
    lines.append("Top Detection Types:")
    for label, count in ranked:
        pct = (count / total) * 100 if total > 0 else 0
        lines.append(f"  • {label}: {count} ({pct:.1f}%)")
    
    lines.append("\nConfidence Statistics:")
    conf_stats = stats.get("confidence_stats", {})
    for label, stat in conf_stats.items():
        lines.append(f"\n  {label}:")
        lines.append(f"    Count: {stat['count']}")
        lines.append(f"    Mean: {stat['mean']:.3f}")
        lines.append(f"    Range: {stat['min']:.3f} - {stat['max']:.3f}")


    regression_stats = stats.get("regression_statistics", {})
    if regression_stats:
        lines.append("\nDegradation Statistics:")
        dist = regression_stats.get("continuous_value_distribution", {})
        if dist.get("count"):
            lines.append(
                f"  Predicted Thickness Loss Mean: {dist['mean']:.3f} "
                f"(range {dist['min']:.3f} - {dist['max']:.3f})"
            )
        threshold = regression_stats.get("degradation_threshold")
        if threshold is not None:
            lines.append(f"  Degradation Threshold: {threshold:.3f}")
            lines.append(
                f"  High Degradation Samples: "
                f"{regression_stats.get('high_degradation_count', 0)}"
            )

        material_stats = regression_stats.get("mean_continuous_value_by_material", {})
        if material_stats:
            lines.append("  Mean Thickness Loss by Material:")
            for material, material_dist in sorted(material_stats.items()):
                if material_dist.get("count"):
                    lines.append(
                        f"    {material}: {material_dist['mean']:.3f} "
                        f"({material_dist['count']} samples)"
                    )
    # Add sensor summary to the fallback report.
    sensor_stats = stats.get("sensor_statistics", {})
    lines.append("\nSensor Statistics:")
    image_dist = sensor_stats.get("image_confidence_distribution", {})
    sensor_dist = sensor_stats.get("sensor_confidence_distribution", {})
    if image_dist.get("count") and sensor_dist.get("count"):
        lines.append(
            f"  Image Confidence Mean: {image_dist['mean']:.3f} "
            f"(range {image_dist['min']:.3f} - {image_dist['max']:.3f})"
        )
        lines.append(
            f"  Sensor Confidence Mean: {sensor_dist['mean']:.3f} "
            f"(range {sensor_dist['min']:.3f} - {sensor_dist['max']:.3f})"
        )
        delta = sensor_stats.get("mean_sensor_minus_image_confidence")
        if delta is not None:
            lines.append(f"  Mean Sensor-Image Confidence Delta: {delta:.3f}")

    lines.append("  Modality Dominance by Class:")
    modality = sensor_stats.get("modality_dominance_per_label", {})
    if modality:
        for label, counts_for_label in modality.items():
            lines.append(
                f"    {label}: image={counts_for_label.get('image_dominant', 0)}, "
                f"sensor={counts_for_label.get('sensor_dominant', 0)}, "
                f"tie={counts_for_label.get('tie', 0)}"
            )
    else:
        lines.append("    No paired image/sensor confidence data available.")

    anomaly = sensor_stats.get("anomaly_flags", {})
    lines.append(f"  Sensor Alert Samples: {anomaly.get('count', 0)}")
    by_channel = anomaly.get("by_channel", {})
    if by_channel:
        channel_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_channel.items()))
        lines.append(f"  Alert Channels: {channel_summary}")

    lines.append("\n" + "="*60)
    return "\n".join(lines)
