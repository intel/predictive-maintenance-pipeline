"""
Ticketing Agent - Generate HTML tickets for defect frames.

Creates self-contained HTML tickets with detection details,
evidence trail context, and embedded frame images.
"""

import base64
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Allowlist pattern: stem must start with alphanumeric and contain only safe chars
_SAFE_STEM_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]*$')


def _safe_filename_stem(value: str) -> str:
    """Return a validated filename stem using os.path.basename (CodeQL-recognized
    sanitizer), then enforce an alphanumeric-only allowlist.  Raises ValueError
    if the value does not pass validation."""
    basename = os.path.basename(value)          # strips directory components
    stem = os.path.splitext(basename)[0]        # strip extension
    if not stem or not _SAFE_STEM_RE.fullmatch(stem):
        raise ValueError(f"Unsafe filename stem: {value!r}")
    return stem


def _filename_lookup_key(value: str) -> str:
    """Return a case-insensitive key for matching an existing viz file.

    This key is used only for dictionary lookup. It is never converted
    directly into a filesystem path.
    """
    basename = os.path.basename(str(value))
    stem = os.path.splitext(basename)[0]
    return stem.casefold()


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_out_dir() -> Path:
    """Get the use-case-specific output directory."""
    import json, yaml
    try:
        with open(PROJECT_ROOT / "config.json") as f:
            use_case_id = json.load(f).get("default-use-case", "pipeline_defects_detection")
        return PROJECT_ROOT / "out" / use_case_id
    except Exception:
        return PROJECT_ROOT / "out" / "pipeline_defects_detection"


def _read_evidence_trail() -> str:
    """Read the evidence trail file if it exists."""
    trail_path = _get_out_dir() / "agent" / "evidence_trail.txt"
    if trail_path.exists():
        return trail_path.read_text(encoding="utf-8")
    return "(No evidence trail available)"


def _encode_image_base64(image_path: Path) -> Optional[str]:
    """Encode an image to base64 for embedding in HTML."""
    if image_path.exists():
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    return None


def _get_frame_detections(db_client, frame_id) -> List[Dict[str, Any]]:
    """Get all detections/classifications for a specific frame/image from the database."""
    try:
        # Try frame_id first (detection use case)
        rows = db_client.execute_query(
            "SELECT * FROM detections WHERE frame_id = ?", (int(frame_id),)
        )
        return rows
    except Exception:
        pass
    try:
        # Try image_id (classification use case)
        rows = db_client.execute_query(
            "SELECT * FROM detections WHERE image_id = ?", (int(frame_id),)
        )
        if rows:
            return rows
    except Exception:
        pass
    try:
        # Try sample_id (audio/text classification use case)
        rows = db_client.execute_query(
            "SELECT * FROM detections WHERE sample_id = ?", (int(frame_id),)
        )
        if rows:
            return rows
    except Exception:
        pass
    try:
        # Try source match (classification use case, string identifier)
        rows = db_client.execute_query(
            "SELECT * FROM detections WHERE source LIKE ?", (f"%{frame_id}%",)
        )
        return rows
    except Exception:
        return []


def generate_ticket_html(
    frame_id,
    detections: List[Dict[str, Any]],
    evidence_trail: str,
    image_path: Optional[Path],
    image_base64: Optional[str],
    include_image: bool = True,
) -> str:
    """Generate a self-contained HTML ticket for a defect/classification."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Determine if this is detection (has x, y, width, height) or classification
    is_detection = detections and 'x' in detections[0]

    # Build detections table rows
    det_rows = ""
    if is_detection:
        table_header = "<tr><th>Label</th><th>Confidence</th><th>Position (x, y)</th><th>Size (w × h)</th></tr>"
        for det in detections:
            det_rows += f"""
        <tr>
            <td>{det.get('label', 'N/A')}</td>
            <td>{det.get('confidence', 0):.3f}</td>
            <td>({det.get('x', 0)}, {det.get('y', 0)})</td>
            <td>{det.get('width', 0)} × {det.get('height', 0)}</td>
        </tr>"""
    else:
        table_header = "<tr><th>Source</th><th>Label</th><th>Confidence</th></tr>"
        for det in detections:
            det_rows += f"""
        <tr>
            <td>{det.get('source', 'N/A')}</td>
            <td>{det.get('label', 'N/A')}</td>
            <td>{det.get('confidence', 0):.3f}</td>
        </tr>"""

    n_cols = 4 if is_detection else 3
    ticket_title = f"Frame {frame_id}" if is_detection else str(frame_id)
    safe_id = str(frame_id).replace('.', '_').replace('/', '_') if not isinstance(frame_id, int) else f"F{frame_id:06d}"

    # Image tag (embedded base64 or path reference)
    image_section = ""
    if include_image:
        if image_base64:
            img_tag = (
                f'<img src="data:image/jpeg;base64,{image_base64}" '
                f'alt="{ticket_title}" '
                f'style="max-width:100%;border-radius:8px;'
                f'border:1px solid #333">'
            )
        else:
            img_tag = (
                f'<p style="color:#f44">'
                f'Image not found: {image_path}</p>'
            )

        image_section = f"""
  <div class="section">
    <h2>📸 Image</h2>
    <div class="img-container">{img_tag}</div>
    <div class="path-info">Image path: {image_path}</div>
  </div>
"""


    # Escape evidence trail for HTML
    evidence_html = (
        evidence_trail.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ticket — {ticket_title}</title>
<style>
  :root {{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#c9d1d9;--accent:#58a6ff;--warn:#f85149}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:32px;line-height:1.6}}
  .ticket{{max-width:900px;margin:0 auto;background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
  .header{{background:linear-gradient(135deg,#1a2332,#0d1117);padding:24px 32px;border-bottom:1px solid var(--border)}}
  .header h1{{color:var(--accent);font-size:1.5rem;margin-bottom:4px}}
  .header .meta{{color:#8b949e;font-size:.85rem}}
  .section{{padding:20px 32px;border-bottom:1px solid var(--border)}}
  .section:last-child{{border-bottom:none}}
  .section h2{{color:var(--accent);font-size:1.1rem;margin-bottom:12px;display:flex;align-items:center;gap:8px}}
  table{{width:100%;border-collapse:collapse;margin-top:8px}}
  th,td{{text-align:left;padding:8px 12px;border:1px solid var(--border);font-size:.88rem}}
  th{{background:#0d1117;color:var(--accent);font-weight:600}}
  td{{color:var(--text)}}
  .evidence{{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:16px;
    font-family:'JetBrains Mono',monospace;font-size:.8rem;white-space:pre-wrap;max-height:400px;overflow-y:auto;color:#8b949e}}
  .img-container{{text-align:center;margin-top:8px}}
  .path-info{{color:#8b949e;font-size:.8rem;margin-top:8px;font-family:monospace}}
  @media print{{body{{background:#fff;color:#000}} .ticket{{border:2px solid #000}} th{{background:#eee;color:#000}} td{{color:#000}}}}
</style>
</head>
<body>
<div class="ticket">
  <div class="header">
    <h1>🎫 Ticket — {ticket_title}</h1>
    <div class="meta">Generated: {timestamp} &nbsp;|&nbsp; Ticket ID: TICKET-{safe_id}</div>
  </div>

  {image_section}
  <div class="section">
    <h2>🔍 Results ({len(detections)} {'detection' if is_detection else 'classification'}{'s' if len(detections) != 1 else ''})</h2>
    <table>
      <thead>{table_header}</thead>
      <tbody>{det_rows if det_rows else f'<tr><td colspan="{n_cols}" style="text-align:center;color:#8b949e">No results found</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>📋 Evidence Trail</h2>
    <div class="evidence">{evidence_html}</div>
  </div>
</div>
</body>
</html>"""
    return html


def create_ticket(
    db_client,
    frame_id,
    include_image: bool = True,
) -> Dict[str, Any]:
    """
    Create a ticket for a specific frame/image.

    Args:
        db_client: SQLiteClient instance for querying detections
        frame_id: Frame ID (int) or source image identifier (str/int)
        include_image: Whether this use case has image-based input

    Returns:
        Dict with 'ok', 'path', and 'message' keys
    """
    # Get all detections for this frame
    detections = _get_frame_detections(db_client, frame_id)

    # Read evidence trail
    evidence_trail = _read_evidence_trail()

    # Locate the image in viz/
    out_dir = _get_out_dir()
    viz_dir = out_dir / "viz"

    # Build an index of actual files in viz_dir. Image paths are selected
    # exclusively from this index — no user input ever flows into a Path
    # constructor, which eliminates CodeQL path-injection taint entirely.
    viz_files: Dict[str, Path] = {}  # lookup key -> enumerated resolved Path
    if viz_dir.is_dir():
        for entry in viz_dir.iterdir():
            if entry.is_file():
                viz_files[_filename_lookup_key(entry.name)] = entry.resolve()

    def _find_viz_image(target_stem: str) -> Optional[Path]:
        """Lookup an image by stem from the pre-built viz file index."""
        return viz_files.get(_filename_lookup_key(target_stem))

    # Locate the image — user input is only used to compute a lookup key (string),
    # the actual Path object always comes from viz_files (filesystem enumeration).
    image_path = None
    if isinstance(frame_id, int) or (isinstance(frame_id, str) and str(frame_id).isdigit()):
        # Numeric frame id — int() coercion produces a safe format string key
        fid = int(frame_id)
        image_path = _find_viz_image(f"frame_{fid:06d}")
        if not image_path and detections:
            src = detections[0].get('source', '')
            if src:
                image_path = _find_viz_image(src)
    else:
        image_path = _find_viz_image(str(frame_id))

    if include_image and not image_path:
        # No matching file found; use a non-existent placeholder so
        # image_base64 will be None (no file to encode).
        image_path = viz_dir / "frame_not_found.jpg"

    image_base64 = (
        _encode_image_base64(image_path)
        if include_image and image_path and image_path.exists()
        else None
    )

    # Determine display label
    display_label = str(frame_id)
    if detections and 'source' in detections[0]:
        display_label = detections[0]['source']

    # Generate the HTML ticket
    html = generate_ticket_html(
        frame_id=frame_id,
        detections=detections,
        evidence_trail=evidence_trail,
        image_path=image_path,
        image_base64=image_base64,
        include_image=include_image,
    )

    # Write ticket to disk
    tickets_dir = out_dir / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(frame_id).replace('/', '_').replace('\\', '_').replace('.', '_')
    ticket_file = tickets_dir / f"ticket_{safe_name}.html"
    ticket_file.write_text(html, encoding="utf-8")

    return {
        "ok": True,
        "path": str(ticket_file),
        "filename": ticket_file.name,
        "frame_id": frame_id,
        "num_detections": len(detections),
        "message": f"Ticket created for {display_label} with {len(detections)} result(s)",
    }
