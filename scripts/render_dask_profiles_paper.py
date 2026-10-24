"""Render the one-page visual brief; optional authoring dependency: reportlab.

Detailed methods stay in docs/benchmarks/dask_profiles_verification.md.
Usage: python scripts/render_dask_profiles_paper.py --source-commit COMMIT
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

INK = colors.HexColor("#162a37")
MUTED = colors.HexColor("#516572")
TEAL = colors.HexColor("#087f79")
PALE = colors.HexColor("#e9f5f2")
LINE = colors.HexColor("#d7e1e5")
WHITE = colors.white


def paragraph(c, text, x, top, width, size=11, color=INK, bold=False, max_height=None):
    style = ParagraphStyle(
        "brief",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=size * 1.3,
        textColor=color,
    )
    p = Paragraph(html.escape(text), style)
    _, height = p.wrap(width, 800)
    if max_height is not None and height > max_height:
        raise ValueError(f"Text exceeds its layout region: {text}")
    p.drawOn(c, x, top - height)
    return height


def box(c, x, y, width, height, fill=WHITE, stroke=LINE):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, y, width, height, 9, fill=1, stroke=1)


def arrow(c, x1, y1, x2, y2, color=MUTED):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.4)
    c.line(x1, y1, x2 - 4, y2)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - 6, y2 + 3)
    p.lineTo(x2 - 6, y2 - 3)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def stage(c, x, title, lines, active=False):
    box(c, x, 560, 120, 78, PALE if active else WHITE, TEAL if active else LINE)
    paragraph(c, title, x + 10, 622, 100, 11, TEAL if active else INK, True, 29)
    paragraph(c, lines, x + 10, 592, 100, 9.5, MUTED, max_height=27)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input", type=Path, default=Path("docs/benchmarks/dask_profiles_brief.json"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/dask-profiles-verification.pdf"))
    args = parser.parse_args()
    brief = json.loads(args.input.read_text())
    raw = Path("docs/benchmarks/raw")
    proof = json.loads((raw / "independent-verifier/gpu-four-trial-comparison.json").read_text())
    checks = json.loads((raw / "coordinator-validation.json").read_text())
    assert proof["verdict"] == "VERIFIED" and proof["exact_export_bytes"] and proof["baseline_repeatable"]
    compared = list(proof["comparisons"].values())
    assert all(p["pass"] for p in compared)
    cameras = {p[key] for p in compared for key in ("baseline_cameras", "treatment_cameras")}
    tracks = {p[key] for p in compared for key in ("baseline_tracks", "treatment_tracks")}
    assert len(cameras) == len(tracks) == 1
    camera_count, track_count = cameras.pop(), tracks.pop()
    tests = checks["checks"]["focused_pytest"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=(612, 792))
    c.setTitle("GTSfM - reusable compute settings in one page")
    c.setAuthor("GTSfM project - AI-assisted engineering brief")
    c.setFillColor(colors.HexColor("#f8fafb"))
    c.rect(0, 0, 612, 792, fill=1, stroke=0)

    paragraph(c, "GTSfM STUDIO  |  COMPUTE INFRASTRUCTURE  |  24 OCT 2026", 36, 761, 540, 8, MUTED)
    paragraph(c, brief["title"], 36, 734, 540, 25, INK, True, 35)
    paragraph(c, brief["subtitle"], 36, 696, 540, 12.5, MUTED, max_height=20)
    paragraph(c, "WHERE THIS FITS", 36, 663, 540, 9, TEAL, True)
    for x, title, body, active in [
        (36, "Photos", "Your input images", False),
        (176, "Studio setup", "NEW: saved profiles + validation", True),
        (316, "Existing SfM", "Dask schedules CPU/GPU workers", False),
        (456, "3D model", "Cameras + sparse points", False),
    ]:
        stage(c, x, title, body, active)
    for x in (156, 296, 436):
        arrow(c, x + 2, 600, x + 18, 600)
    paragraph(c, brief["context"], 36, 548, 540, 10.5, MUTED, max_height=29)

    box(c, 36, 403, 260, 108)
    box(c, 308, 403, 268, 108, PALE)
    paragraph(c, "BEFORE: setup lived in the form", 48, 495, 236, 11, INK, True)
    paragraph(c, brief["before"], 48, 472, 236, 11, max_height=60)
    paragraph(c, "AFTER: a reusable named profile", 320, 495, 244, 11, TEAL, True)
    paragraph(c, brief["after"], 320, 472, 244, 11, max_height=60)
    paragraph(c, brief["why"], 36, 387, 540, 12, INK, True, 20)

    paragraph(c, "CHECK: DID THE RECONSTRUCTION CHANGE?", 36, 352, 540, 9, TEAL, True)
    box(c, 36, 199, 540, 139)
    box(c, 48, 283, 150, 36)
    box(c, 48, 226, 150, 36, PALE)
    paragraph(c, "Manual settings - 2 runs", 58, 307, 130, 10.2, INK, True)
    paragraph(c, "Saved profile - 2 runs", 58, 250, 130, 10.2, TEAL, True)
    box(c, 228, 226, 105, 93, colors.HexColor("#eef2f5"))
    paragraph(c, "Same RTX 3080", 238, 301, 85, 10.5, INK, True)
    paragraph(c, "Same images and SfM configuration", 238, 271, 85, 10, MUTED, max_height=40)
    arrow(c, 201, 301, 225, 301)
    arrow(c, 201, 244, 225, 244)
    arrow(c, 336, 272, 360, 272, TEAL)
    box(c, 363, 226, 201, 93, PALE, TEAL)
    paragraph(c, "EXACT FILE MATCH", 375, 303, 177, 9, TEAL, True)
    paragraph(c, "Same exported files.", 375, 280, 177, 15, INK, True)
    paragraph(c, f"{camera_count} cameras  |  {track_count:,} tracks", 375, 251, 177, 11, MUTED)
    paragraph(c, brief["finding"], 48, 218, 516, 9.5, MUTED, max_height=16)

    paragraph(c, "THE GAIN: repeatable setup + visible execution", 36, 174, 540, 14, INK, True)
    paragraph(c, brief["scope"], 36, 149, 540, 11, MUTED, max_height=32)
    paragraph(
        c,
        f"Also checked: restart persistence, invalid-setting rejection and {tests} focused tests.",
        36,
        110,
        540,
        10,
        MUTED,
        max_height=28,
    )
    c.setStrokeColor(LINE)
    c.line(36, 72, 576, 72)
    paragraph(c, "Detailed methods and raw results: " + brief["evidence_index"], 36, 58, 540, 8, MUTED)
    paragraph(
        c,
        "Evidence revision: " + args.source_commit[:8] + "  |  Engineering repeatability result",
        36,
        43,
        540,
        8,
        MUTED,
    )
    c.showPage()
    c.save()
    print(args.output)


if __name__ == "__main__":
    main()
