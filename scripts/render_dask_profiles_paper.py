"""Render the verification report; optional authoring dependency: reportlab.

Usage: python scripts/render_dask_profiles_paper.py --source-commit COMMIT
"""

from __future__ import annotations

import argparse
import html
import re
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


def inline(text: str) -> str:
    text = text.replace("\u2014", " - ").replace("\u2013", "-").replace("\u2011", "-")
    text = text.replace("\u2192", "to").replace("\u2265", ">=").replace("\u00b0", " degrees")
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # Repository-relative references remain readable in a standalone PDF.
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input", type=Path, default=Path("docs/benchmarks/dask_profiles_verification.md"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/dask-profiles-verification.pdf"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=21, leading=25, spaceAfter=13))
    styles.add(
        ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=9.4, leading=13.4, spaceAfter=7, allowWidows=0, allowOrphans=0
        )
    )
    styles.add(
        ParagraphStyle(
            "Section",
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        )
    )
    styles.add(ParagraphStyle("Subsection", parent=styles["Section"], fontSize=10.5, leading=14))
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=10.8, alignment=TA_LEFT))
    styles.add(ParagraphStyle("CodeLine", fontName="Courier", fontSize=7.2, leading=10, leftIndent=8, spaceAfter=1))
    styles.add(
        ParagraphStyle(
            "Meta", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=9
        )
    )
    story = []
    lines = args.input.read_text().splitlines()
    width = 7.1 * inch
    index = 0
    title_seen = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                for wrapped in textwrap.wrap(
                    lines[index], width=102, replace_whitespace=False, drop_whitespace=False
                ) or [""]:
                    story.append(Paragraph(html.escape(wrapped).replace(" ", "&nbsp;"), styles["CodeLine"]))
                index += 1
            story.append(Spacer(1, 7))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in cells):
                    rows.append([Paragraph(inline(cell), styles["Cell"]) for cell in cells])
                index += 1
            columns = len(rows[0])
            widths = [width / columns] * columns
            if columns == 2:
                widths = [width * 0.31, width * 0.69]
            table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#94a3b8")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([table, Spacer(1, 8)])
            continue
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["ReportTitle"]))
            if not title_seen:
                story.append(Paragraph("TECHNICAL VERIFICATION REPORT | 24 OCTOBER 2026", styles["Meta"]))
                story.append(
                    Paragraph(f"Source and evidence commit: {html.escape(args.source_commit)}", styles["Meta"])
                )
                story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=10))
                title_seen = True
        elif line.startswith("## ") or line.startswith("### "):
            level = "Subsection" if line.startswith("### ") else "Section"
            story.append(Paragraph(inline(line.lstrip("# ")), styles[level]))
        else:
            block = [line]
            is_list = bool(re.match(r"(?:[-*] |\d+\. )", line))
            index += 1
            while (
                index < len(lines) and lines[index].strip() and not re.match(r"(?:#|\||```|[-*] |\d+\. )", lines[index])
            ):
                block.append(lines[index].strip())
                index += 1
            text = inline(" ".join(block))
            if is_list:
                text = re.sub(r"^[-*] ", "- ", text)
            story.append(Paragraph(text, styles["Body"]))
            continue
        index += 1

    def page(canvas, doc):
        canvas.setTitle("GTSFM Studio Dask execution profiles - verification")
        canvas.setAuthor("GTSFM project - AI-assisted technical verification")
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(0.7 * inch, 0.55 * inch, 7.8 * inch, 0.55 * inch)
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(0.7 * inch, 0.36 * inch, "GTSFM Studio | Bounded engineering verification | 2026-10-24")
        canvas.drawRightString(7.8 * inch, 0.36 * inch, str(doc.page))

    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=(8.5 * inch, 11 * inch),
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(args.output)


if __name__ == "__main__":
    main()
