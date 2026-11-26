"""Build Prince Modi's CV from cv_config.json.

Run from any directory: python3 cv_build/generate_cv.py
Use --output PATH to choose a different PDF destination.
"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "prince_modi_CV_generated.pdf"
INK = colors.HexColor("#18222e")
MUTED = colors.HexColor("#4c5a67")
RULE = colors.HexColor("#8a99a8")
WIDTH = letter[0] - 1.20 * inch


def safe(value):
    return escape(str(value))


def highlight_name(authors, name):
    return safe(authors).replace(safe(name), "<b>" + safe(name) + "</b>")


def styles():
    return {
        "name": ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, textColor=INK, spaceAfter=4),
        "contact": ParagraphStyle("contact", fontName="Helvetica", fontSize=8.8, leading=12, alignment=TA_CENTER, textColor=MUTED, spaceAfter=9),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=10.7, leading=13.5, textColor=INK),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=10.1, leading=13.5, textColor=INK),
        "date": ParagraphStyle("date", fontName="Helvetica", fontSize=9.1, leading=12, alignment=TA_RIGHT, textColor=MUTED),
        "role": ParagraphStyle("role", fontName="Helvetica-Oblique", fontSize=9.7, leading=13.2, textColor=INK),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.8, leading=13.5, textColor=INK, spaceAfter=2.5),
        "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.8, leading=13.5, leftIndent=11, firstLineIndent=-8, textColor=INK, spaceAfter=3),
        "pub": ParagraphStyle("pub", fontName="Helvetica", fontSize=9.7, leading=13.2, textColor=INK, spaceAfter=6),
    }


def heading(title, st):
    table = Table([[Paragraph(safe(title.upper()), st["section"])]], colWidths=[WIDTH])
    table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.55, RULE),
    ]))
    return table


def linked(label, url):
    return f'<link href="{safe(url)}" color="#376987">{safe(label)}</link>'


def entry(title, date, st, subtitle=None):
    data = [[Paragraph(safe(title), st["title"]), Paragraph(safe(date), st["date"])]]
    if subtitle:
        data.append([Paragraph(safe(subtitle), st["role"]), ""])
    table = Table(data, colWidths=[WIDTH * 0.66, WIDTH * 0.34], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def bullet(text, st):
    return Paragraph("&#8226; " + safe(text), st["bullet"])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(letter[0] - 0.60 * inch, 0.31 * inch, str(doc.page))
    canvas.restoreState()


def build(config, output):
    st = styles()
    doc = SimpleDocTemplate(
        str(output), pagesize=letter, topMargin=0.50 * inch,
        bottomMargin=0.48 * inch, leftMargin=0.60 * inch,
        rightMargin=0.60 * inch, title="Prince Modi CV",
        author=config["name"], subject="Curriculum Vitae",
    )
    story = [Paragraph(safe(config["name"]).upper(), st["name"])]
    contact = config["contact"]
    links = [
        (contact["email"], "mailto:" + contact["email"]),
        ("princemodi.me", contact["website"]),
        ("github.com/prince-modi", contact["github"]),
    ]
    story.append(Paragraph(" | ".join(
        f'<link href="{safe(url)}">{safe(label)}</link>' for label, url in links
    ), st["contact"]))

    def section(title):
        story.extend([CondPageBreak(1.05 * inch), heading(title, st), Spacer(1, 6)])

    section("Education")
    for item in config["education"]:
        block = [entry(item["institution"], item["date"], st, item["degree"])]
        block += [Paragraph(safe(x), st["body"]) for x in item.get("details", [])]
        block.append(Spacer(1, 6))
        story.append(KeepTogether(block))

    def add_experience(items):
        for item in items:
            story.append(KeepTogether([entry(item["organization"], item["date"], st, item["role"]),
                                       *([Paragraph(safe(item["role_detail"]), st["body"])] if item.get("role_detail") else []),
                                       *([Paragraph(safe(item["subheading"]), st["role"])] if item.get("subheading") else []),
                                       bullet(item["bullets"][0], st)]))
            story.extend(bullet(x, st) for x in item["bullets"][1:])
            story.append(Spacer(1, 6))

    section("Technical Skills")
    story.extend(Paragraph(safe(x), st["body"]) for x in config["skills"])

    section("Experience")
    add_experience(config["experience"])

    story.append(PageBreak())
    section("Selected Projects")
    for item in config["projects"]:
        links = [linked("Post", item["post_url"])]
        if item.get("code_url"):
            links.append(linked("Code", item["code_url"]))
        header = entry(item["title"], item["date"], st)
        links_line = Paragraph(safe(item["stack"]) + "  |  " + "  |  ".join(links), st["body"])
        story.append(KeepTogether([header, links_line, bullet(item["bullets"][0], st)]))
        story.extend(bullet(x, st) for x in item["bullets"][1:])
        story.append(Spacer(1, 7))

    for label, key in [("Publication", "publications"), ("Selected Posters", "posters")]:
        section(label)
        for item in config[key]:
            title = safe(item["title"])
            links = [linked("PDF", item["url"])] if item.get("url") else []
            if item.get("code_url"):
                links.append(linked("Code", item["code_url"]))
            venue_line = "<i>" + safe(item["venue"]) + "</i>"
            if links:
                venue_line += "  |  " + "  |  ".join(links)
            story.append(KeepTogether([Paragraph(
                "<b>" + title + "</b><br/>" +
                highlight_name(item["authors"], config["name"]) + "<br/>" + venue_line, st["pub"])]))

    section("Teaching Experience")
    for item in config["teaching"]:
        block = [entry(item["organization"], item["date"], st, item["role"])]
        bullets = item.get("bullets", [])
        if bullets:
            block.append(bullet(bullets[0], st))
        story.append(KeepTogether(block))
        story.extend(bullet(x, st) for x in bullets[1:])
        story.append(Spacer(1, 4))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "cv_config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(config, args.output)
    print(f"CV generated: {args.output.resolve()}")


if __name__ == "__main__":
    main()
