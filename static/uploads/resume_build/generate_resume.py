import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

output_file = os.getenv("RESUME_OUTPUT_NAME", "../resume.pdf")

CONFIG_FILE = "resume_config.json"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_resume():
    config = load_config(CONFIG_FILE)
    flotilla_bullets = config["work_experience"]["flotilla_bullets"]
    technical_skills = config["technical_skills"]
    dynamic_project = config["selected_project"]
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        topMargin=0.2 * inch,
        bottomMargin=0.2 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        title="Resume_Prince_Modi",
        author="Prince Modi",
        subject="Software Engineering Internship Resume",
    )

    FULL_WIDTH = 7.5 * inch
    styles = getSampleStyleSheet()

    # --- STYLES ---
    style_name = ParagraphStyle(
        "Name",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    style_contact = ParagraphStyle(
        "Contact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    style_section_header = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        spaceBefore=2,
        spaceAfter=3,
        uppercase=True,
    )
    style_title = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
    )
    style_date = ParagraphStyle(
        "Date",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        alignment=TA_RIGHT,
    )
    style_subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=12,
    )
    style_project_sub = ParagraphStyle(
        "ProjectSub",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=12,
        leftIndent=0,
        spaceBefore=8,
    )
    style_bullet = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=11.5,
        spaceAfter=1,
        alignment=TA_JUSTIFY,
    )
    style_pub = ParagraphStyle(
        "Publication",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=11.5,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
    )

    story = []

    def add_section(title):
        t = Table([[Paragraph(title, style_section_header)]], colWidths=[FULL_WIDTH])
        t.setStyle(
            TableStyle(
                [
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.black),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.05 * inch))

    def add_entry(institution, location, date, role="", project_title=""):
        header_text = f"{institution}, {location}" if location else institution
        col_widths = [3.5 * inch, 4.0 * inch]
        data = [[Paragraph(header_text, style_title), Paragraph(date, style_date)]]

        if role:
            data.append([Paragraph(role, style_subtitle), ""])
        if project_title:
            underlined_title = f"<u>{project_title}</u>"
            data.append([Paragraph(underlined_title, style_project_sub), ""])

        t = Table(
            data,
            colWidths=col_widths,
            rowHeights=[None, None, 16] if project_title else None,
        )
        t.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(t)

    # --- HEADER ---
    story.append(Paragraph("PRINCE MODI", style_name))
    story.append(
        Paragraph(
            "San Diego, CA | "
            '<link href="tel:+16287248625">+1 (628) 724-8625</link> | '
            '<link href="mailto:princebmodi@outlook.com">princebmodi@outlook.com</link> | '
            '<link href="https://www.linkedin.com/in/modi-prince/">linkedin/modi-prince</link> | '
            '<link href="https://princemodi.me">princemodi.me</link> | '
            '<link href="https://github.com/prince-modi">github.com/prince-modi</link>',
            style_contact,
        )
    )

    story.append(Spacer(1, 0.06 * inch))

    # --- EDUCATION ---
    add_section("EDUCATION")
    add_entry(
        "University of California, San Diego",
        "",
        "Sep 2025 – Jun 2027",
        "MS Computer Science | GPA: 3.88/4.00",
    )
    story.append(
        Paragraph(
            "• Relevant Coursework: Distributed Systems, "
            "Deep Learning Systems, LLM System Optimization",
            style_bullet,
        )
    )
    story.append(Spacer(1, 0.05 * inch))

    add_entry(
        "Ganpat University",
        "India",
        "Aug 2018 – Jul 2022",
        "B. Tech. Computer Engineering | GPA: 3.96/4.00",
    )
    story.append(
        Paragraph(
            "• Relevant Coursework: Operating Systems, Cloud Computing | "
            "<b>Academic Scholarship</b> | 2nd rank out of 60+ students",
            style_bullet,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    # --- WORK EXPERIENCE ---
    add_section("WORK EXPERIENCE")

    add_entry(
        "Indian Institute of Science (IISc)",
        "Bangalore",
        "Jan 2023–Apr 2024 (Full-time); Apr 2024–Nov 2024 (Voluntary)",
        "Project Associate, Distributed Systems",
        project_title="Flotilla: Distributed Federated Learning Framework",
    )
    for b in flotilla_bullets:
        story.append(Paragraph(f"• {b}", style_bullet))

    story.append(Spacer(1, 0.06 * inch))

    add_entry(
        "Sterlite Technologies Limited",
        "Ahmedabad",
        "Jan 2022 – Aug 2022",
        "Software Engineering Intern",
    )
    story.append(
        Paragraph(
            "• Refactored Docker images on Linux CI/CD pipelines, reducing image size by <b>35%</b> and accelerating build times by <b>50%</b>",
            style_bullet,
        )
    )
    story.append(
        Paragraph(
            "• Integrated a Liquibase module to track MongoDB schema changes, improving version control reliability across environments",
            style_bullet,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    # --- PUBLICATION ---
    add_section("PUBLICATION")
    story.append(
        Paragraph(
            "Roopkatha B., <b>Prince Modi</b>, et al. "
            "<b>Flotilla: A Scalable, Modular and Resilient Federated Learning Framework for Heterogeneous Resources.</b> "
            "<i>Journal of Parallel and Distributed Computing (JPDC 2025)</i>",
            style_pub,
        )
    )
    story.append(Spacer(1, 0.06 * inch))

    # --- PROJECTS ---
    add_section("PROJECTS")
    add_entry(
        "Distributed Container Snapshotting",
        "",
        "June 2026",
    )
    story.append(
        Paragraph(
            "• Designed a distributed <b>snapshotting system</b> for stateful containerized applications, implementing the <b>Chandy-Lamport algorithm</b> to capture globally consistent cuts across nodes communicating over UDP",
            style_bullet,
        )
    )
    story.append(
        Paragraph(
            "• Built a <b>reliable-UDP sidecar proxy</b> using sequence numbering, buffering, and acknowledgment retransmission to guarantee in-order, exactly-once delivery without modifying application code",
            style_bullet,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    add_entry(
        'FlashAttention Kernel | <font name="Helvetica-Oblique">Triton, Python</font>',
        "",
        "Feb 2026",
    )
    story.append(
        Paragraph(
            "• Implementing <b>FlashAttention</b> and RMSNorm kernels in <b>Triton</b>, applying <b>tiled softmax</b> and <b>online normalization</b> to minimize HBM memory accesses and improve throughput for long-context LLM inference",
            style_bullet,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    add_entry(
        'System Performance Characterization | <font name="Helvetica-Oblique">C, Linux</font>',
        "",
        "Dec 2025",
    )
    story.append(
        Paragraph(
            "• Developed a suite of micro-benchmarks to evaluate the Rockchip RK3588S SoC, utilizing ARMv8 cycle counters to measure CPU scheduling latencies",
            style_bullet,
        )
    )
    story.append(Spacer(1, 0.05 * inch))

    """
    dynamic_header = (
        f'<b>{dynamic_project["title"]}</b> | '
        f'<font name="Helvetica-Oblique">{dynamic_project["stack"]}</font>'
    )
    add_entry(dynamic_header, "", dynamic_project["date"])
    for b in dynamic_project["bullets"]:
        story.append(Paragraph(f"• {b}", style_bullet))
    story.append(Spacer(1, 0.08 * inch))
    """

    # --- TECHNICAL SKILLS ---
    add_section("TECHNICAL SKILLS")
    for s in technical_skills:
        story.append(Paragraph(f"• {s}", style_bullet))

    doc.build(story)
    print(f"Resume generated: {output_file}")


if __name__ == "__main__":
    create_resume()
