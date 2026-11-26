"""Build the one-page industry resume from industry_resume_config.json.

Run from uploads: cv_build/.venv/bin/python resume_build/generate_industry_resume.py
"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)


HERE = Path(__file__).resolve().parent
WIDTH = letter[0] - 72


def register_fonts():
    font_dir = Path('/System/Library/Fonts/Supplemental')
    fonts = {
        'Resume': 'Arial.ttf',
        'Resume-Bold': 'Arial Bold.ttf',
        'Resume-Italic': 'Arial Italic.ttf',
        'Resume-BoldItalic': 'Arial Bold Italic.ttf',
    }
    for name, filename in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        'Resume', normal='Resume', bold='Resume-Bold',
        italic='Resume-Italic', boldItalic='Resume-BoldItalic',
    )


def link(label, url):
    return f'<link href="{escape(url, {chr(34): "&quot;"})}">{escape(label)}</link>'


def build(config, output):
    register_fonts()
    body = ParagraphStyle('Body', fontName='Resume', fontSize=10.5, leading=12.2)
    st = {
        'body': body,
        'name': ParagraphStyle('Name', parent=body, fontName='Resume-Bold', fontSize=18,
                               leading=21, alignment=TA_CENTER, spaceAfter=3),
        'contact': ParagraphStyle('Contact', parent=body, alignment=TA_CENTER),
        'section': ParagraphStyle('Section', parent=body, fontName='Resume-Bold',
                                  fontSize=11, leading=13),
        'title': ParagraphStyle('Title', parent=body, fontName='Resume-Bold'),
        'role': ParagraphStyle('Role', parent=body, fontName='Resume-Italic'),
        'date': ParagraphStyle('Date', parent=body, alignment=TA_RIGHT),
        'bullet': ParagraphStyle('Bullet', parent=body, leftIndent=10, firstLineIndent=-8,
                                 spaceAfter=1.5),
        'research': ParagraphStyle('Research', parent=body, spaceAfter=3),
    }
    doc = BaseDocTemplate(
        str(output), pagesize=letter, leftMargin=36, rightMargin=36,
        topMargin=36, bottomMargin=36, title='Prince Modi - Industry Resume',
        author=config['name'], subject='Systems and ML Infrastructure Resume',
    )
    frame = Frame(36, 36, WIDTH, letter[1] - 72,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='Resume', frames=frame))
    story = []

    def section(title):
        story.append(Spacer(1, 6))
        rule = Table([[Paragraph(escape(title), st['section'])]], colWidths=[WIDTH])
        rule.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LINEBELOW', (0, 0), (-1, -1), .5, colors.black),
        ]))
        rule.keepWithNext = True
        story.extend([rule, Spacer(1, 3)])

    def header(title, date, role=None):
        row = Table([[Paragraph(title, st['title']), Paragraph(escape(date), st['date'])]],
                    colWidths=[WIDTH - 135, 135])
        row.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        result = [row]
        if role:
            result.append(Paragraph(escape(role), st['role']))
        return result

    def bullet(text):
        return Paragraph('&#8226; ' + text, st['bullet'])

    story.append(Paragraph(escape(config['name'].upper()), st['name']))
    contact = config['contact']
    story.append(Paragraph(' | '.join([
        escape(contact['location']), link(contact['phone'], contact['phone_url']),
        link(contact['email'], 'mailto:' + contact['email']),
    ]), st['contact']))
    story.append(Paragraph(' | '.join([
        link('princemodi.me', contact['website']),
        link('github.com/prince-modi', contact['github']),
        link('linkedin.com/in/modi-prince', contact['linkedin']),
    ]), st['contact']))

    section('EDUCATION')
    for index, item in enumerate(config['education']):
        block = header(escape(item['institution']), item['date'])
        block.append(Paragraph(escape(item['degree']), st['body']))
        story.append(KeepTogether(block))
        if index == 0:
            story.append(Spacer(1, 3))

    section('EXPERIENCE')
    for index, item in enumerate(config['experience']):
        block = header(escape(item['organization']), item['date'], item['role'])
        if item.get('detail'):
            block.append(Paragraph(escape(item['detail']), st['body']))
        block.extend(bullet(text) for text in item['bullets'])
        story.append(KeepTogether(block))
        if index < len(config['experience']) - 1:
            story.append(Spacer(1, 4))

    section('SELECTED PROJECTS')
    for index, item in enumerate(config['projects']):
        title = escape(item['title']) + ' | ' + link('Code', item['code_url'])
        block = header(title, item['date'])
        block.extend(bullet(text) for text in item['bullets'])
        story.append(KeepTogether(block))
        if index == 0:
            story.append(Spacer(1, 4))

    section('SELECTED PUBLICATIONS & POSTERS')
    for item in config['research']:
        title = link(item['title'], item['url']) if item.get('url') else escape(item['title'])
        story.append(Paragraph('<b>' + escape(item['venue']) + ':</b> ' + title + '.', st['research']))

    section('TECHNICAL SKILLS')
    for item in config['skills']:
        story.append(Paragraph('<b>' + escape(item['label']) + ':</b> ' + escape(item['text']), st['body']))

    doc.build(story)
    print(f'Resume generated: {output.resolve()} ({doc.page} page(s))')
    if doc.page != 1:
        raise RuntimeError(f'Resume must fit one page; got {doc.page}. Edit content or layout.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'industry_resume_config.json')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'prince_modi_industry_resume.pdf')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(config, args.output)


if __name__ == '__main__':
    main()
