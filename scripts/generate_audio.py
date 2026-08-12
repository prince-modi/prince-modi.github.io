#!/usr/bin/env python3
"""Generate narration audio for blog posts via the OpenAI TTS API.

Scans content/post/<slug>/index.md, and for any post that doesn't already
have a static/audio/<slug>.mp3, strips the markdown down to plain text,
splits it into chunks under OpenAI's per-request character limit, synthesizes
each chunk, and concatenates them into a single mp3 via ffmpeg.

Requires OPENAI_API_KEY in the environment. If it's not set, exits quietly
(so local `hugo` builds without the key don't fail).
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = REPO_ROOT / "content" / "post"
AUDIO_DIR = REPO_ROOT / "static" / "audio"
TTS_MODEL = "tts-1"
TTS_VOICE = "onyx"
MAX_CHARS = 3800  # OpenAI's TTS input limit is 4096 chars; leave margin
API_URL = "https://api.openai.com/v1/audio/speech"


def strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:]
    return text


def markdown_to_text(md):
    text = md
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # fenced code blocks
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> link text
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code -> content
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # heading markers
    text = re.sub(r"(\*\*\*|\*\*|\*|___|__|_)", "", text)  # bold/italic markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)  # blockquote markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullet markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered list markers
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)  # horizontal rules
    text = re.sub(r"<[^>]+>", "", text)  # stray HTML tags
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, max_chars=MAX_CHARS):
    """Greedily pack sentences into chunks under max_chars, splitting on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def synthesize_chunk(text, api_key):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": TTS_MODEL, "voice": TTS_VOICE, "input": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.content


def concat_mp3s(mp3_paths, output_path):
    if len(mp3_paths) == 1:
        mp3_paths[0].rename(output_path)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in mp3_paths:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", str(output_path)],
            check=True, capture_output=True, text=True,
        )
    finally:
        os.unlink(list_path)


def generate_for_post(post_dir, api_key):
    slug = post_dir.name
    index_md = post_dir / "index.md"
    if not index_md.exists():
        return False
    output_path = AUDIO_DIR / f"{slug}.mp3"
    if output_path.exists():
        return False

    raw = index_md.read_text(encoding="utf-8")
    text = markdown_to_text(strip_frontmatter(raw))
    if len(text) < 50:
        print(f"  skip {slug}: not enough text content")
        return False

    chunks = chunk_text(text)
    print(f"  {slug}: {len(text)} chars -> {len(chunks)} chunk(s)")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        part_paths = []
        for i, chunk in enumerate(chunks):
            audio_bytes = synthesize_chunk(chunk, api_key)
            part_path = tmp_path / f"part-{i:03d}.mp3"
            part_path.write_bytes(audio_bytes)
            part_paths.append(part_path)
        concat_mp3s(part_paths, output_path)

    print(f"  wrote {output_path.relative_to(REPO_ROOT)}")
    return True


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping audio generation.")
        return 0

    if not POSTS_DIR.exists():
        print(f"No posts directory at {POSTS_DIR}")
        return 0

    for post_dir in sorted(POSTS_DIR.iterdir()):
        if not post_dir.is_dir():
            continue
        try:
            generate_for_post(post_dir, api_key)
        except requests.HTTPError as e:
            body = e.response.text[:500] if e.response is not None else ""
            print(f"  ERROR generating audio for {post_dir.name}: {e}\n  {body}", file=sys.stderr)
        except Exception as e:
            print(f"  ERROR generating audio for {post_dir.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
