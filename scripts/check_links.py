#!/usr/bin/env python3
"""Check that every internal link and asset in the built site exists.

Usage: python3 scripts/check_links.py public [https://princemodi.me]

Walks the HTML files under the build folder, collects href/src values that
point inside the site (relative paths, root paths, or absolute URLs on the
site's own domain), and fails if any target is missing. The check is
case-sensitive, matching GitHub Pages, so /uploads/cv.pdf does not pass just
because /uploads/CV.pdf exists (macOS file systems would let that slip by).
External links are not fetched.
"""

import html
import os
import re
import sys
from functools import lru_cache
from urllib.parse import unquote

ATTR = re.compile(r"""(?:href|src)=(?:"([^"#?]*)"|'([^'#?]*)'|([^\s>"'#?]+))""")
SKIP = re.compile(r"^(https?:|mailto:|tel:|data:|javascript:|whatsapp:|//)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = os.path.abspath(sys.argv[1])
    base = (sys.argv[2] if len(sys.argv) > 2 else "https://princemodi.me").rstrip("/")

    @lru_cache(maxsize=None)
    def listing(folder):
        try:
            return frozenset(os.listdir(folder))
        except OSError:
            return frozenset()

    def exists_exact(path):
        cur = root
        for part in os.path.relpath(path, root).split(os.sep):
            if part not in listing(cur):
                return False
            cur = os.path.join(cur, part)
        return True

    broken = {}
    checked = 0
    for folder, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".html"):
                continue
            page = os.path.join(folder, name)
            with open(page, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            for m in ATTR.finditer(text):
                url = html.unescape(m.group(1) or m.group(2) or m.group(3) or "").strip()
                if not url or "{{" in url:
                    continue
                if url.startswith(base):
                    url = url[len(base):] or "/"
                if SKIP.match(url):
                    continue
                if not url.startswith("/"):
                    url = "/" + os.path.relpath(os.path.normpath(os.path.join(folder, url)), root)
                target = os.path.normpath(os.path.join(root, unquote(url).lstrip("/")))
                if not target.startswith(root):
                    continue
                checked += 1
                ok = (os.path.isfile(target) and exists_exact(target)) or (
                    os.path.isfile(os.path.join(target, "index.html"))
                    and exists_exact(os.path.join(target, "index.html"))
                )
                if not ok:
                    broken.setdefault(url, set()).add(os.path.relpath(page, root))

    print(f"Internal links checked: {checked}, broken: {len(broken)}")
    for url, pages in sorted(broken.items()):
        shown = ", ".join(sorted(pages)[:3])
        more = f" (+{len(pages) - 3} more)" if len(pages) > 3 else ""
        print(f"  {url}  <- {shown}{more}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
