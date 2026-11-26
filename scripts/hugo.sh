#!/usr/bin/env bash
# Run the Hugo version this site is pinned to (the same one CI uses), downloading it on first use.
# The theme (Hugo Blox blox-bootstrap v5.9.6) does not build on newer Hugo releases.
# Usage: scripts/hugo.sh server    |    scripts/hugo.sh --gc --minify
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(sed -n 's/^ *HUGO_VERSION: *//p' .github/workflows/hugo.yaml | head -1)
bin=".hugo/$version/hugo"
if [ ! -x "$bin" ]; then
  case "$(uname -s)" in
    Darwin) asset="hugo_extended_${version}_darwin-universal.tar.gz" ;;
    Linux) asset="hugo_extended_${version}_linux-amd64.tar.gz" ;;
    *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
  esac
  mkdir -p ".hugo/$version"
  echo "Downloading Hugo $version..." >&2
  curl -fsSL "https://github.com/gohugoio/hugo/releases/download/v${version}/${asset}" | tar -xz -C ".hugo/$version" hugo
fi
exec "$bin" "$@"
