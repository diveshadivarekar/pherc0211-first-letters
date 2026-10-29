#!/bin/bash
# Build the self-contained Progress Prize bundle (a folder that can be pushed as its own git repo).
# Run from the repo root:  bash scripts/build_prize_bundle.sh [out_dir]
# Copies code, docs, curated figures and small result tables; never data, weights or volumes.
set -e
OUT=${1:-progress_prize_oct2026_pherc0211}
rm -rf "$OUT"
mkdir -p "$OUT/docs" "$OUT/results" "$OUT/data/real_scans"

cp -r src scripts tests configs "$OUT"/
cp requirements.txt pytest.ini "$OUT"/
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f "$OUT/scripts/build_prize_bundle.sh"   # the bundle is the output of this script, not its source
cp scripts/build_prize_bundle.sh "$OUT/scripts/build_prize_bundle.sh"

cp docs/PRIZE_BUNDLE_README.md "$OUT/README.md"

cat > "$OUT/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 Divesh Adivarekar, Pranav Gugale, Rohan Gorde, Samarth Sangolgi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The scan data and third-party models are not covered by this licence; see the
"Third-party material" table in README.md.
EOF

# documentation (no course PDFs / internal to-do list)
for f in PHERC0211_SPIRAL_SCREEN PHERC0211_READER_V2 PHERC0211_READER_V3 INK_EVALUATION LITERATURE_GAP_ANALYSIS \
         PROGRESS_PRIZE_SUBMISSION SCROLL_SURVEY TIFXYZ WRAP_SPLITTING; do
  cp "docs/$f.md" "$OUT/docs/"
done
cp -r docs/figures docs/ink_eval docs/tifxyz "$OUT/docs/"
cp data/real_scans/README.md "$OUT/data/real_scans/README.md"

# results: tracked tables plus the spiral-screen tables kept in the older package
cp -r results/* "$OUT/results/"
mkdir -p "$OUT/results/pherc0211_spiral_screen"
cp submission_pherc0211_spiral_screen/results/* "$OUT/results/pherc0211_spiral_screen/"

# checksums over everything except itself
( cd "$OUT" && find . -type f ! -name CHECKSUMS.sha256 -print0 | sort -z | xargs -0 sha256sum > CHECKSUMS.sha256 )
du -sh "$OUT"
find "$OUT" -type f | wc -l
