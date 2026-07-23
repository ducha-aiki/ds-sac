#!/usr/bin/env bash
# Fetch ONE PhotoTourism validation scene (st_peters_square) for F estimation.
# The needed h5 files are the first entries of the ValOnly tar, so a 500 MB
# range request suffices instead of the full 2.2 GB archive (and no images —
# the pose-error metric doesn't use them).
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=st_peters_square
DEST="data/f_data/$SCENE"
[ -f "$DEST/matches.h5" ] && { echo "already present: $DEST"; exit 0; }
mkdir -p "$DEST"
URL="https://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-ValOnly.tar"
TMP=$(mktemp /tmp/valonly_head.XXXX.tar)
curl -fsS "$URL" --range 0-500000000 -o "$TMP"
tar -xf "$TMP" -C data/f_data --strip-components=2 \
  RANSAC-Tutorial-Data-ValOnly/val/$SCENE/matches.h5 \
  RANSAC-Tutorial-Data-ValOnly/val/$SCENE/match_conf.h5 \
  RANSAC-Tutorial-Data-ValOnly/val/$SCENE/K1_K2.h5 \
  RANSAC-Tutorial-Data-ValOnly/val/$SCENE/R.h5 \
  RANSAC-Tutorial-Data-ValOnly/val/$SCENE/T.h5 \
  RANSAC-Tutorial-Data-ValOnly/val/$SCENE/Fgt.h5 2>/dev/null || true
rm -f "$TMP"
if [ ! -d data/rb2025 ]; then
    git clone --depth 1 https://github.com/ducha-aiki/ransac-benchmark-2025 data/rb2025
fi
ls -la "$DEST"
