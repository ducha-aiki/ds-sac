#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/tutorial ]; then
    git clone --depth 1 https://github.com/ducha-aiki/ransac-tutorial-2020-data data/tutorial
fi
# Homography archive URL: verify against data/tutorial/README.md ("homography" link)
URL="http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/homography.tar.gz"
if [ ! -f data/homography.tar.gz ]; then
    curl -fL -o data/homography.tar.gz "$URL"
fi
tar xzf data/homography.tar.gz -C data
find data -maxdepth 3 -type d | sort | head -30
