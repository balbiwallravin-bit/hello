#!/usr/bin/env bash
set -euo pipefail

READY_LIST="/home/lht/blurtrack/video_maked/train_segments_ready.txt"
SRC_ROOT="/home/lht/blurtrack/video_maked"
DST_ROOT="/home/lht/blurtrack/video_maked_ready"

mkdir -p "${DST_ROOT}"

added=0
while read -r seg; do
  [ -z "${seg}" ] && continue
  src="${SRC_ROOT}/${seg}"
  dst="${DST_ROOT}/${seg}"
  if [ ! -e "${dst}" ]; then
    ln -s "${src}" "${dst}"
    added=$((added+1))
  fi
done < "${READY_LIST}"

echo "[OK] added symlinks: ${added}"
echo "[OK] total ready symlinks: $(ls -1 "${DST_ROOT}" | wc -l)"
