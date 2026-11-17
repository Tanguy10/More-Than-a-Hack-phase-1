#!/usr/bin/env bash
# Download external datasets into MTAH_EXTERNAL_DIR (default: Git-ignored data/external).
# Never commit these datasets to the repository.
# bash "Modele 1 Tanguy/external/download_external.sh"  # public DFL + SkillCorner
# Pass soccertrack after accepting its Hugging Face terms and setting HF_TOKEN.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EXT="${MTAH_EXTERNAL_DIR:-$ROOT/data/external}"
mkdir -p "$EXT"
what="${1:-public}"

if [[ "$what" == "public" || "$what" == "dfl" ]]; then
  # DFL, Bassek et al. 2025 (Scientific Data), figshare 28196177, CC BY 4.0, ~2.6 GB.
  mkdir -p "$EXT/dfl_figshare" && cd "$EXT/dfl_figshare"
  curl -s "https://api.figshare.com/v2/articles/28196177" | python3 -I -c "
import json, sys
for f in json.load(sys.stdin)['files']: print(f['download_url'], f['name'])" |
  while read -r url name; do
    [[ -s "$name" ]] || { echo "-> $name"; curl -sL --retry 3 -o "$name" "$url"; }
  done
fi

if [[ "$what" == "public" || "$what" == "skillcorner" ]]; then
  # SkillCorner open data (MIT), 10 fps broadcast tracking, ~2 GB with git-lfs.
  mkdir -p "$EXT/skillcorner_opendata" && cd "$EXT/skillcorner_opendata"
  [[ -d repo ]] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -q https://github.com/SkillCorner/opendata.git repo
  cd repo && git lfs pull --include="data/matches/**"
fi

if [[ "$what" == "soccertrack" ]]; then
  # SoccerTrack v2 (CC BY 4.0), Hugging Face access after accepting the terms.
  # Exclude video (83 GB) and GSR (56 GB): fetch annotations, ball and calibration only.
  : "${HF_TOKEN:?Set HF_TOKEN (huggingface.co/settings/tokens)}"
  mkdir -p "$EXT/soccertrack_v2" && cd "$EXT/soccertrack_v2"
  curl -s -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/datasets/atomscott/soccertrack-v2/tree/main?recursive=true" |
  python3 -I -c "
import json, sys
for f in json.load(sys.stdin):
    p = f['path']
    if f['type'] == 'file' and p.split('/')[0] in ('bas', 'ball', 'raw') or p.endswith('.md'): print(p)" |
  while read -r path; do
    [[ -s "$path" ]] || { mkdir -p "$(dirname "$path")"; echo "-> $path"
      curl -sL --retry 3 -H "Authorization: Bearer $HF_TOKEN" -o "$path" \
        "https://huggingface.co/datasets/atomscott/soccertrack-v2/resolve/main/$path"; }
  done
fi

echo "ok -> $EXT"
