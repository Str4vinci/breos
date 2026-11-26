#!/usr/bin/env bash
# Study 16: build one venv per dependency combination under data/envs/<name>.
# "lock" = the uv.lock versions; others pin one package away from the lock or resolve fresh.
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
ENVS=data/envs
mkdir -p "$ENVS"
uv export --project "$REPO" --frozen --extra fast --extra optimization --no-emit-project --no-hashes -o data/lock-req.txt -q

mk() {  # name python [extra uv pip args...]
  local name=$1 py=$2; shift 2
  [ -x "$ENVS/$name/bin/python" ] && [ -f "$ENVS/$name/.ok" ] && return 0
  rm -rf "$ENVS/$name"
  uv venv -q --python "$py" "$ENVS/$name" || { echo "$name: venv failed"; return 1; }
  if uv pip install -q --python "$ENVS/$name/bin/python" "$@"; then
    touch "$ENVS/$name/.ok"; echo "$name: ok"
  else
    echo "$name: install failed"
  fi
}
LOCK=(-r data/lock-req.txt)
SELF=(-e "$REPO[fast,optimization]")
mk lock-py313 3.13 "${LOCK[@]}" "${SELF[@]}"
mk lock-py312 3.12 "${LOCK[@]}" "${SELF[@]}"
mk lock-py311 3.11 "${LOCK[@]}" "${SELF[@]}"
# One package moved off the lock (python 3.13), everything else locked.
lock_without() { grep -v "^$1==" data/lock-req.txt > "data/lock-req-no-$1.txt"; echo -r "data/lock-req-no-$1.txt"; }
mk pandas233-py313 3.13 $(lock_without pandas) "${SELF[@]}" "pandas==2.3.3"
mk pvlib014-py313 3.13 $(lock_without pvlib) "${SELF[@]}" "pvlib>=0.14,<0.15"
mk numpy22-py313 3.13 $(lock_without numpy) "${SELF[@]}" "numpy>=2.2,<2.3"
# Fresh resolutions.
mk lowest-py311 3.11 --resolution lowest-direct "${SELF[@]}"
mk highest-py313 3.13 --resolution highest "${SELF[@]}"
