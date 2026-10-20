#!/usr/bin/env bash
# Resumable N-way ranged downloader for one large HTTP(S) file.
#
# Correctness notes (verified by experiment on this host):
#   * `curl --append -r A-B -o f` does NOT append — it truncates and writes, so
#     attempts must land in a per-attempt temp file and only be appended to the
#     part after the response is validated;
#   * a range-ignoring 200 response declares the whole remaining body, so
#     --max-filesize is set to the part span to refuse it up-front;
#   * every third attempt goes through $DL_PROXY when set, because one route can
#     break while the other still works.
#
# usage: dl_ranged.sh <url> <dest> <total_bytes> [parts] [sha256]
# env:   DL_PROXY=http://host:port   (optional alternate route)
set -u
URL=$1
DEST=$2
TOTAL=$3
PARTS=${4:-4}
SHA=${5:-}
STAGE="${DEST}.stage"
mkdir -p "$STAGE"
SPAN=$(( (TOTAL + PARTS - 1) / PARTS ))
ATTEMPTS_MAX=5000

curl_attempt() {  # <start> <end> <have> <outfile> <attempt>
  local start=$1 end=$2 have=$3 out=$4 attempt=$5
  local from=$(( start + have ))
  local span=$(( end - start + 1 ))
local common=(-sSL --http1.1 --connect-timeout 20 --max-time 240
                --speed-limit 4096 --speed-time 30
                --max-filesize "$span"
                -r "$from-$end" -o "$out" -w '%{http_code}')
  if [ -n "${DL_PROXY:-}" ] && (( attempt % 3 == 0 )); then
    http_proxy=$DL_PROXY https_proxy=$DL_PROXY curl "${common[@]}" "$URL" 2>/dev/null
  else
    curl "${common[@]}" "$URL" 2>/dev/null
  fi
}

worker() {
  local i=$1
  local start=$(( i * SPAN ))
  local end=$(( start + SPAN - 1 ))
  (( end > TOTAL - 1 )) && end=$(( TOTAL - 1 ))
  local span=$(( end - start + 1 ))
  local part="$STAGE/part.$i" tmp="$STAGE/tmp.$i"
  local have=0 got=0 code attempts=0
  [ -f "$part" ] && have=$(stat -c %s "$part")

  while (( have < span )); do
    if (( attempts++ >= ATTEMPTS_MAX )); then
      echo "part.$i: giving up after $ATTEMPTS_MAX attempts" >&2
      exit 1
    fi
    rm -f "$tmp"
    code=$(curl_attempt "$start" "$end" "$have" "$tmp" "$attempts")
    got=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
    if [ "$code" = "206" ] && (( got > 0 && got <= span - have )); then
      cat "$tmp" >> "$part"
      have=$(( have + got ))
    else
      (( attempts % 10 == 0 )) && echo "part.$i: retry (code=$code got=$got have=$have/$span)" >&2
      sleep 2
    fi
    (( attempts % 5 == 0 || have >= span )) && echo "part.$i: $have/$span" >&2
  done
  rm -f "$tmp"
}

for i in $(seq 0 $(( PARTS - 1 ))); do worker "$i" & done
wait

parts=()
for i in $(seq 0 $(( PARTS - 1 ))); do
  part="$STAGE/part.$i"
  want=$(( TOTAL - i * SPAN ))
  (( want > SPAN )) && want=$SPAN
  got=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if [ "$got" != "$want" ]; then echo "FAIL part.$i size $got != $want" >&2; exit 1; fi
  parts+=("$part")
done
cat "${parts[@]}" > "$DEST.tmp"
size=$(stat -c %s "$DEST.tmp")
if [ "$size" != "$TOTAL" ]; then echo "FAIL size mismatch: got $size want $TOTAL" >&2; exit 1; fi
if [ -n "$SHA" ]; then
  if ! echo "$SHA  $DEST.tmp" | sha256sum -c -; then echo "FAIL sha256 mismatch" >&2; exit 1; fi
fi
mv "$DEST.tmp" "$DEST"
rm -rf "$STAGE"
echo "OK $DEST ($size bytes)"