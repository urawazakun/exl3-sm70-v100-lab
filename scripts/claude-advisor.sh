#!/usr/bin/env bash
# claude-advisor.sh -- ask Claude Code (default: Opus 5.5) for advice, read-only, and log the answer.
#
# This is the "consult the senior engineer" path: it never edits or writes anything in the lab (no Write,
# no Edit), it may only read files and run read-only shell commands, and every consultation is stored so
# the reasoning behind a decision survives the session.
#
# Usage:
#   claude-advisor.sh <question.md> [model] [effort] [max_turns]
#   echo "short question" | claude-advisor.sh -            # question from stdin
#
# Notes:
#   * `opus` resolves to claude-opus-5-5 on this machine (verified 2026-09-25).
#   * Subscription is Claude Pro: the "cost" printed by --output-format json is the notional API price,
#     not a bill; track it anyway, it is the right relative measure between consultations.
#   * --max-turns bounds the agentic loop (default 40). A research question with several sub-parts needs
#     many calls: a run that hits the cap returns `subtype: error_max_turns` with an EMPTY answer and the
#     work is lost (measured 2026-09-25: 17 turns, USD 1.76, no answer). Give the question an explicit
#     tool-call budget in its text as well, and split multi-part questions into separate consultations.
#   * Read-only by construction: no Write/Edit; git is allowed read-only (log/show/diff/ls-tree/rev-parse)
#     because "is this in upstream? which commit?" cannot be answered without it.
set -uo pipefail
LAB=/h/exl3-lab
QFILE="${1:-}"
MODEL="${2:-opus}"
EFFORT="${3:-high}"
MAX_TURNS="${4:-40}"
SLUG=$(basename "${1:-consult}" .md | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/-/g' | cut -c1-28)
STAMP=$(date +%Y%m%d-%H%M%S)-$$-$SLUG
OUT="$LAB/logs/advisor-$STAMP.md"
JSON="$LAB/logs/advisor-$STAMP.json"

if [ -z "$QFILE" ]; then
    echo "usage: claude-advisor.sh <question.md|-> [model] [effort]"; exit 2
fi
if [ "$QFILE" = "-" ]; then
    QUESTION=$(cat)
elif [ -f "$QFILE" ]; then
    QUESTION=$(cat "$QFILE")
elif [ -f "$LAB/$QFILE" ]; then
    QUESTION=$(cat "$LAB/$QFILE")
else
    echo "question file not found: $QFILE"; exit 2
fi

{
  echo "# Consultation to Claude Code ($MODEL, effort $EFFORT) -- $STAMP"
  echo
  echo "Question asked:"
  echo '```'
  echo "$QUESTION"
  echo '```'
  echo
  echo "## Answer"
  echo
} > "$OUT"

cd "$LAB" || exit 1
claude -p "$QUESTION" \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --max-turns "$MAX_TURNS" \
    --output-format json \
    --allowedTools "Read,Grep,Glob,WebSearch,WebFetch,Bash(grep *),Bash(head *),Bash(tail *),Bash(ls *),Bash(wc *),Bash(cat *),Bash(curl *),Bash(find *),Bash(git log *),Bash(git show *),Bash(git diff *),Bash(git ls-tree *),Bash(git rev-parse *),Bash(git status *)" \
    > "$JSON" 2>"$OUT.stderr"

python - "$(cygpath -m "$JSON" 2>/dev/null || echo "$JSON")" "$(cygpath -m "$OUT" 2>/dev/null || echo "$OUT")" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding='utf-8', errors='replace').read()
out = open(sys.argv[2], 'a', encoding='utf-8')
try:
    d = json.loads(raw)
except Exception:
    out.write("(no parseable JSON answer)\n\n```\n" + raw[:2000] + "\n```\n")
    print("advisor: unparseable answer, see", sys.argv[2]); raise SystemExit(0)
ans = d.get('result') or ''
out.write(ans + "\n\n")
models = ", ".join((d.get('modelUsage') or {}).keys())
out.write(f"---\nmodel: {models} | turns: {d.get('num_turns')} | notional cost USD: {d.get('total_cost_usd')} | {d.get('duration_ms')} ms | session: {d.get('session_id')}\n")
out.close()
print(ans)
print("---")
print("model:", models, "| turns:", d.get('num_turns'), "| notional USD:", d.get('total_cost_usd'), "| session:", d.get('session_id'))
PY
echo "advice saved: $OUT"
