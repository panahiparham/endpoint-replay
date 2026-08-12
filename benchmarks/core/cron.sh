#!/usr/bin/env bash
#
# Cron entry point for the weekly benchmark. Runs `check`, and acts only on
# what's safe unattended: `finish` (git/gh, no MFA). Dispatching always stays
# a command a human runs by hand - see the repo README's "Weekly Automated
# Benchmarks" section. Meant for a host that holds a persistent, already
# -authenticated ssh ControlMaster to Vulcan (so it never hits the MFA
# prompt), not a laptop that sleeps.
#
# crontab -e:
#   0 * * * * PATH="$HOME/.local/bin:$PATH" /path/to/online-gsp/benchmarks/core/cron.sh
#
# Notifies by email (WEEKLY_BENCHMARK_NOTIFY, default below) via the local
# `mail` command - "due" only once per sha (it would otherwise refire every
# tick until dispatched), "finish" every time (success or failure), and a
# Vulcan connection failure once when it starts and once when it clears (not
# every tick of a multi-hour outage).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

NOTIFY="${WEEKLY_BENCHMARK_NOTIFY:-parham1@ualberta.ca}"
LOG="$REPO/.cluster/weekly-cron.log"
DUE_MARKER="$REPO/.cluster/weekly-cron.due-notified"
UNREACHABLE_MARKER="$REPO/.cluster/weekly-cron.unreachable"
mkdir -p "$(dirname "$LOG")"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

check_out=$(uv run benchmarks/core/weekly.py check 2>&1)
check_status=$?
log "check ($check_status): $check_out"

if [ "$check_status" != "30" ] && [ -f "$UNREACHABLE_MARKER" ]; then
  rm -f "$UNREACHABLE_MARKER"
  echo "Vulcan is reachable again as of this tick." \
    | mail -s "[online-gsp] weekly benchmark: Vulcan connection recovered" "$NOTIFY"
fi

case "$check_status" in
  0)
    ;;  # idle, or a dispatch is still queued - nothing to do
  10)
    due_sha=$(echo "$check_out" | grep -oE '[0-9a-f]{7,40}' | head -1)
    if [ "$due_sha" != "$(cat "$DUE_MARKER" 2>/dev/null || true)" ]; then
      echo "$check_out" | mail -s "[online-gsp] weekly benchmark due" "$NOTIFY"
      echo "$due_sha" > "$DUE_MARKER"
    fi
    ;;
  20)
    if finish_out=$(uv run benchmarks/core/weekly.py finish 2>&1); then
      log "finish: $finish_out"
      echo "$finish_out" | mail -s "[online-gsp] weekly benchmark PR opened" "$NOTIFY"
    else
      log "finish FAILED: $finish_out"
      echo "$finish_out" | mail -s "[online-gsp] weekly benchmark finish FAILED" "$NOTIFY"
    fi
    ;;
  30)
    if [ ! -f "$UNREACHABLE_MARKER" ]; then
      echo "$check_out" | mail -s "[online-gsp] weekly benchmark: can't reach Vulcan" "$NOTIFY"
      touch "$UNREACHABLE_MARKER"
    fi
    ;;
  *)
    log "check exited unexpectedly"
    echo "$check_out" | mail -s "[online-gsp] weekly benchmark check FAILED ($check_status)" "$NOTIFY"
    ;;
esac
