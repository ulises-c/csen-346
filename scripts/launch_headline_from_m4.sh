#!/bin/bash
# Start the headline rerun on ollie-cachy from m4-mini: wake the machine, check
# that it answers, then start scripts/rerun_headline_r9700.sh there as a systemd
# user unit (so it does not depend on this ssh session).
#
# Installed on m4-mini as ~/.local/bin/csen346-headline-launch and run by cron.
# The crontab line carries the marker "csen346-headline-once"; this script
# removes that line when it runs, so the job fires one time only.
#
#   csen346-headline-launch            wake, verify, start
#   csen346-headline-launch --dry-run  wake, verify, run the preflight checks, start nothing
#
# Log:  ~/Library/Logs/csen346-headline-launch.log (on m4-mini)
# Stop: ssh ollie-cachy 'systemctl --user stop csen346-headline'
# Env:  CACHY_HOST, CACHY_WAKE, CACHY_WORKTREE, LATEST_START_HOUR (13), RUN_ENV ("K=V K=V")
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin

HOST="${CACHY_HOST:-ollie-cachy}"
WAKE="${CACHY_WAKE:-$HOME/.local/bin/wake-cachy}"
WORKTREE="${CACHY_WORKTREE:-github/csen-346-main}" # relative to the remote home
UNIT=csen346-headline
# The run takes about 5.5 hours and the GPU is loud; do not start one that would run into the night.
LATEST_START_HOUR="${LATEST_START_HOUR:-13}"
LOG="$HOME/Library/Logs/csen346-headline-launch.log"
MODE="${1:-run}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
say() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [$MODE] $*"; }
reachable() { ssh -n -o ConnectTimeout=5 -o BatchMode=yes "$HOST" true 2>/dev/null; }
remote() { ssh -n -o ConnectTimeout=10 -o BatchMode=yes "$HOST" "$@"; }

if [[ "$MODE" == "run" ]]; then
  # One-shot: drop the crontab entry first, whatever happens next.
  if crontab -l 2>/dev/null | grep -q 'csen346-headline-once'; then
    crontab -l | grep -v 'csen346-headline-once' | crontab - && say "crontab entry removed"
  fi
  if (( 10#$(date +%H) > LATEST_START_HOUR )); then
    say "FAILED: it is past ${LATEST_START_HOUR}:59, too late to start a 5.5-hour run today"
    exit 1
  fi
fi

# 1. Wake
if reachable; then
  say "$HOST already awake"
else
  [[ -x "$WAKE" ]] || { say "FAILED: no $WAKE"; exit 1; }
  say "sending Wake-on-LAN"
  "$WAKE" >/dev/null
  woke=0
  for i in $(seq 1 72); do # up to 6 minutes
    if reachable; then woke=1; say "$HOST answers ssh after about $((i * 5)) s"; break; fi
    (( i % 12 == 0 )) && "$WAKE" >/dev/null
    sleep 5
  done
  if [[ $woke -ne 1 ]]; then
    say "FAILED: no ssh answer from $HOST after 360 s; nothing started"
    exit 1
  fi
  sleep 45 # let the user session and the GPU driver settle after boot or resume
fi

# 2. Verify it is really up and usable
if ! remote "uptime; cd $WORKTREE && git log --oneline -1"; then
  say "FAILED: $HOST answered ssh but the worktree check failed; nothing started"
  exit 1
fi

if [[ "$MODE" == "--dry-run" ]]; then
  remote "cd $WORKTREE && bash scripts/rerun_headline_r9700.sh --preflight"
  say "dry run finished (preflight exit $?)"
  exit 0
fi

# 3. Start
if remote "systemctl --user is-active --quiet $UNIT"; then
  say "unit $UNIT is already running on $HOST; not starting a second one"
  exit 0
fi
env_args=""
for kv in ${RUN_ENV:-}; do env_args+=" -E $kv"; done
# shellcheck disable=SC2029  # $WORKTREE and $env_args are meant to expand here
if ! remote "systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit $UNIT --collect -p WorkingDirectory=\$HOME/$WORKTREE -E HOME=\$HOME$env_args /usr/bin/bash scripts/rerun_headline_r9700.sh"; then
  say "FAILED: could not start the unit"
  exit 1
fi
say "unit $UNIT started; checking in 120 s"
sleep 120
remote "systemctl --user is-active $UNIT; tail -4 \$HOME/csen346-runs/STATUS"
say "launch finished"
