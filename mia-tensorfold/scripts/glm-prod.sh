#!/bin/bash
# Production control for GLM-5.3 Flash on 2x DGX Spark: Mia's TensorFold recipe, 8 streams, LimeChain patches.
# usage: glm-prod.sh start|stop|restart|status|watch          (run on the head)
#   start   : free page cache on both nodes (no sudo), start.sh, require /health ok:true and 8 streams
#   stop    : stop.sh + DISABLED marker (the watchdog leaves it down until start)
#   watch   : cron tick; unhealthy 3 ticks in a row (ok:false, no answer, or a rank's container down) -> restart
set -u
# Settings (environment or defaults):
#   KIT      checkout of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (its scripts/local.sh: WORKER, PARALLEL=8,
#            IMAGE=glm53-lc:prod, PORT, ...)
#   STATE    where this script keeps its log, lock and markers
#   WORKER   the worker's ssh target (the same as in scripts/local.sh)
KIT=${KIT:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold}; K=$KIT
STATE=${STATE:-$HOME/.cache/glm53-lc}; mkdir -p "$STATE"; B=$(cd "$(dirname "$0")" && pwd)
WORKER=${WORKER:-$(grep -E '^WORKER=' "$K/scripts/local.sh" | tail -1 | cut -d= -f2)}
W="ssh -o BatchMode=yes $WORKER"; LOG=$STATE/prod.log; MARK=$STATE/prod.DISABLED; FAILS=$STATE/prod.fails
NAME=glm53-flash-tf; PORT=${PORT:-$(grep -E '^PORT=' "$K/scripts/local.sh" | tail -1 | cut -d= -f2)}; PORT=${PORT:-8888}
EXTRA=(--reasoning-effort high)
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
want(){ grep -E '^PARALLEL=' "$K"/scripts/local.sh | tail -1 | cut -d= -f2; }
streams(){ curl -s --max-time 10 "127.0.0.1:$PORT/health" | python3 -c 'import json,sys; print(json.load(sys.stdin)["streams"]["max"])' 2>/dev/null; }
healthy(){ curl -s --max-time 10 "127.0.0.1:$PORT/health" | grep -q '"ok": *true' || return 1
  [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || return 1
  [ "$($W "docker inspect -f '{{.State.Running}}' $NAME" 2>/dev/null)" = true ]; }
freemem(){
  docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || python3 "$B"/free_page_cache.py 6 >> "$LOG" 2>&1
  $W "docker ps --format '{{.Names}}' | grep -q '^$NAME\$' || python3 -" < "$B"/free_page_cache.py >> "$LOG" 2>&1
}
stop_both(){ (cd "$K" && bash stop.sh >> "$LOG" 2>&1); docker rm $NAME >/dev/null 2>&1; $W "docker rm $NAME" >/dev/null 2>&1; }
start(){
  for a in 1 2; do
    log "start attempt $a"; stop_both; freemem
    (cd "$K" && PREPARE=0 timeout 1800 bash start.sh "${EXTRA[@]}" >> "$LOG" 2>&1)
    n=$(streams); log "health=$(healthy && echo ok || echo bad) streams=${n:-?}"
    if healthy && [ "${n:-0}" -ge "$(want)" ]; then rm -f "$MARK" "$FAILS"; log "READY $n streams"; return 0; fi
  done
  log "START FAILED"; return 1
}
# one start/stop at a time: a watch tick never starts a second restart while one runs (flock on LOCK)
LOCK=$STATE/prod.lock
locked(){ exec 9>"$LOCK"; flock -n 9 || { log "$1: another start/stop holds the lock; skipped"; exit 0; }; }
case "${1:-status}" in
  start) locked start; start ;;
  stop) touch "$MARK"; exec 9>"$LOCK"; flock 9; stop_both; log "stopped (DISABLED marker set)" ;;
  restart) locked restart; start ;;
  status) healthy && echo "healthy streams=$(streams)" || echo unhealthy; [ -e "$MARK" ] && echo DISABLED; tail -3 "$LOG" ;;
  watch)
    [ -e "$MARK" ] && exit 0
    exec 9>"$LOCK"; flock -n 9 || exit 0          # a start (manual or a previous tick's) is running
    if healthy; then rm -f "$FAILS"; exit 0; fi
    n=$(( $(cat "$FAILS" 2>/dev/null || echo 0) + 1 )); echo $n > "$FAILS"; log "watch: unhealthy tick $n"
    if [ $n -ge 3 ]; then log "watch: restarting"; start; fi ;;
esac
