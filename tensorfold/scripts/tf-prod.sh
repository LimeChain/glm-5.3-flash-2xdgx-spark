#!/usr/bin/env bash
# Production wrapper around the TensorFold kit's scripts/serve.sh for 2x DGX Spark (run on the head node).
#   tf-prod.sh start|stop|restart|status|watch
# start   : evict clean page cache on both nodes (no sudo needed), serve.sh start, verify >= MIN_SLOTS slots, retry once
# stop    : serve.sh stop and write a DISABLED marker so `watch` leaves it down
# watch   : for cron (`* * * * * /path/tf-prod.sh watch`): restart after 3 consecutive failed /health checks
#
# Why the eviction: on GB10 unified memory, page cache counts against the kit's load-time slot rule. Without
# passwordless sudo (MEM_GATE_DROP_CACHES) a node that just copied weights boots with 1 slot instead of 8.
#
# Environment (defaults in brackets):
#   TF_KIT      path to the glm53-tensorfold-spark checkout on the head          [~/glm53-tensorfold-spark]
#   CONFIG      kit config file, relative to TF_KIT                                [config/prod.env]
#   WORKER_SSH  ssh target of the worker (key auth)                              [read from CONFIG]
#   EVICT_DIRS  dirs whose large files are evicted from page cache, both nodes   [~/.cache ~/models]
#   MIN_SLOTS   slots that must load before start is declared READY              [8]
#   STATE       state dir for log / markers                                      [~/.local/state/glm53-tf-prod]
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TF_KIT="${TF_KIT:-$HOME/glm53-tensorfold-spark}"; CONFIG="${CONFIG:-config/prod.env}"
MIN_SLOTS="${MIN_SLOTS:-8}"; EVICT_DIRS="${EVICT_DIRS:-$HOME/.cache $HOME/models}"
STATE="${STATE:-$HOME/.local/state/glm53-tf-prod}"; mkdir -p "$STATE"
LOG=$STATE/tf-prod.log; MARK=$STATE/DISABLED; FAILS=$STATE/fails
set -a
# shellcheck disable=SC1090
source "$TF_KIT/$CONFIG"
set +a
PORT="${PORT:-8000}"; NAME="${NAME:-glm53-tf}"
export CONFIG
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
healthy(){ [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "127.0.0.1:$PORT/health")" = 200 ]; }
slots(){ docker logs "$NAME-r0" 2>&1 | grep -oE 'batching [0-9]+ requests' | tail -1 | grep -oE '[0-9]+'; }
evict(){
  # shellcheck disable=SC2086
  python3 "$HERE/evict_cache.py" $EVICT_DIRS >> "$LOG" 2>&1
  ssh -o BatchMode=yes "$WORKER_SSH" "python3 - $EVICT_DIRS" < "$HERE/evict_cache.py" >> "$LOG" 2>&1
}
serve(){ (cd "$TF_KIT" && timeout 1800 scripts/serve.sh "$@" >> "$LOG" 2>&1); }
start(){
  for a in 1 2; do
    log "start attempt $a"; evict; serve start
    n=$(slots); log "health=$(healthy && echo ok || echo bad) slots=${n:-?}"
    if healthy && [ "${n:-0}" -ge "$MIN_SLOTS" ]; then rm -f "$MARK" "$FAILS"; log "READY ${n} slots"; return 0; fi
    serve stop
  done
  log "START FAILED"; return 1
}
case "${1:-status}" in
  start) start ;;
  stop) touch "$MARK"; serve stop; log "stopped (DISABLED marker set)" ;;
  restart) serve stop; start ;;
  status) healthy && echo "healthy slots=$(slots)" || echo unhealthy; [ -e "$MARK" ] && echo DISABLED; tail -3 "$LOG" ;;
  watch)
    [ -e "$MARK" ] && exit 0
    pgrep -f "tf-prod.sh (start|restart)" | grep -vx "$$" >/dev/null && exit 0
    if healthy; then rm -f "$FAILS"; exit 0; fi
    n=$(( $(cat "$FAILS" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAILS"; log "watch: unhealthy tick $n"
    if [ "$n" -ge 3 ]; then log "watch: restarting"; serve stop; start; fi ;;
  *) echo "usage: $0 start|stop|restart|status|watch" >&2; exit 2 ;;
esac
