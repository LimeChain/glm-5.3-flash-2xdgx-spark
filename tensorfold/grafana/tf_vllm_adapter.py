#!/usr/bin/env python3
"""TensorFold -> vLLM-compatible Prometheus adapter (runs on the Spark head, stdlib only).

Serves 127.0.0.1:$ADAPTER_PORT/metrics (default 8892) with vllm:* series built from
  - TensorFold's own /metrics ($TF_METRICS_URL) (token counters, in-flight)
  - TensorFold's per-request log $TF_REQUEST_LOG = <HEAD_SESSIONS>/requests.jsonl (latency histograms, finish reasons)
If the backend is vLLM itself (it already exports vllm:*), this adapter emits nothing, so
dashboards never see duplicate series and switching back to vLLM needs no change.
"""
import json, os, re, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = os.environ.get('TF_METRICS_URL', 'http://127.0.0.1:8000/metrics')
LOG = os.environ.get('TF_REQUEST_LOG', os.path.expanduser('~/.cache/glm53-tf/sessions/requests.jsonl'))
PORT = int(os.environ.get('ADAPTER_PORT', '8892'))
POOL_PAGES = int(os.environ.get('TF_POOL_PAGES', '4097'))   # GLM53_TF_KV_POOL_TOKENS / 256 (+1)
SLOTS = int(os.environ.get('TF_SLOTS', '8'))

B_TTFT = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
B_TPOT = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]
B_E2E = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]
B_LEN = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]


class Hist:
    def __init__(self, bounds):
        self.b = bounds; self.c = [0] * len(bounds); self.n = 0; self.s = 0.0

    def add(self, v):
        if v is None:
            return
        self.n += 1; self.s += v
        for i, x in enumerate(self.b):
            if v <= x:
                self.c[i] += 1

    def lines(self, name, lab):
        out = [f'# TYPE {name} histogram']
        for x, c in zip(self.b, self.c):
            out.append(f'{name}_bucket{{{lab},le="{x}"}} {c}')
        out += [f'{name}_bucket{{{lab},le="+Inf"}} {self.n}', f'{name}_sum{{{lab}}} {self.s}', f'{name}_count{{{lab}}} {self.n}']
        return out


LOCK = threading.Lock()
ST = {'off': 0, 'ino': None, 'finish': {}, 'last_kv_free': None, 'since': 0.0,
      'h': {'ttft': Hist(B_TTFT), 'tpot': Hist(B_TPOT), 'e2e': Hist(B_E2E), 'plen': Hist(B_LEN), 'glen': Hist(B_LEN),
            'queue': Hist(B_E2E), 'prefill': Hist(B_E2E), 'decode': Hist(B_E2E)}}


def ingest():
    try:
        st = os.stat(LOG)
    except FileNotFoundError:
        return
    if ST['ino'] != st.st_ino or st.st_size < ST['off']:        # rotated / truncated: start over
        ST.update(off=0, ino=st.st_ino, finish={})
        for k, h in ST['h'].items():
            ST['h'][k] = Hist(h.b)
    with open(LOG, 'rb') as fh:
        fh.seek(ST['off'])
        data = fh.read()
    end = data.rfind(b'\n')
    if end < 0:
        return
    ST['off'] += end + 1
    for line in data[:end].splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get('ts') or 0) < ST['since']:      # finish reasons count the current server run only
            continue
        fin = r.get('finish') or ('error' if r.get('error') else 'stop')
        fin = {'tool_calls': 'stop', 'cancelled': 'abort'}.get(fin, fin)
        if r.get('error'):
            fin = 'error'
        ST['finish'][fin] = ST['finish'].get(fin, 0) + 1
        h = ST['h']
        q = r.get('queue_s') or 0.0; pf = r.get('prefill_s') or 0.0
        h['ttft'].add(r['first_s'] if r.get('first_s') is not None else q + pf)
        dt = r.get('decode_tokens') or 0
        if dt > 1 and r.get('decode_s'):
            h['tpot'].add(r['decode_s'] / dt)
        if r.get('ts') and r.get('start'):
            h['e2e'].add(r['ts'] - r['start'])
        h['plen'].add(r.get('prompt')); h['glen'].add(dt)
        h['queue'].add(q); h['prefill'].add(pf); h['decode'].add(r.get('decode_s'))
        if r.get('kv_free') is not None:
            ST['last_kv_free'] = r['kv_free']


def scrape_up():
    with urllib.request.urlopen(UP, timeout=5) as resp:
        return resp.read().decode('utf-8', 'replace')


def render():
    try:
        t = scrape_up()
    except Exception:
        return '# upstream unreachable\n'
    if 'vllm:' in t or 'tensorfold_' not in t:
        return '# upstream is not TensorFold (or already vLLM): adapter silent\n'
    v = {}
    model = 'unknown'
    for m in re.finditer(r'^(tensorfold_[a-z_]+)\{model="([^"]*)"\} ([0-9.eE+-]+)$', t, re.M):
        v[m.group(1)] = float(m.group(3)); model = m.group(2)
    with LOCK:
        up = v.get('tensorfold_uptime_seconds')
        if up is not None:
            since = time.time() - up
            if abs(since - ST['since']) > 30:       # a new server run: finish counts start again
                ST['since'] = since; ST['finish'] = {}
                ST['off'] = 0; ST['ino'] = None
                for k, hh in ST['h'].items():
                    ST['h'][k] = Hist(hh.b)
        ingest()
        lab = f'engine="0",model_name="{model}"'
        o = []
        def g(name, typ, val, extra=''):
            o.append(f'# TYPE {name} {typ}')
            o.append(f'{name}{{{lab}{extra}}} {val}')
        g('vllm:prompt_tokens_total', 'counter', v.get('tensorfold_prompt_tokens_total', 0))
        g('vllm:generation_tokens_total', 'counter', v.get('tensorfold_completion_tokens_total', 0))
        g('vllm:prefix_cache_queries_total', 'counter', v.get('tensorfold_prompt_tokens_total', 0))
        g('vllm:prefix_cache_hits_total', 'counter', v.get('tensorfold_cached_tokens_total', 0))
        inflight = v.get('tensorfold_requests_inflight', 0)
        g('vllm:num_requests_running', 'gauge', min(inflight, SLOTS))
        o.append('# TYPE vllm:num_requests_waiting gauge')
        o.append(f'vllm:num_requests_waiting{{{lab}}} {max(0.0, inflight - SLOTS)}')
        o.append('# TYPE vllm:num_requests_waiting_by_reason gauge')
        o.append(f'vllm:num_requests_waiting_by_reason{{{lab},reason="capacity"}} {max(0.0, inflight - SLOTS)}')
        o.append(f'vllm:num_requests_waiting_by_reason{{{lab},reason="deferred"}} 0')
        kvf = ST['last_kv_free']
        kv = 0.0 if kvf is None else max(0.0, min(1.0, (POOL_PAGES - kvf) / POOL_PAGES))
        g('vllm:kv_cache_usage_perc', 'gauge', kv)
        fin = dict(ST['finish'])
        fin['error'] = v.get('tensorfold_request_errors_total', fin.get('error', 0))
        o.append('# TYPE vllm:request_success_total counter')
        for reason in ('stop', 'length', 'abort', 'error', 'repetition'):
            o.append(f'vllm:request_success_total{{{lab},finished_reason="{reason}"}} {fin.get(reason, 0)}')
        h = ST['h']
        o += h['ttft'].lines('vllm:time_to_first_token_seconds', lab)
        o += h['tpot'].lines('vllm:request_time_per_output_token_seconds', lab)
        o += h['tpot'].lines('vllm:inter_token_latency_seconds', lab)
        o += h['e2e'].lines('vllm:e2e_request_latency_seconds', lab)
        o += h['queue'].lines('vllm:request_queue_time_seconds', lab)
        o += h['prefill'].lines('vllm:request_prefill_time_seconds', lab)
        o += h['decode'].lines('vllm:request_decode_time_seconds', lab)
        o += h['plen'].lines('vllm:request_prompt_tokens', lab)
        o += h['glen'].lines('vllm:request_generation_tokens', lab)
        o.append('# TYPE tf_adapter_info gauge')
        o.append(f'tf_adapter_info{{{lab},backend="tensorfold",source="tf_vllm_adapter"}} 1')
    return '\n'.join(o) + '\n'


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip('/') != '/metrics':
            self.send_response(404); self.end_headers(); return
        body = render().encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; version=0.0.4')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
