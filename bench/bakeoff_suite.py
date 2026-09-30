#!/usr/bin/env python3
"""Matched bake-off suite for GLM-5.3-Flash arms on head loopback. stdlib only.

usage: suite.py --arm NAME --out DIR [--base http://127.0.0.1:8000] [--model glm-5.3-flash-nvfp4-sm121]
                [--only a,b,c] [--skip a,b]
Sections (in order): api, correct, korean, count, c1, repo, c8think, c8x19k, depth, prefill128k, ping120k
Each section writes DIR/<section>.json. Memory of both ranks is sampled every 5 s into DIR/mem.jsonl.
Owned guard: if either rank MemAvailable < 2 GiB for 3 consecutive samples, stop (cancel our streams).
"""
import argparse, json, os, pathlib, re, subprocess, sys, threading, time, urllib.request, uuid

A = None
STOP = threading.Event()
MEM = {'head': [], 'worker': []}
LOW = {'n': 0}
FLOOR = 2 * 1024 ** 3
WSSH = ['ssh', '-i', os.environ.get('WORKER_KEY', os.path.expanduser('~/.ssh/id_ed25519')), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', os.environ.get('WORKER_SSH', 'user@worker')]
CAL = pathlib.Path(__file__).with_name('calibration.json')


def now():
    return time.time()


def save(name, obj):
    p = pathlib.Path(A.out) / f'{name}.json'
    p.write_text(json.dumps(obj, indent=2) + '\n')


def http(path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(A.base + path, data=data, headers={'Content-Type': 'application/json'},
                                 method='GET' if data is None else 'POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode('utf-8', 'replace')
    try:
        return json.loads(raw)
    except Exception:
        return raw


MK = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total',
      'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total', 'vllm:spec_decode_num_accepted_tokens_total',
      'vllm:spec_decode_num_draft_tokens_total', 'vllm:prompt_tokens_total', 'vllm:generation_tokens_total')


def metrics():
    try:
        t = http('/metrics', timeout=5)
        if not isinstance(t, str):
            return {}
    except Exception:
        return {}
    o = {}
    for k in MK:
        m = re.findall(r'^' + re.escape(k) + r'(?:\{[^}]*\})? ([0-9.eE+-]+)', t, re.M)
        if m:
            o[k] = sum(float(x) for x in m)
    return o


# ---------------- memory sampling (both ranks, 5 s) ----------------
def meminfo():
    return next(int(l.split()[1]) * 1024 for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))


def mem_loop():
    fh = open(pathlib.Path(A.out) / 'mem.jsonl', 'a')
    wp = subprocess.Popen(WSSH + ['for i in $(seq 1 2880); do grep MemAvailable /proc/meminfo; sleep 5; done'],
                          stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    import atexit
    atexit.register(lambda: wp.poll() is None and wp.kill())
    wlast = {'v': None}

    def wread():
        for line in wp.stdout:
            try:
                wlast['v'] = int(line.split()[1]) * 1024
            except Exception:
                pass
    threading.Thread(target=wread, daemon=True).start()
    while not STOP.is_set():
        h = meminfo(); w = wlast['v']
        MEM['head'].append(h)
        if w is not None:
            MEM['worker'].append(w)
        fh.write(json.dumps({'t': now(), 'head': h, 'worker': w}) + '\n'); fh.flush()
        if min(h, w if w is not None else h) < FLOOR:
            LOW['n'] += 1
            if LOW['n'] >= 3:
                (pathlib.Path(A.out) / 'GUARD_MEMORY').write_text(json.dumps({'t': now(), 'head': h, 'worker': w}))
                STOP.set()
        else:
            LOW['n'] = 0
        STOP.wait(5)
    wp.terminate()


def mem_window(t0, t1):
    rows = [json.loads(l) for l in (pathlib.Path(A.out) / 'mem.jsonl').read_text().splitlines() if l.strip()]
    rows = [r for r in rows if t0 - 5 <= r['t'] <= t1 + 5]
    hs = [r['head'] for r in rows if r['head']]; ws = [r['worker'] for r in rows if r['worker']]
    return {'min_head_gib': round(min(hs) / 2 ** 30, 2) if hs else None,
            'min_worker_gib': round(min(ws) / 2 ** 30, 2) if ws else None}


# ---------------- streaming request ----------------
def chat_stream(messages, max_tokens, think, min_tokens=None, extra=None, timeout=3600, row=None):
    body = {'model': A.model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': bool(think), **({'reasoning_effort': 'high'} if think else {})}}
    if min_tokens and A.min_tokens:
        body['min_tokens'] = min_tokens
    if extra:
        body.update(extra)
    row = row if row is not None else {}
    row.update({'ev': [], 'content': '', 'reasoning': '', 'usage': None, 'finish': None, 'error': None, 'submit': now()})
    req = urllib.request.Request(A.base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if STOP.is_set():
                    row['error'] = 'cancelled_by_guard'; break
                ln = raw.decode('utf-8', 'replace').strip()
                if not ln.startswith('data: ') or ln == 'data: [DONE]':
                    continue
                e = json.loads(ln[6:])
                if e.get('usage'):
                    row['usage'] = e['usage']
                for ch in e.get('choices') or []:
                    d = ch.get('delta') or {}
                    c = d.get('content') or ''; r = d.get('reasoning_content') or d.get('reasoning') or ''
                    if c or r:
                        row['ev'].append(now())
                    row['content'] += c; row['reasoning'] += r
                    row['finish'] = ch.get('finish_reason') or row['finish']
    except Exception as ex:
        row['error'] = repr(ex)[:300]
    row['end'] = now()
    return row


def summarize_row(r):
    u = r.get('usage') or {}; ct = u.get('completion_tokens'); ev = r['ev']
    dec = (ct - 1) / (ev[-1] - ev[0]) if ct and len(ev) > 1 and ev[-1] > ev[0] else None
    return {'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': ct, 'finish': r['finish'], 'error': r['error'],
            'cached_tokens': (u.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'ttft_s': round(ev[0] - r['submit'], 3) if ev else None, 'decode_tok_s': round(dec, 2) if dec else None,
            'e2e_s': round(r['end'] - r['submit'], 2), 'first': ev[0] if ev else None, 'last': ev[-1] if ev else None,
            'reasoning_chars': len(r['reasoning']), 'content_head': r['content'][:160]}


def sampler(tag, ev, out):
    while not ev.is_set():
        m = metrics()
        if m:
            out.append({'t': now(), 'running': m.get('vllm:num_requests_running'), 'waiting': m.get('vllm:num_requests_waiting'),
                        'kv': m.get('vllm:kv_cache_usage_perc')})
        ev.wait(0.5)


def wave(tag, msg_sets, max_t, think, min_t=None, extra=None, stagger=None):
    n = len(msg_sets); m0 = metrics(); rows = [dict() for _ in range(n)]; samp = []; ev = threading.Event()
    th = threading.Thread(target=sampler, args=(tag, ev, samp), daemon=True); th.start()
    bar = threading.Barrier(n)

    def go(i):
        bar.wait()
        if stagger and stagger[i]:
            time.sleep(stagger[i])
        chat_stream(msg_sets[i], max_t, think, min_t, extra, row=rows[i])
    ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    t0 = now(); [t.start() for t in ts]; [t.join() for t in ts]; t1 = now()
    ev.set(); th.join(timeout=3); m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in MK if k.endswith('_total') and k in m1}
    per = [summarize_row(r) for r in rows]
    fs = [p['first'] for p in per if p['first']]; ls = [p['last'] for p in per if p['last']]
    decs = sorted(p['decode_tok_s'] for p in per if p['decode_tok_s'])
    tot = sum(p['completion_tokens'] or 0 for p in per)
    q, h = d.get('vllm:prefix_cache_queries_total'), d.get('vllm:prefix_cache_hits_total')
    acc, dr = d.get('vllm:spec_decode_num_accepted_tokens_total'), d.get('vllm:spec_decode_num_draft_tokens_total')
    ptot = sum(p['prompt_tokens'] or 0 for p in per); ctot = sum(p['cached_tokens'] or 0 for p in per)
    s = {'tag': tag, 'n': n, 'wall_s': round(t1 - t0, 2), 'errors': sum(1 for p in per if p['error']),
         'completion_tokens_total': tot, 'prompt_tokens_each': [p['prompt_tokens'] for p in per],
         'aggregate_e2e_tok_s': round(tot / (t1 - t0), 2),
         'decode_median_tok_s': decs[len(decs) // 2] if decs else None,
         'decode_min_tok_s': decs[0] if decs else None,
         'all_overlap_s': round(max(0.0, min(ls) - max(fs)), 2) if len(fs) == n and len(ls) == n else 0.0,
         'ttft_min_s': min((p['ttft_s'] for p in per if p['ttft_s'] is not None), default=None),
         'ttft_median_s': sorted(p['ttft_s'] for p in per if p['ttft_s'] is not None)[len(fs) // 2] if fs else None,
         'ttft_max_s': max((p['ttft_s'] for p in per if p['ttft_s'] is not None), default=None),
         'peak_running': max((x['running'] or 0 for x in samp), default=None),
         'peak_waiting': max((x['waiting'] or 0 for x in samp), default=None),
         'peak_kv_pct': round(100 * max((x['kv'] or 0 for x in samp), default=0), 1) if samp else None,
         'metric_cache_hit_pct': round(100 * h / q, 1) if q else None,
         'usage_cached_pct': round(100 * ctot / ptot, 1) if ptot and any(p['cached_tokens'] is not None for p in per) else None,
         'spec_acceptance_pct': round(100 * acc / dr, 1) if dr else None,
         'new_preemptions': d.get('vllm:num_preemptions_total'),
         **mem_window(t0, t1)}
    return s, per, rows


# ---------------- prompt builders ----------------
def ledger(nrec, i, salt):
    recs = [f'Record {salt}-{k}: shipment {((k * 7919 + i * 104729) % 1000003):07d} to depot {k % 37}; status '
            f'{"delivered" if k % 3 else "pending"}; weight {(k * 13 + i) % 900 + 100} kg.' for k in range(nrec)]
    return (f'Session {salt}. You are reviewing a private logistics ledger for client {i}.\n' + '\n'.join(recs) +
            '\nBriefly describe what this ledger contains and list three notable patterns.')


def nrec_for(target):
    cal = json.loads(CAL.read_text()) if CAL.exists() else {}
    key = str(target)
    if key in cal:
        return cal[key]
    # calibrate with /tokenize (vLLM arms); fallback to ratio
    n = max(10, target // 30)
    try:
        for _ in range(5):
            c = http('/tokenize', {'model': A.model, 'messages': [{'role': 'user', 'content': ledger(n, 0, 'a' * 12)}]}, 120)['count']
            if 0.985 * target <= c <= 1.01 * target:
                break
            n = max(10, int(n * target / c))
        cal[key] = n; CAL.write_text(json.dumps(cal, indent=1))
    except Exception:
        n = int(target / 34.5)
    return n


# ---------------- sections ----------------
def s_api():
    out = {'models': http('/v1/models')}
    ids = [m['id'] for m in out['models'].get('data', [])]
    out['alias_present'] = os.environ.get('MODEL', 'glm-5.3-flash') in ids
    out['max_model_len'] = [m.get('max_model_len') for m in out['models'].get('data', [])]
    try:
        with urllib.request.urlopen(A.base + '/health', timeout=5) as r:
            out['health'] = r.status
    except Exception as e:
        out['health'] = repr(e)
    # default thinking check: no chat_template_kwargs -> should produce reasoning
    body = {'model': A.model, 'messages': [{'role': 'user', 'content': 'What is 12*13? Answer with the number.'}], 'max_tokens': 2000,
            'temperature': 0}
    r = http('/v1/chat/completions', body, 600)
    msg = r['choices'][0]['message']
    out['default_thinking'] = {'reasoning_chars': len(msg.get('reasoning_content') or msg.get('reasoning') or ''),
                               'content': (msg.get('content') or '')[:200], 'usage': r.get('usage')}
    out['default_thinking_ok'] = out['default_thinking']['reasoning_chars'] > 0 and '156' in (msg.get('content') or '')
    save('api', out); return out


KNOWN = [('What is 17 * 23? Reply with only the number.', '391'),
         ('What is the capital of Australia? Reply with one word.', 'canberra'),
         ('How many days are in a leap year? Reply with only the number.', '366'),
         ('What is the chemical symbol for gold? Reply with only the symbol.', 'au'),
         ('Reverse the string "bakeoff". Reply with only the result.', 'ffoekab'),
         ('What is 2 to the power of 20? Reply with only the number.', '1048576')]


def s_correct():
    res = []
    for think in (False, True):
        for q, a in KNOWN:
            body = {'model': A.model, 'messages': [{'role': 'user', 'content': q}], 'max_tokens': 3000 if think else 64,
                    'temperature': 0, 'chat_template_kwargs': {'enable_thinking': think}}
            try:
                r = http('/v1/chat/completions', body, 900); c = (r['choices'][0]['message'].get('content') or '')
            except Exception as e:
                c = 'ERR ' + repr(e)
            res.append({'q': q, 'think': think, 'expect': a, 'got': c[:200], 'ok': a in c.lower().replace(',', '')})
    out = {'passed': sum(r['ok'] for r in res), 'total': len(res), 'rows': res}
    save('correct', out); return out


KOREAN = ('서울은 대한민국의 수도이며 한강을 따라 발전한 도시이다. 조선 시대부터 정치와 문화의 중심지였고, 오늘날에는 '
          '세계적인 기술 기업과 대학, 박물관이 모여 있다. 봄에는 벚꽃이 피고 가을에는 단풍이 아름다워 많은 관광객이 찾는다. '
          '지하철과 버스가 잘 연결되어 있어 이동이 편리하며, 전통 시장에서는 떡볶이와 김밥 같은 음식을 쉽게 맛볼 수 있다. '
          '경복궁과 창덕궁 같은 궁궐은 역사를 보여 주고, 북촌 한옥마을은 옛 주거 문화를 간직하고 있다.')


def s_korean():
    rows = []
    msgs = [{'role': 'user', 'content': '다음 문단을 한 글자도 바꾸지 말고 그대로 다시 써 주세요. 다른 말은 하지 마세요.\n\n' + KOREAN}]
    for i in range(6):
        body = {'model': A.model, 'messages': msgs, 'max_tokens': 600, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
        r = http('/v1/chat/completions', body, 600); c = (r['choices'][0]['message'].get('content') or '').strip()
        rows.append({'exact': c == KOREAN, 'contains': KOREAN in c, 'len': len(c), 'fffd': c.count('\ufffd'),
                     'nonhangul_mid': len(re.findall(r'[\uac00-\ud7a3][^\s\uac00-\ud7a3.,!?\'"()\-]+[\uac00-\ud7a3]', c)),
                     'text': c[:600]})
    # free-form Korean essays (rare tokens) - flag U+FFFD / foreign-script intrusions inside Hangul words
    for i in range(3):
        body = {'model': A.model, 'messages': [{'role': 'user', 'content': f'({i}) 한국의 사계절과 전통 음식에 대해 400자 정도의 짧은 글을 써 주세요.'}],
                'max_tokens': 900, 'temperature': 0 if i == 0 else 0.7, 'seed': 1234 + i, 'chat_template_kwargs': {'enable_thinking': False}}
        r = http('/v1/chat/completions', body, 600); c = (r['choices'][0]['message'].get('content') or '')
        rows.append({'essay': i, 'fffd': c.count('\ufffd'),
                     'nonhangul_mid': len(re.findall(r'[\uac00-\ud7a3][^\s\uac00-\ud7a3.,!?\'"()\-·~:;0-9]+[\uac00-\ud7a3]', c)),
                     'weird_scripts': len(re.findall(r'[\u0400-\u04ff\u0e00-\u0e7f\u0600-\u06ff\u3040-\u30ff]', c)), 'text': c[:900]})
    rep = [r for r in rows if 'exact' in r]
    out = {'verbatim_exact': sum(r['exact'] for r in rep), 'verbatim_total': len(rep),
           'fffd_total': sum(r['fffd'] for r in rows), 'suspect_mid_word': sum(r['nonhangul_mid'] for r in rows),
           'weird_scripts': sum(r.get('weird_scripts', 0) for r in rows), 'rows': rows}
    save('korean', out); return out


def s_count():
    runs = []
    for i in range(3):
        body = {'model': A.model, 'max_tokens': 900, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': 'Count from 1 to 200, one number per line, digits only.'}]}
        r = http('/v1/chat/completions', body, 600); t = (r['choices'][0]['message'].get('content') or '')
        nums = [l.strip() for l in t.strip().split('\n')]
        runs.append({'clean': nums == [str(k) for k in range(1, 201)], 'lines': len(nums),
                     'first_bad': next((j for j, (a, b) in enumerate(zip(nums, [str(k) for k in range(1, 201)])) if a != b), None)})
    out = {'clean': sum(r['clean'] for r in runs), 'total': len(runs), 'runs': runs}
    save('count', out); return out


PROSE = 'Write a detailed 700-word essay on the history of lighthouses, from the Pharos of Alexandria to modern automated beacons.'
CODE = ('Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, '
        'plus a pytest test suite covering eviction, expiry and concurrency. Output only code.')


def s_c1():
    out = {}
    for name, p in (('prose', PROSE), ('code', CODE)):
        for think in (False, True):
            s, per, _ = wave(f'c1-{name}-{"high" if think else "off"}', [[{'role': 'user', 'content': f'[{uuid.uuid4().hex[:8]}] ' + p}]],
                             2048 if think else 1024, think)
            out[f'{name}_{"high" if think else "off"}'] = {**s, 'per': per}
            save('c1', out)
    return out


COUNTP = ('Generate a continuous deterministic sequence of decimal integers starting at 1, separated only by commas and spaces. '
          'Continue until the output limit; do not explain, summarize, or stop early.')


def s_repo():
    out = {}
    for c in (1, 8):
        waves = []
        for w in range(3):
            s, per, _ = wave(f'repo-c{c}-{w}', [[{'role': 'user', 'content': COUNTP}] for _ in range(c)], 512, False, min_t=512)
            waves.append({**s, 'per': per if w == 2 else None})
        m = waves[1:]
        out[f'c{c}'] = {'waves': waves,
                        'per_stream_median': sorted(x['decode_median_tok_s'] or 0 for x in m)[len(m) // 2],
                        'aggregate_decode': round(sum(x['decode_median_tok_s'] or 0 for x in m) / len(m) * c, 2),
                        'aggregate_e2e': round(sum(x['aggregate_e2e_tok_s'] for x in m) / len(m), 2)}
        save('repo', out)
    return out


REASON_Q = ('A delivery service has 3 vans of capacity 7, 9, and 12 boxes. It must move 83 boxes. Every round can use each van once, '
            'but each round has a loading cost of 5 minutes and each van trip costs 11, 13, and 17 minutes respectively. Trips in a round '
            'run concurrently. Find a schedule minimizing total elapsed time, and explain why it is optimal.')


def s_c8think():
    msgs = [[{'role': 'user', 'content': f'Request {uuid.uuid4().hex[:10]}. ' + REASON_Q}] for _ in range(8)]
    s, per, _ = wave('c8think', msgs, 1536, True)
    out = {**s, 'per': per}; save('c8think', out); return out


def c8_history(target, tag, follow=True):
    n = nrec_for(target); salts = [uuid.uuid4().hex[:12] for _ in range(8)]
    cold = [[{'role': 'user', 'content': ledger(n, i, salts[i])}] for i in range(8)]
    s, per, rows = wave(f'{tag}-cold', cold, 256, False, min_t=256)
    out = {'target': target, 'nrec': n, 'cold': {**s, 'per': per}}
    if follow and s['errors'] == 0 and not STOP.is_set():
        rev = [cold[i] + [{'role': 'assistant', 'content': rows[i]['content']},
                          {'role': 'user', 'content': 'Now give a one-sentence summary of your previous answer.'}] for i in range(8)]
        s2, per2, _ = wave(f'{tag}-follow', rev, 256, False, min_t=256)
        out['follow'] = {**s2, 'per': per2}
    return out


def s_c8x19k():
    out = c8_history(19000, 'c8x19k'); save('c8x19k', out); return out


def s_depth():
    out = {'levels': []}
    for T in [int(x) for x in os.environ.get('DEPTHS', '32000,48000,64000').split(',')]:
        if STOP.is_set():
            break
        r = c8_history(T, f'depth{T // 1000}k', follow=False)
        c = r['cold']
        # 8-way = all 8 decoding at the same time (overlap>0), no preemption, no errors, no guard
        ok = c['errors'] == 0 and not c['new_preemptions'] and c['all_overlap_s'] > 0 and not STOP.is_set()
        r['ok_8way_no_wait'] = ok
        out['levels'].append(r); save('depth', out)
        if not ok:
            break
    good = [l['target'] for l in out['levels'] if l['ok_8way_no_wait']]
    out['max_depth_ok'] = max(good) if good else None
    save('depth', out); return out


def s_prefill128k():
    n = nrec_for(128000)
    msgs = [[{'role': 'user', 'content': ledger(n, 77, uuid.uuid4().hex[:12])}]]
    s, per, _ = wave('prefill128k', msgs, 64, False)
    p = per[0]
    out = {**s, 'per': per, 'prefill_tok_s': round(p['prompt_tokens'] / p['ttft_s'], 1) if p['prompt_tokens'] and p['ttft_s'] else None,
           'decode_tok_s': p['decode_tok_s']}
    save('prefill128k', out); return out


def s_ping120k():
    n = nrec_for(120000)
    big = [{'role': 'user', 'content': ledger(n, 55, uuid.uuid4().hex[:12])}]
    ping = [{'role': 'user', 'content': f'[{uuid.uuid4().hex[:6]}] Say "pong".'}]
    s, per, _ = wave('ping120k', [big, ping], 32, False, stagger=[0, 8])
    out = {**s, 'per': per, 'big_ttft_s': per[0]['ttft_s'], 'ping_ttft_s': round(per[1]['ttft_s'] - 8, 2) if per[1]['ttft_s'] else None}
    # ping_ttft_s measured from its own submit (stagger sleeps before submit) -> use raw
    out['ping_ttft_s'] = per[1]['ttft_s']
    save('ping120k', out); return out


def s_toolcall():
    here = pathlib.Path(__file__).parent
    env = dict(os.environ, GLM_URL=A.base + '/v1/chat/completions', GLM_MODEL=A.model)
    res = {}
    for name, cmd in (('tensorfold_harness', ['python3', str(here / 'toolcall_harness.py'), '--reps', '2', '--stream',
                                               '--out', str(pathlib.Path(A.out) / 'toolcall_harness_raw.json')]),
                      ('mmastrac_corruption', ['python3', str(here / 'toolcall_corruption.py'), '--url', A.base, '--model', A.model,
                                               '--runs', '16', '--concurrency', '4'])):
        if name == 'mmastrac_corruption' and A.no_logprobs:
            res[name] = 'skipped: arm has no logprobs'; continue
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=5400)
        res[name] = {'rc': p.returncode, 'tail': p.stdout[-2500:], 'stderr': p.stderr[-800:]}
        save('toolcall', res)
    return res


SECTIONS = {'api': s_api, 'correct': s_correct, 'korean': s_korean, 'count': s_count, 'toolcall': s_toolcall, 'c1': s_c1,
            'repo': s_repo, 'c8think': s_c8think, 'c8x19k': s_c8x19k, 'depth': s_depth, 'prefill128k': s_prefill128k,
            'ping120k': s_ping120k}


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='http://127.0.0.1:8000'); ap.add_argument('--model', default=os.environ.get('MODEL', 'glm-5.3-flash'))
    ap.add_argument('--only', default=''); ap.add_argument('--skip', default='')
    ap.add_argument('--no-min-tokens', dest='min_tokens', action='store_false')
    ap.add_argument('--no-logprobs', action='store_true')
    A = ap.parse_args(); A.base = A.base.rstrip('/')
    pathlib.Path(A.out).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=mem_loop, daemon=True).start(); time.sleep(6)
    only = [x for x in A.only.split(',') if x]; skip = set(x for x in A.skip.split(',') if x)
    order = only or list(SECTIONS)
    log = pathlib.Path(A.out) / 'suite.log'
    for name in order:
        if name in skip:
            continue
        if STOP.is_set():
            break
        t0 = now()
        try:
            r = SECTIONS[name]()
            msg = 'ok'
        except Exception as e:
            msg = 'ERROR ' + repr(e)[:400]
        line = f'{time.strftime("%H:%M:%S", time.gmtime())}Z {name} {msg} {round(now() - t0)}s'
        print(line, flush=True)
        with log.open('a') as fh:
            fh.write(line + '\n')
    STOP.set()
    allm = {'min_head_gib': round(min(MEM['head']) / 2 ** 30, 2) if MEM['head'] else None,
            'min_worker_gib': round(min(MEM['worker']) / 2 ** 30, 2) if MEM['worker'] else None}
    save('memfloor', allm); print(json.dumps(allm))


if __name__ == '__main__':
    main()
