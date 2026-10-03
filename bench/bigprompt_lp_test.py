#!/usr/bin/env python3
"""Verifier-shaped test: ~140K-token prompt, logprobs + top_logprobs 20, reasoning high, max_tokens 65536, temp 0,
non-streaming then streaming, with short requests alongside. Also one malformed-shape request. Writes JSON."""
import json, os, sys, threading, time, urllib.request, urllib.error

OUT = sys.argv[1]
B = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:8000'
M = os.environ.get('MODEL', 'GLM-5.3-Flash-EXL3')


def post(b, t=3600):
    req = urllib.request.Request(B + '/v1/chat/completions', data=json.dumps(b).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=t) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {'HTTP': e.code, 'body': e.read().decode()[:300]}


def stream(b, t=3600):
    req = urllib.request.Request(B + '/v1/chat/completions', data=json.dumps(dict(b, stream=True)).encode(),
                                 headers={'Content-Type': 'application/json'})
    ct = ''; rc = 0; lps = []; fin = None; err = None; cc = cwl = 0
    with urllib.request.urlopen(req, timeout=t) as r:
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith('data: ') or l == 'data: [DONE]':
                continue
            e = json.loads(l[6:])
            if 'error' in e:
                err = e['error']
            for ch in e.get('choices') or []:
                d = ch.get('delta') or {}; c = d.get('content') or ''; ct += c; rc += len(d.get('reasoning_content') or '')
                lp = (ch.get('logprobs') or {}).get('content')
                if c:
                    cc += 1; cwl += lp is not None
                if lp:
                    lps += lp
                fin = ch.get('finish_reason') or fin
    return dict(finish=fin, error=err, reasoning_chars=rc, content=ct[:120], n_logprobs=len(lps),
                tokens_rebuild_content=''.join(x['token'] for x in lps) == ct,
                valid=bool(lps) and all(x['logprob'] <= 0 and len(x['top_logprobs']) == 20 for x in lps),
                content_chunks_with_logprobs=f'{cwl}/{cc}')


recs = '\n'.join(f'Record {k}: shipment {(k * 7919) % 1000003:07d} to depot {k % 37}; weight {(k * 13) % 900 + 100} kg; '
                 f'status {"delivered" if k % 3 else "pending"}.' for k in range(4300))


def big(tag):
    return [{'role': 'system', 'content': 'You are a verifier. Read the ledger and answer.'},
            {'role': 'user', 'content': f'[{tag}] {recs}\n\nHow many records are pending? Reply with a short answer.'}]


body = dict(model=M, max_tokens=65536, temperature=0, reasoning_effort='high', logprobs=True, top_logprobs=20)
res = {'prompt_chars': len(recs)}
small = []
stop = threading.Event()


def pinger():
    i = 0
    while not stop.is_set():
        stop.wait(20)
        r = post({'model': M, 'messages': [{'role': 'user', 'content': f'{i} say ok'}], 'max_tokens': 40,
                  'chat_template_kwargs': {'enable_thinking': False}}, 600)
        small.append(r['choices'][0]['finish_reason'] if 'choices' in r else r)
        i += 1


th = threading.Thread(target=pinger); th.start()
t0 = time.time(); r = post(dict(body, messages=big('ns')))
if 'choices' in r:
    ch = r['choices'][0]; m = ch['message']; lp = (ch.get('logprobs') or {}).get('content') or []
    res['nonstream'] = dict(prompt_tokens=r['usage']['prompt_tokens'], finish=ch['finish_reason'],
                            reasoning_chars=len(m.get('reasoning_content') or ''), content=(m.get('content') or '')[:120],
                            n_logprobs=len(lp), tokens_rebuild_content=''.join(x['token'] for x in lp) == (m.get('content') or ''),
                            valid=bool(lp) and all(x['logprob'] <= 0 and len(x['top_logprobs']) == 20 for x in lp))
else:
    res['nonstream'] = r
res['nonstream']['seconds'] = round(time.time() - t0)
t0 = time.time(); res['stream'] = stream(dict(body, messages=big('st'))); res['stream']['seconds'] = round(time.time() - t0)
stop.set(); th.join()
res['concurrent_small'] = small
h = json.loads(urllib.request.urlopen(B + '/health', timeout=10).read())
res['health_after'] = {k: h.get(k) for k in ('ok', 'errors', 'fatal')}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
