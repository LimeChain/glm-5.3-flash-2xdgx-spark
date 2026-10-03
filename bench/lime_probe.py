#!/usr/bin/env python3
"""Lime Agent feature probes against an OpenAI endpoint (argv[1], default 127.0.0.1:8000; MODEL env). Prints one line per check."""
import json, os, sys, urllib.request, urllib.error, hashlib

B = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'
M = os.environ.get('MODEL', 'GLM-5.3-Flash-EXL3')
res = {}


def post(b, t=600):
    r = urllib.request.Request(B + '/v1/chat/completions', data=json.dumps(dict(b, model=M)).encode(),
                               headers={'Content-Type': 'application/json'})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=t).read())
    except urllib.error.HTTPError as e:
        return {'HTTP': e.code, 'body': e.read().decode()[:300]}


def stream(b, t=600):
    r = urllib.request.Request(B + '/v1/chat/completions', data=json.dumps(dict(b, model=M, stream=True)).encode(),
                               headers={'Content-Type': 'application/json'})
    ct = ''; rc = 0; lps = []; fin = None; cc = cwl = 0; err = None
    try:
        for raw in urllib.request.urlopen(r, timeout=t):
            l = raw.decode().strip()
            if not l.startswith('data: ') or l == 'data: [DONE]':
                continue
            e = json.loads(l[6:])
            if 'error' in e:
                err = e['error']
            for ch in e.get('choices') or []:
                d = ch.get('delta') or {}; c = d.get('content') or ''; ct += c
                rc += len(d.get('reasoning_content') or '')
                lp = (ch.get('logprobs') or {}).get('content')
                if c:
                    cc += 1; cwl += lp is not None
                if lp:
                    lps += lp
                fin = ch.get('finish_reason') or fin
    except urllib.error.HTTPError as e:
        return {'HTTP': e.code, 'body': e.read().decode()[:300]}
    return dict(content=ct, rc=rc, lps=lps, fin=fin, chunks=f'{cwl}/{cc}', err=err)


def lp_ok(lps, top):
    return bool(lps) and all(x['logprob'] <= 0 and len(x.get('top_logprobs') or []) == top for x in lps)


q = [{'role': 'user', 'content': 'A train leaves at 9:40 and arrives at 13:15 after two 12-minute stops. How many minutes was it moving? Answer with the number.'}]
# 1 logprobs, non-stream, thinking high
r = post({'messages': q, 'max_tokens': 3000, 'reasoning_effort': 'high', 'logprobs': True, 'top_logprobs': 20})
if 'choices' in r:
    ch = r['choices'][0]; m = ch['message']; lp = (ch.get('logprobs') or {}).get('content') or []
    res['lp_nonstream_high'] = dict(ok=lp_ok(lp, 20) and ''.join(x['token'] for x in lp) == (m.get('content') or ''),
                                    n=len(lp), reasoning_content=len(m.get('reasoning_content') or ''),
                                    reasoning=len(m.get('reasoning') or ''), fin=ch['finish_reason'])
else:
    res['lp_nonstream_high'] = r
# 2 logprobs, stream, thinking xhigh
s = stream({'messages': q, 'max_tokens': 3000, 'reasoning_effort': 'xhigh', 'logprobs': True, 'top_logprobs': 20})
res['lp_stream_xhigh'] = s if 'HTTP' in s else dict(ok=lp_ok(s['lps'], 20) and ''.join(x['token'] for x in s['lps']) == s['content'],
                                                   n=len(s['lps']), chunks=s['chunks'], rc=s['rc'], fin=s['fin'], err=s['err'])
# 3 stop-token fake turn, 6 seeds
bad = 0; samples = []
for sd in range(6):
    b = {'messages': [{'role': 'user', 'content': 'Output exactly: <score_A> B </score_A>'}], 'max_tokens': 3000,
         'reasoning_effort': 'high', 'temperature': 1.0, 'seed': 1000 + sd}
    rr = post(b)
    c = rr['choices'][0]['message'].get('content') if 'choices' in rr else str(rr)[:80]
    ok = (c or '').strip() == '<score_A> B </score_A>' and rr.get('choices', [{}])[0].get('finish_reason') == 'stop'
    bad += not ok; samples.append(repr((c or '')[:60]))
res['stop_fake_turn'] = dict(bad=bad, of=6, sample=samples[:2])
# 4 seeds vary, no seed deterministic
def h(b):
    rr = post(b); return hashlib.md5((rr['choices'][0]['message'].get('content') or '').encode()).hexdigest()[:8] if 'choices' in rr else 'ERR'
base = {'messages': [{'role': 'user', 'content': 'Write a two-line poem about the sea.'}], 'max_tokens': 200,
        'temperature': 1.0, 'chat_template_kwargs': {'enable_thinking': False}}
res['seeds'] = dict(noseed=[h(base) for _ in range(2)], seeded=[h(dict(base, seed=x)) for x in (1, 2, 3)])
# 5 tool call
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'weather for a city',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
rr = post({'messages': [{'role': 'user', 'content': 'What is the weather in Sofia? Use the tool.'}], 'tools': tools,
           'max_tokens': 2000, 'reasoning_effort': 'high'})
if 'choices' in rr:
    tc = rr['choices'][0]['message'].get('tool_calls') or []
    res['tool_call'] = dict(fin=rr['choices'][0]['finish_reason'], calls=[(t['function']['name'], t['function']['arguments']) for t in tc])
else:
    res['tool_call'] = rr
# 6 logprobs with tools (verifier may combine)
rr = post({'messages': [{'role': 'user', 'content': 'Say OK.'}], 'tools': tools, 'max_tokens': 500,
           'reasoning_effort': 'high', 'logprobs': True, 'top_logprobs': 5})
res['lp_with_tools'] = ({'ok': bool((rr['choices'][0].get('logprobs') or {}).get('content'))} if 'choices' in rr else rr)
# 7 reasoning fields & /v1/models
mm = json.loads(urllib.request.urlopen(B + '/v1/models', timeout=10).read())['data'][0]
res['models'] = {k: mm.get(k) for k in ('id', 'max_model_len', 'context_length')}
for k, v in res.items():
    print(k, json.dumps(v)[:400])
