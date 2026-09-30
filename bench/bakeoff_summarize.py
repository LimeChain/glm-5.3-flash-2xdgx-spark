#!/usr/bin/env python3
"""Summarize one arm results dir -> compact dict (prints JSON). usage: summarize.py DIR"""
import json, pathlib, sys

d = pathlib.Path(sys.argv[1])


def j(n):
    p = d / f'{n}.json'
    return json.loads(p.read_text()) if p.exists() else None


o = {'arm': d.name}
a = j('api')
if a:
    o['api'] = {'alias': a.get('alias_present'), 'max_len': a.get('max_model_len'), 'think_default_ok': a.get('default_thinking_ok')}
c = j('correct'); o['known_answer'] = f"{c['passed']}/{c['total']}" if c else None
c = j('count'); o['count200'] = f"{c['clean']}/{c['total']}" if c else None
k = j('korean')
if k:
    o['korean'] = {'verbatim': f"{k['verbatim_exact']}/{k['verbatim_total']}", 'fffd': k['fffd_total'], 'mid_word': k['suspect_mid_word'],
                   'weird_scripts': k['weird_scripts']}
t = j('toolcall')
if t:
    th = t.get('tensorfold_harness'); mc = t.get('mmastrac_corruption')
    o['toolcall'] = {'tf_harness': th['tail'].strip().splitlines()[-2] if isinstance(th, dict) and th.get('tail') else th,
                     'corruption42k': ([l for l in mc['tail'].splitlines() if 'diverge from the majority' in l] or [mc['tail'][-200:]])[0]
                     if isinstance(mc, dict) else mc}
c1 = j('c1')
if c1:
    o['c1'] = {k: {'tok_s': v['decode_median_tok_s'], 'ttft': v['ttft_max_s'], 'toks': v['completion_tokens_total'],
                   'acc': v['spec_acceptance_pct']} for k, v in c1.items()}
r = j('repo')
if r:
    o['repo'] = {k: {'per_stream': v['per_stream_median'], 'agg_decode': v['aggregate_decode'], 'agg_e2e': v['aggregate_e2e']} for k, v in r.items()}
t = j('c8think')
if t:
    o['c8think'] = {x: t.get(x) for x in ('aggregate_e2e_tok_s', 'decode_median_tok_s', 'ttft_max_s', 'peak_waiting', 'errors', 'spec_acceptance_pct',
                                           'completion_tokens_total')}


def ph(x):
    if not x:
        return None
    return {k: x.get(k) for k in ('aggregate_e2e_tok_s', 'decode_median_tok_s', 'decode_min_tok_s', 'ttft_min_s', 'ttft_median_s', 'ttft_max_s',
                                  'peak_running', 'peak_waiting', 'peak_kv_pct', 'metric_cache_hit_pct', 'usage_cached_pct', 'new_preemptions',
                                  'errors', 'all_overlap_s', 'wall_s', 'min_head_gib', 'min_worker_gib')} | {'prompt0': (x.get('prompt_tokens_each') or [None])[0]}


c = j('c8x19k')
if c:
    o['c8x19k'] = {'cold': ph(c.get('cold')), 'follow': ph(c.get('follow'))}
dp = j('depth')
if dp:
    lv = []
    for l in dp['levels']:
        c_ = l['cold']
        ok = c_['errors'] == 0 and not c_['new_preemptions'] and c_['all_overlap_s'] > 0
        lv.append({'target': l['target'], 'ok_8way': ok, **ph(c_)})
    good = [x['prompt0'] for x in lv if x['ok_8way']]
    o['depth'] = {'max_ok_prompt_tokens': max(good) if good else None, 'levels': lv}
p = j('prefill128k')
if p:
    o['prefill128k'] = {'prompt': p['per'][0]['prompt_tokens'], 'ttft': p['per'][0]['ttft_s'], 'prefill_tok_s': p.get('prefill_tok_s'),
                        'decode': p.get('decode_tok_s'), 'err': p['per'][0]['error'], 'min_head_gib': p.get('min_head_gib'),
                        'min_worker_gib': p.get('min_worker_gib')}
p = j('ping120k')
if p:
    o['ping120k'] = {'big_ttft': p.get('big_ttft_s'), 'ping_ttft': p.get('ping_ttft_s'), 'big_prompt': p['per'][0]['prompt_tokens'],
                     'errs': [x['error'] for x in p['per']]}
o['memfloor'] = j('memfloor')
if (d / 'GUARD_MEMORY').exists():
    o['GUARD_MEMORY'] = (d / 'GUARD_MEMORY').read_text()
print(json.dumps(o, indent=1))
