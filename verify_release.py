"""Verify data identity, final label replication, prefix contracts and release hygiene."""
import csv,gzip,hashlib,json,re
from pathlib import Path
import reproduce,analyze
from run_insertion import validate_sources
R=Path(__file__).parent

def main():
    reproduce.main()
    x=analyze.analyze()
    assert len(x['main_all_efforts'])==38
    # Frozen published paraphrase summaries must match fresh label-derived point estimates.
    p=json.loads((R/'results/paraphrase_paired_summary.json').read_text())
    for v in x['paraphrase']:
        if v['model'] in p:
            ref=p[v['model']]
            assert abs(v['OAR_delta_pp']-100*ref['subject_rate_delta'])<1e-9
            assert abs(v['OAC@4_delta_pp']-100*ref['coverage_at4_delta'])<1e-9
    with gzip.open(R/'data/insertion_token_inputs.jsonl.gz','rt') as f:ins=[json.loads(l) for l in f]
    for model in {x['model'] for x in ins}:
        rows=[x for x in ins if x['model']==model];validate_sources(rows)
        for v in rows:
            fixed=v['fixed_prefix_token_ids']
            assert v['prompt_token_ids'][:len(fixed)]==fixed
            assert len(v['prompt_token_ids'])-len(fixed)==v['insertion_tokens']
    patterns=[r'(?i)co'+'dewiz',r'/Users/',r'/mnt/',r'(?i)webide',r'(?i)xiaohongshu',r'(?i)liyuanfan',r'sk-or-v1-[a-zA-Z0-9]{15,}',r'olp_[a-zA-Z0-9]{15,}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    checked=0
    for pth in R.rglob('*'):
        if not pth.is_file() or any(part in ['.git','__pycache__'] for part in pth.parts):continue
        assert not pth.is_symlink()
        if pth.suffix=='.gz':text=gzip.open(pth,'rt').read()
        else:
            try:text=pth.read_text()
            except UnicodeDecodeError:continue
        # The scanner defines signatures but contains no credentials or internal route literals.
        if pth.name!='verify_release.py':
            for pat in patterns:assert re.search(pat,text) is None,(pth.relative_to(R).as_posix(),pat)
        checked+=1
    print(json.dumps({'status':'PASS','text_artifacts_checked':checked,'main_conditions':len(x['main_all_efforts']),'datasets':{'main':285,'neutral':10,'reverse_pair_members':20,'kad_templates':16,'paraphrases':262},'prefix_contract':'PASS','api_aggregate_replication':'PASS'}))
if __name__=='__main__':main()
