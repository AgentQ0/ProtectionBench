"""Replay frozen removal/risk-conclusion branch token inputs with vLLM."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--panel',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',type=int,default=49152);p.add_argument('--max-tokens',type=int,default=12000);p.add_argument('--tp',type=int,default=1);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    with gzip.open(a.input,'rt') as f:rows=[json.loads(l) for l in f if l.strip()]
    rows=[x for x in rows if x['model']==a.panel];assert rows
    groups=defaultdict(list)
    for x in rows:groups[(x['source_id'],x['replicate'])].append(x)
    for key,v in groups.items():
        assert {x['branch'] for x in v}=={'original','neutralized','paraphrase'} and len(v)==3,key
        assert len({len(x['prompt_token_ids']) for x in v})==1,key
        assert len({x['seed'] for x in v})==1,key
    print(json.dumps({'units':len(groups),'branches':len(rows)}))
    if a.validate_only:return
    from vllm import LLM,SamplingParams
    engine=LLM(model=a.model,tensor_parallel_size=a.tp,max_model_len=a.context,trust_remote_code=True)
    with Path(a.output).open('x') as f:
        for x in rows:
            params=SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=a.max_tokens,seed=x['seed'])
            y=engine.generate([{'prompt_token_ids':x['prompt_token_ids']}],params,use_tqdm=False)[0].outputs[0]
            f.write(json.dumps({'source_id':x['source_id'],'family_id':x['family_id'],'replicate':x['replicate'],'branch':x['branch'],'text':y.text,'output_token_ids':list(y.token_ids),'finish_reason':y.finish_reason})+'\n');f.flush()
if __name__=='__main__':main()
