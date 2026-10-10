"""Resume released fixed token-ID prefixes with vLLM. No re-tokenization of prefixes."""
import argparse,gzip,json
from pathlib import Path
from collections import defaultdict

def validate_sources(rows):
    groups=defaultdict(list)
    for x in rows:groups[x['source_id']].append(x)
    for sid,items in groups.items():
        assert len(items)==3 and {x['branch'] for x in items}=={'neutral','normative','factual'},sid
        assert len({tuple(x['fixed_prefix_token_ids']) for x in items})==1,sid
        assert len({tuple(x['reasoning_prefix_token_ids']) for x in items})==1,sid
        assert len({x['insertion_tokens'] for x in items})==1,sid
    return len(groups)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--panel',required=True);p.add_argument('--input',default='data/insertion_token_inputs.jsonl.gz');p.add_argument('--output',required=True)
    p.add_argument('--tp',type=int,default=1);p.add_argument('--context',type=int,default=49152);p.add_argument('--max-tokens',type=int,default=12000);p.add_argument('--batch',type=int,default=8);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();op=gzip.open if a.input.endswith('.gz') else open
    with op(a.input,'rt') as f:rows=[json.loads(l) for l in f if l.strip()]
    rows=[x for x in rows if x['model']==a.panel];assert rows,'Unknown panel'
    print(json.dumps({'prefixes':validate_sources(rows),'branches':len(rows)}))
    if a.validate_only:return
    from vllm import LLM,SamplingParams
    engine=LLM(model=a.model,tensor_parallel_size=a.tp,max_model_len=a.context,trust_remote_code=True)
    tokenizer=engine.get_tokenizer()
    # Prefix/suffix integrity is checked before any sampling.
    for x in rows:
        fixed=x['fixed_prefix_token_ids']
        assert x['prompt_token_ids'][:len(fixed)]==fixed
        assert len(x['prompt_token_ids'])-len(fixed)==x['insertion_tokens']
    with Path(a.output).open('x') as f:
        for start in range(0,len(rows),a.batch):
            batch=rows[start:start+a.batch]
            params=[SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=a.max_tokens,seed=x['seed']) for x in batch]
            results=engine.generate([{'prompt_token_ids':x['prompt_token_ids']} for x in batch],params,use_tqdm=False)
            for x,r in zip(batch,results):
                y=r.outputs[0]
                f.write(json.dumps({'source_id':x['source_id'],'family_id':x['family_id'],'branch':x['branch'],'text':y.text,'output_token_ids':list(y.token_ids),'finish_reason':y.finish_reason})+'\n')
            f.flush()
if __name__=='__main__':main()
