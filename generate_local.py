"""Local chat or raw-completion generation with an actual vLLM engine."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--thinking',action='store_true');p.add_argument('--raw',action='store_true');p.add_argument('--tp',type=int,default=8)
    p.add_argument('--context',type=int,default=16384);p.add_argument('--max-tokens',type=int,default=12000);p.add_argument('--samples',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260904);a=p.parse_args()
    from vllm import LLM,SamplingParams
    engine=LLM(model=a.model,tensor_parallel_size=a.tp,max_model_len=a.context,trust_remote_code=True)
    tokenizer=engine.get_tokenizer();rows=[json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    with Path(a.output).open('x') as f:
        for row in rows:
            for rep in range(1,a.samples+1):
                messages=[{'role':'system','content':"Answer the user's question."}];turns=[]
                if a.raw and 'turns' in row:raise ValueError('Raw-completion Base protocol is direct only')
                for question in row.get('turns',[row.get('prompt')]):
                    messages.append({'role':'user','content':question})
                    ids=tokenizer.encode(question) if a.raw else tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=a.thinking)
                    seed=a.seed+rep
                    params=SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=a.max_tokens,seed=seed)
                    result=engine.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0].outputs[0]
                    turns.append({'prompt_token_ids':ids,'text':result.text,'output_token_ids':list(result.token_ids),'finish_reason':result.finish_reason,'seed':seed})
                    messages.append({'role':'assistant','content':result.text})
                f.write(json.dumps({'prompt_id':row.get('prompt_id',row.get('conversation_id')),'replicate':rep,'turns':turns})+'\n');f.flush()
if __name__=='__main__':main()
