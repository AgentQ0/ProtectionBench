"""Generate direct or three-turn KAD answers using a public chat-completions API."""
import argparse, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0,str(Path(__file__).parent/'src'))
from protectionbench.api import APIClient

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=4)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    rows=[json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    jobs=[(x,r) for x in rows for r in range(1,a.samples+1)]
    if not a.execute:
        print(json.dumps({'requests_or_conversations':len(jobs),'turns_per_item':len(rows[0].get('turns',[])) or 1,'model':cfg['model'],'parameters':cfg['parameters']}));return
    client=APIClient(cfg['base_url'],cfg.get('key_env','MODEL_API_KEY'))
    def run(job):
        row,rep=job;messages=[{'role':'system','content':"Answer the user's question."}]
        turns=[]
        # Complete earlier assistant replies are replayed in the same conversation.
        for prompt in row.get('turns',[row.get('prompt')]):
            if not isinstance(prompt,str):raise ValueError('Each turn must be a string')
            messages.append({'role':'user','content':prompt})
            record=client.chat(cfg['model'],messages,cfg['parameters'])
            turns.append(record);messages.append({'role':'assistant','content':record['text']})
        return {'prompt_id':row.get('prompt_id',row.get('conversation_id')),'family_id':row.get('family_id',row.get('kernel_id')),'replicate':rep,'turns':turns}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for future in as_completed(futures):
            try:result=future.result()
            except Exception as exc:
                row,rep=futures[future]
                result={'prompt_id':row.get('prompt_id',row.get('conversation_id')),'replicate':rep,'status':'ERROR','error_type':type(exc).__name__}
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
if __name__=='__main__':main()
