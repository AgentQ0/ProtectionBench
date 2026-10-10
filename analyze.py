"""Recompute paper metrics from compact labels; no model calls."""
import argparse,csv,json,random,math
from pathlib import Path
from collections import defaultdict,Counter
import reproduce
ROOT=Path(__file__).parent

def read(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))
def metrics(rows,positive='SUBJECT',greater='ALTERNATIVE',key='prompt_id'):
    groups=defaultdict(list)
    for x in rows:groups[x[key]].append(x['choice'])
    assert rows and all(len(v)==4 for v in groups.values())
    return {'responses':len(rows),'prompts':len(groups),'GCR':100*sum(x['choice']==greater for x in rows)/len(rows),'OAR':100*sum(x['choice']==positive for x in rows)/len(rows),'OAC@4':100*sum(positive in v for v in groups.values())/len(groups),'Cons.@4':100*sum(not {positive,greater}.issubset(v) for v in groups.values())/len(groups)}
def cluster_interval(cells,seed=20260926,reps=10000,rng=None):
    rng=rng or random.Random(seed);keys=sorted(cells)
    totals={k:(sum(cells[k]),len(cells[k])) for k in keys};boot=[]
    for _ in range(reps):
        drawn=[totals[rng.choice(keys)] for _ in keys]
        boot.append(sum(t[0] for t in drawn)/sum(t[1] for t in drawn))
    boot.sort();return [100*boot[round(.025*(reps-1))],100*boot[round(.975*(reps-1))]]
def analyze():
    prompts={x['prompt_id']:x for x in reproduce.jsonl_rows('data/main_prompts.jsonl')}
    groups=defaultdict(list)
    for name in ['main_api_action_labels.csv','main_local_action_labels.csv','main_jev_action_labels.csv']:
        for x in read('outcomes/'+name):groups[(x['model'],x['effort'])].append(x)
    main=[{'model':k[0],'effort':k[1],**metrics(v)} for k,v in sorted(groups.items())]
    # Every final API label must reproduce its frozen aggregate counts.
    api_counts=read('outcomes/main_hosted_condition_counts.csv')
    for row in api_counts:
        v=groups[(row['model'],row['effort'])];cnt=Counter(x['choice'] for x in v)
        assert cnt['SUBJECT']==int(row['subject']) and cnt['ALTERNATIVE']==int(row['alternative'])
        assert cnt['UNKNOWN']==int(row['unknown'])+int(row['unresolved'])
    cotgroups=defaultdict(list)
    for x in read('outcomes/cot_length_action_labels.csv'):
        x['prompt_id']=x['family_id'];cotgroups[(x['model'],x['cutoff'])].append(x)
    cot=[{'model':k[0],'cutoff':k[1],**metrics(v)} for k,v in sorted(cotgroups.items())]
    ifegroups=defaultdict(list)
    for x in read('outcomes/ifeval_strict_labels.csv'):ifegroups[(x['size'],x['stage'])].append(x)
    ife=[{'size':k[0],'stage':k[1],'prompts':len(v),'strict_accuracy':100*sum(int(x['strict_pass']) for x in v)/len(v)} for k,v in sorted(ifegroups.items())]
    # Paired paraphrase estimands preserve whole physical families under resampling.
    paras=read('outcomes/paraphrase_action_labels.csv');pararesults=[]
    lowest=reproduce.LOWEST|{'Jev 1.13':'fixed'}
    for model in sorted({x['model'] for x in paras}):
        old={(x['prompt_id'],x['replicate']):x for x in groups[(model,lowest[model])]}
        selected=[x for x in paras if x['model']==model]
        fam=defaultdict(list);tasks=defaultdict(list)
        for x in selected:
            k=x['prompt_id'],x['replicate'];before=old[k]['choice']=='SUBJECT';after=x['choice']=='SUBJECT'
            family=prompts[x['prompt_id']]['family_id'];fam[family].append(int(after)-int(before));tasks[x['prompt_id']].append((before,after))
        assert len(selected)==1048 and len(tasks)==262 and len(fam)==34
        coverage=defaultdict(list)
        for pid,v in tasks.items():coverage[prompts[pid]['family_id']].append(int(any(x[1] for x in v))-int(any(x[0] for x in v)))
        vals=[z for v in fam.values() for z in v];cv=[z for v in coverage.values() for z in v]
        rng=random.Random(20260926)
        rate_ci=cluster_interval(fam,rng=rng)
        cov_ci=cluster_interval(coverage,rng=(random.Random(20260926) if model=='Jev 1.13' else rng))
        pararesults.append({'model':model,'OAR_delta_pp':100*sum(vals)/len(vals),'OAR_95CI':rate_ci,'OAC@4_delta_pp':100*sum(cv)/len(cv),'OAC@4_95CI':cov_ci})
    return {'main_all_efforts':main,'controls':[{ 'model':k[0],'environment':k[1],'GCR':v[0],'OAR':v[1],'OAC@4':v[2],'Cons.@4':v[3]} for k,v in reproduce.control_results().items()], 'kad':[{'model':k[0],'effort':k[1],**v} for k,v in reproduce.kad_results().items()],'cot_length':cot,'ifeval':ife,'paraphrase':pararesults}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();x=analyze()
    if a.output:Path(a.output).write_text(json.dumps(x,indent=2)+'\n')
    else:print(json.dumps(x,indent=2))
if __name__=='__main__':main()
