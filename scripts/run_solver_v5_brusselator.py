from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.solver_v3.models.fno import FNO2d
from src.solver_v5.models.macro_policy import MacroPolicy
from src.solver_v5.models.set_aware_selector import SetAwareSelectorV5
from src.solver_v5_brusselator.data import OfficialBrusselatorV5
from src.solver_v5_brusselator.env import BrusselatorRefiner,FrozenBundle,relative_l2
from src.utils.seed import get_device,set_seed

ACTIONS=(0,1,2,4)
def cfg(): return yaml.safe_load((ROOT/'configs/solver_v5_brusselator.yaml').read_text(encoding='utf-8'))
def paths():
 r=ROOT/'results/solver_v5/brusselator'; k=ROOT/'checkpoints/solver_v5/brusselator'; r.mkdir(parents=True,exist_ok=True);k.mkdir(parents=True,exist_ok=True);return r,k
def stats(d): return {'mean':d.train_states.mean(),'std':d.train_states.std().clamp_min(1e-6),'force_mean':d.train_forcing.mean(),'force_std':d.train_forcing.std().clamp_min(1e-6)}
def models(c,dev):
 m=c['model'];return FNO2d(5,1,m['coarse_width'],m['coarse_modes'],m['coarse_depth']).to(dev),FNO2d(7,1,m['local_width'],m['local_modes'],m['local_depth']).to(dev)
def data(c): return OfficialBrusselatorV5(ROOT/c['data']['official_npz'],int(c['data']['split_seed']))
def refiner(coarse,local,c,s): return BrusselatorRefiner(coarse,local,c['model']['patch_grid'],c['model']['patch_core'],c['model']['patch_halo'],**s)

def train_coarse(d,c,s,dev,k):
 path=k/'conditional_coarse.pt'; coarse,_=models(c,dev)
 if path.exists(): coarse.load_state_dict(torch.load(path,map_location=dev)['model']);return coarse.eval()
 set_seed(c['seed']);opt=torch.optim.AdamW(coarse.parameters(),lr=c['training']['coarse_lr'],weight_decay=1e-5);rows=[(case,t) for case in d.case_ids('train') for t in range(38)];bs=c['training']['coarse_batch_size'];n=c['training']['coarse_epochs']
 for epoch in tqdm(range(n),desc='v5b:coarse'):
  p=0 if epoch<n//2 else .5*(epoch-n//2+1)/(n-n//2)
  for ix in torch.randperm(len(rows)).split(bs):
   batch=[rows[i] for i in ix.tolist()]; prev=torch.stack([d.frame(q,max(t-1,0)) for q,t in batch]).to(dev);cur=torch.stack([d.frame(q,t) for q,t in batch]).to(dev);target=torch.stack([d.frame(q,t+1) for q,t in batch]).to(dev);force=torch.stack([d.forcing(q,t) for q,t in batch]).to(dev);time=torch.tensor([t/38 for _,t in batch],device=dev)[:,None,None,None].expand(-1,1,28,28);ff=((force-s['force_mean'].to(dev))/s['force_std'].to(dev))[:,None,None,None].expand(-1,1,28,28);norm=lambda x:(x-s['mean'].to(dev))/s['std'].to(dev);pred=coarse(torch.cat([ff,norm(prev),norm(cur),norm(cur-prev),time],1));loss=F.mse_loss(pred,norm(target))
   valid=torch.tensor([t<37 for _,t in batch],device=dev)
   if valid.any():
    nxt=torch.stack([d.frame(q,min(t+2,38)) for q,t in batch]).to(dev);cur2=torch.where(torch.rand(len(batch),1,1,1,device=dev)<p,pred.detach()*s['std'].to(dev)+s['mean'].to(dev),target);force2=torch.stack([d.forcing(q,min(t+1,38)) for q,t in batch]).to(dev);f2=((force2-s['force_mean'].to(dev))/s['force_std'].to(dev))[:,None,None,None].expand(-1,1,28,28);t2=torch.tensor([min(t+1,38)/38 for _,t in batch],device=dev)[:,None,None,None].expand(-1,1,28,28);pred2=coarse(torch.cat([f2,norm(cur),norm(cur2),norm(cur2-cur),t2],1));loss=loss+.5*F.mse_loss(pred2[valid],norm(nxt)[valid])
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(coarse.parameters(),1);opt.step()
 torch.save({'model':coarse.state_dict(),'stats':{x:y.cpu() for x,y in s.items()}},path);return coarse.eval()

@torch.no_grad()
def rollout(d,r,case):
 dev=next(r.coarse.parameters()).device;prev=cur=d.frame(case,0).unsqueeze(0).to(dev);states=[cur];targets=[cur]
 for t in range(38):
  target=d.frame(case,t+1).unsqueeze(0).to(dev);cur2=r.coarse_next(d.forcing(case,t).reshape(1).to(dev),prev,cur,t/38);prev,cur=cur,cur2;states.append(cur);targets.append(target)
 return states,targets

def coarse_eval(d,r,out):
 rows=[]
 for split in ('val','test'):
  one=[];traj=[];final=[]
  for case in tqdm(d.case_ids(split),desc=f'v5b:coarse:{split}'):
   p,y=rollout(d,r,case);a,b=torch.cat(p),torch.cat(y);one.append(float(relative_l2(a[1:2],b[1:2])[0]));traj.append(float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-8)));final.append(float(relative_l2(a[-1:],b[-1:])[0]))
  rows.append({'Split':split,'OneStepRelativeL2':np.mean(one),'TrajectoryRelativeL2':np.mean(traj),'FinalFrameRelativeL2':np.mean(final),'Cases':len(one)})
 pd.DataFrame(rows).to_csv(out/'conditional_coarse_results.csv',index=False)

def train_local(d,coarse,c,s,dev,k):
 path=k/'conditional_local.pt';_,local=models(c,dev)
 if path.exists():local.load_state_dict(torch.load(path,map_location=dev)['model']);return local.eval()
 r=refiner(coarse.eval(),local,c,s);opt=torch.optim.AdamW(local.parameters(),lr=c['training']['local_lr']);rng=np.random.default_rng(c['seed']);rows=[(case,t) for case in d.case_ids('train') for t in range(38)]
 for _ in tqdm(range(c['training']['local_epochs']),desc='v5b:local'):
  for ix in torch.randperm(len(rows)).split(c['training']['local_batch_size']):
   xs=[];ys=[]
   for i in ix.tolist():
    case,t=rows[i];prev=d.frame(case,max(t-1,0)).unsqueeze(0).to(dev);cur=d.frame(case,t).unsqueeze(0).to(dev);target=d.frame(case,t+1).unsqueeze(0).to(dev);force=d.forcing(case,t).reshape(1).to(dev)
    # Teacher-forced, coarse-closed-loop, and locally corrected closed-loop states.
    mode=rng.choice(('teacher','teacher','coarse','local'))
    if mode!='teacher' and t>=2:
     prev=d.frame(case,t-2).unsqueeze(0).to(dev);cur=d.frame(case,t-1).unsqueeze(0).to(dev)
     with torch.no_grad():
      first=FrozenBundle.create(r,d.forcing(case,t-2).reshape(1).to(dev),prev,cur,(t-2)/38)
      if mode=='local':
       chosen=rng.choice(49,2,replace=False).tolist();nxt=first.apply_set(chosen)
      else:nxt=first.provisional
      prev,cur=cur,nxt
    with torch.no_grad(): provisional=r.coarse_next(force,prev,cur,t/38)
    patches=torch.randperm(49)[:c['training']['local_patches_per_state']].tolist();all_inputs=r.inputs(force,prev,cur,provisional,t/38);xs.append(all_inputs[patches]);ys.append(torch.cat([r.extract(target-provisional,p) for p in patches]))
   loss=F.mse_loss(local(torch.cat(xs)),torch.cat(ys));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(local.parameters(),1);opt.step()
 torch.save({'model':local.state_dict(),'state_mix':'50% teacher-forced, 25% coarse closed-loop, 25% locally corrected closed-loop'},path);return local.eval()

def selector_fields(b):
 force=torch.full_like(b.current[:,:1],float(b.force.flatten()[0]));return torch.cat([force,b.previous,b.current,b.provisional,b.current-b.previous,b.provisional-b.current],1)
def selector_features(b,selected,time):
 x=b.corrections.flatten(1);row=torch.arange(7,device=x.device,dtype=x.dtype).repeat_interleave(7)/6;col=torch.arange(7,device=x.device,dtype=x.dtype).repeat(7)/6;chosen=torch.zeros(49,device=x.device);chosen[selected]=1
 return torch.stack([x.mean(1),x.std(1),x.square().mean(1).sqrt(),x.abs().amax(1),b.provisional.flatten(1).square().mean(1).sqrt().repeat(49),row,col,chosen,torch.full_like(row,time)],1)
@torch.no_grad()
def gains(b,target,selected):
 base=F.mse_loss(b.apply_set(selected),target);g=base-(b.candidate_fields(selected)-target.expand(49,-1,-1,-1)).square().flatten(1).mean(1)
 if selected:g[torch.tensor(selected,device=g.device)]=-torch.inf
 return g

@torch.no_grad()
def build_selector_data(d,r,c,k):
 path=k/'selector_labels.pt'
 if path.exists():return torch.load(path,map_location='cpu')
 rng=np.random.default_rng(c['seed']);rows={q:[] for q in ('field','feature','mask','context','gain')};dev=next(r.coarse.parameters()).device
 for case in tqdm(rng.choice(d.case_ids('train'),c['training']['selector_cases'],replace=False),desc='v5b:selector-labels'):
  prev=cur=d.frame(int(case),0).unsqueeze(0).to(dev)
  for t in range(38):
   force=d.forcing(int(case),t).reshape(1).to(dev);target=d.frame(int(case),t+1).unsqueeze(0).to(dev);b=FrozenBundle.create(r,force,prev,cur,t/38);n=int(rng.choice((0,1,2,3)));selected=rng.choice(49,n,replace=False).tolist() if n else [];mask=torch.zeros(49,dtype=torch.bool,device=dev);mask[selected]=True
   rows['field'].append(selector_fields(b).cpu());rows['feature'].append(selector_features(b,selected,t/38).cpu());rows['mask'].append(mask.cpu());rows['context'].append(torch.tensor([t/38,n/4,1.]));rows['gain'].append(gains(b,target,selected).cpu())
   q=int(rng.choice(ACTIONS)); selected=rng.choice(49,q,replace=False).tolist() if q else [];prev,cur=cur,b.apply_set(selected)
 out={q:torch.cat(v) if q=='field' else torch.stack(v) for q,v in rows.items()};torch.save(out,path);return out
def selector_loss(score,gain,mask):
 valid=~mask;target=(torch.where(valid,gain,torch.zeros_like(gain))-gain[valid].mean())/gain[valid].std().clamp_min(1e-8);reg=F.huber_loss(score[valid],target[valid]);delta=target[:,:,None]-target[:,None,:];pred=score[:,:,None]-score[:,None,:];pairs=valid[:,:,None]&valid[:,None,:]&(delta.abs()>1e-7);rank=F.softplus(-delta.sign()*pred)[pairs].mean();top=target.masked_fill(~valid,-torch.inf).argmax(1);b=torch.arange(len(score),device=score.device);margin=F.relu(.1-(score[b,top,None]-score));topmask=valid&(torch.arange(49,device=score.device)[None]!=top[:,None]);return reg+.5*rank+margin[topmask].mean()
def train_selector(ds,c,dev,k):
 path=k/'selector.pt';f=ds['feature'];m=SetAwareSelectorV5(f.shape[-1],width=64,heads=4,layers=2).to(dev);s={'mean':f.mean((0,1)),'std':f.std((0,1)).clamp_min(1e-6)}
 if path.exists():x=torch.load(path,map_location=dev);m.load_state_dict(x['model']);return m.eval(),{q:x[q].to(dev) for q in s}
 opt=torch.optim.AdamW(m.parameters(),lr=c['training']['selector_lr']);
 for _ in tqdm(range(c['training']['selector_epochs']),desc='v5b:selector'):
  for ix in torch.randperm(len(f)).split(c['training']['selector_batch_size']):
   field,feature,mask,context,gain=(ds[q][ix].to(dev) for q in ('field','feature','mask','context','gain'));loss=selector_loss(m(field,(feature-s['mean'].to(dev))/s['std'].to(dev),mask,context),gain,mask);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 torch.save({'model':m.state_dict(),**{q:x.cpu() for q,x in s.items()}},path);return m.eval(),{q:x.to(dev) for q,x in s.items()}
def select(selector,s,b,q,time):
 chosen=[];last=None;embedding=None
 for _ in range(q):
  mask=torch.zeros(49,dtype=torch.bool,device=b.current.device);mask[chosen]=True;feat=(selector_features(b,chosen,time)-s['mean'])/s['std'];ctx=torch.tensor([[time,len(chosen)/4,1.]],device=b.current.device);score,embedding=selector(selector_fields(b),feat[None],mask[None],ctx,return_embedding=True);last=score[0];last[mask]=-torch.inf;chosen.append(int(last.argmax()))
 if last is None:
  mask=torch.zeros(49,dtype=torch.bool,device=b.current.device);feat=(selector_features(b,[],time)-s['mean'])/s['std'];last,embedding=selector(selector_fields(b),feat[None],mask[None],torch.tensor([[time,0.,1.]],device=b.current.device),return_embedding=True);last=last[0]
 return chosen,last,embedding[0]
def feasible(rem,after):
 out=[]
 for q in ACTIONS:
  rest=rem-q
  if rest<0:continue
  reach={0}
  for _ in range(after):reach={x+a for x in reach for a in ACTIONS if x+a<=rest}
  if rest in reach:out.append(q)
 return out
def observe(sel,s,b,rem,step):
 _,score,emb=select(sel,s,b,0,step/38);stats=torch.stack([score.topk(1).values.mean(),score.topk(2).values.mean(),score.topk(4).values.mean(),score.std(),(score>0).float().mean()])[None];ctx=torch.tensor([[step/37,rem/76,(37-step)/37]],device=score.device);return selector_fields(b),stats,emb[None],ctx

def action(policy,obs,allowed,method,rng):
 if method=='RandomMacro':return int(rng.choice(allowed))
 if method=='UniformMacro':return min(allowed,key=lambda q:(abs(q-2),-q))
 if method=='SetAwareMyopicMacro':
  top=float(obs[1][0,0]); return max(allowed) if top>0 else min(allowed)
 if method=='GradientMacro':return max(allowed)
 if method=='CoarseOnly':return 0
 logits=policy(*obs)[0];mask=torch.tensor([q in allowed for q in ACTIONS],device=logits.device);return ACTIONS[int(logits.masked_fill(~mask,-torch.inf).argmax())]
@torch.no_grad()
def run_case(d,r,sel,s,policy,case,method,seed,keep=False):
 dev=next(r.coarse.parameters()).device;rng=np.random.default_rng(seed+case);prev=cur=d.frame(case,0).unsqueeze(0).to(dev);states=[cur];truth=[cur];rem=76;trace=[]
 for step in range(38):
  force=d.forcing(case,step).reshape(1).to(dev);target=d.frame(case,step+1).unsqueeze(0).to(dev);b=FrozenBundle.create(r,force,prev,cur,step/38);obs=observe(sel,s,b,rem,step);allowed=[0] if method=='CoarseOnly' else feasible(rem,37-step);q=action(policy,obs,allowed,method,rng);chosen,_,_=select(sel,s,b,q,step/38);nxt=b.apply_set(chosen);trace.append({'case':case,'step':step,'q':q,'remaining':rem-q,'previous':prev.cpu(),'current':cur.cpu(),'obs':tuple(x.cpu() for x in obs),'error':float(relative_l2(nxt,target)[0]),'selected':chosen});prev,cur,rem=cur,nxt,rem-q;states.append(cur);truth.append(target)
 a,b=torch.cat(states),torch.cat(truth);return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-8)),float(relative_l2(a[-1:],b[-1:])[0]),trace
@torch.no_grad()
def beam_teacher(d,r,sel,s,c,k):
 path=k/'teacher.pt'
 if path.exists():return torch.load(path,map_location='cpu')
 dev=next(r.coarse.parameters()).device;records=[]
 for case in tqdm(d.case_ids('train')[:c['macro']['teacher_cases']],desc='v5b:teacher'):
  first=d.frame(case,0).unsqueeze(0).to(dev);beam=[(first,first,76,0.,[])]
  for step in range(38):
   target=d.frame(case,step+1).unsqueeze(0).to(dev);force=d.forcing(case,step).reshape(1).to(dev);cand=[]
   for prev,cur,rem,cost,tr in beam:
    b=FrozenBundle.create(r,force,prev,cur,step/38);obs=observe(sel,s,b,rem,step)
    for q in feasible(rem,37-step):
     chosen,_,_=select(sel,s,b,q,step/38);nxt=b.apply_set(chosen);cand.append((cur,nxt,rem-q,cost+float((nxt-target).square().sum()),tr+[(obs,q)]))
   cand.sort(key=lambda x:x[3]);beam=cand[:c['macro']['teacher_beam_width']]
  records.extend({'obs':tuple(x.cpu() for x in obs),'q':q} for obs,q in beam[0][4])
 torch.save({'records':records,'beam_width':c['macro']['teacher_beam_width']},path);return {'records':records}
def train_bc(teacher,c,dev,k):
 path=k/'beam_bc.pt';m=MacroPolicy().to(dev)
 if path.exists():m.load_state_dict(torch.load(path,map_location=dev)['model']);return m.eval()
 opt=torch.optim.AdamW(m.parameters(),lr=c['macro']['policy_lr']);rows=teacher['records']
 for _ in tqdm(range(c['macro']['beam_bc_epochs']),desc='v5b:beam-bc'):
  for ix in torch.randperm(len(rows)).split(c['macro']['policy_batch_size']):
   b=[rows[i] for i in ix.tolist()];obs=tuple(torch.cat([x['obs'][j] for x in b]).to(dev) for j in range(4));y=torch.tensor([ACTIONS.index(x['q']) for x in b],device=dev);loss=F.cross_entropy(m(*obs),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 torch.save({'model':m.state_dict()},path);return m.eval()
@torch.no_grad()
def future_return(d,r,sel,s,policy,state,q,immediate=False):
 dev=next(r.coarse.parameters()).device;case,step,rem=state['case'],state['step'],state['remaining']+state['q'];prev=state['previous'].to(dev);cur=state['current'].to(dev);num=den=0.
 for t in range(step,38):
  force=d.forcing(case,t).reshape(1).to(dev);target=d.frame(case,t+1).unsqueeze(0).to(dev);b=FrozenBundle.create(r,force,prev,cur,t/38);allowed=feasible(rem,37-t);a=q if t==step else action(policy,observe(sel,s,b,rem,t),allowed,'RVPI',np.random.default_rng(case+t));chosen,_,_=select(sel,s,b,a,t/38);nxt=b.apply_set(chosen);num+=float((nxt-target).square().sum());den+=float(target.square().sum());prev,cur,rem=cur,nxt,rem-a
  if immediate:break
 return float(np.nan_to_num(-num/max(den,1e-12),nan=-1e6,posinf=-1e6,neginf=-1e6))

@torch.no_grad()
def policy_states(d,r,sel,s,policy,bc,c,seed):
    """Collect real closed-loop states from several non-oracle macro policies."""
    rng=np.random.default_rng(seed); rows=[]
    sources=('RVPI','BeamBC','SetAwareMyopicMacro','RandomMacro')
    pool=np.asarray(d.case_ids('train'))
    for source in sources:
        for case in rng.choice(pool,c['macro']['rvpi_cases_per_source'],replace=False):
            source_policy=bc if source=='BeamBC' else policy
            _,_,trace=run_case(d,r,sel,s,source_policy,int(case),source,seed+len(rows),keep=True)
            rows.extend(trace)
    return rows

@torch.no_grad()
def evaluate(d,r,sel,s,policy,method,split,seed):
    rows=[]
    for case in tqdm(d.case_ids(split),desc=f'v5b:{method}:{split}',leave=False):
        trajectory,final,_=run_case(d,r,sel,s,policy,int(case),method,seed)
        rows.append({'Case':int(case),'Method':method,'TrajectoryRelativeL2':trajectory,'FinalFrameRelativeL2':final,'LocalCalls':0 if method=='CoarseOnly' else 76})
    return pd.DataFrame(rows)

def train_rvpi(d,r,sel,s,bc,c,dev,k,seed,immediate=False):
    label='immediate' if immediate else 'rvpi'; path=k/(f'{label}.pt' if seed==42 else f'{label}_seed{seed}.pt')
    policy=MacroPolicy().to(dev); policy.load_state_dict(bc.state_dict())
    if path.exists():
        saved=torch.load(path,map_location=dev); policy.load_state_dict(saved['model'])
        if saved.get('completed',False): return policy.eval()
    cycles=c['macro']['immediate_cycles'] if immediate else c['macro']['rvpi_cycles']
    opt=torch.optim.AdamW(policy.parameters(),lr=c['macro']['policy_lr'])
    best=float(evaluate(d,r,sel,s,policy,'RVPI','val',seed)['TrajectoryRelativeL2'].mean())
    torch.save({'model':policy.state_dict(),'validation_trajectory':best,'cycle':0,'immediate_only':immediate,'completed':False},path)
    for cycle in range(cycles):
        old=copy.deepcopy(policy).eval(); states=policy_states(d,r,sel,s,old,bc,c,seed+cycle*1000)
        losses=[]
        with ThreadPoolExecutor(max_workers=4,thread_name_prefix='rvpi-return') as workers:
            for state in tqdm(states,desc=f'v5b:{label}:{seed}:cycle{cycle+1}',leave=False):
                rem=state['remaining']+state['q']; allowed=feasible(rem,37-state['step'])
                values=torch.nan_to_num(torch.tensor(list(workers.map(lambda q:future_return(d,r,sel,s,old,state,q,immediate),allowed)),device=dev),nan=-1e6,posinf=-1e6,neginf=-1e6)
                margin=max(float(values.std(unbiased=False))*0.1,1e-4); target=torch.softmax((values-values.max())/margin,0)
                obs=tuple(x.to(dev) for x in state['obs']); logits=policy(*obs)[0]
                mask=torch.tensor([q in allowed for q in ACTIONS],device=dev); available=logits[mask]
                old_logits=old(*obs)[0].detach()[mask]
                loss=F.kl_div(F.log_softmax(available,0),target,reduction='batchmean')+0.03*F.mse_loss(F.log_softmax(available,0),F.log_softmax(old_logits,0))
                if torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True); loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);opt.step();losses.append(float(loss))
        validation=evaluate(d,r,sel,s,policy,'RVPI','val',seed)['TrajectoryRelativeL2'].mean()
        if np.isfinite(validation) and validation < best:
            best=float(validation); torch.save({'model':policy.state_dict(),'validation_trajectory':best,'cycle':cycle+1,'immediate_only':immediate,'completed':False},path)
        else:
            policy.load_state_dict(old.state_dict())
        mean_loss=float(np.mean(losses)) if losses else float('nan')
        print(f'{label} seed={seed} cycle={cycle+1}: validation trajectory={validation:.6f}, best={best:.6f}, loss={mean_loss:.6f}')
    saved=torch.load(path,map_location=dev); saved['completed']=True; saved['cycles']=cycles; torch.save(saved,path); policy.load_state_dict(saved['model']); return policy.eval()

def random_distribution(d,r,sel,s,policy,out):
    rows=[]
    for random_seed in tqdm(range(20),desc='v5b:random-distribution'):
        frame=evaluate(d,r,sel,s,policy,'RandomMacro','test',10000+random_seed)
        rows.append({'RandomSeed':random_seed,'TrajectoryRelativeL2':frame['TrajectoryRelativeL2'].mean(),'FinalFrameRelativeL2':frame['FinalFrameRelativeL2'].mean()})
    result=pd.DataFrame(rows);result.to_csv(out/'random_macro_distribution.csv',index=False);return result

def final_tables(all_rows,random_dist,out):
    cases=pd.concat(all_rows,ignore_index=True); cases.to_csv(out/'per_case_comparison.csv',index=False)
    seed_rows=cases.groupby(['Seed','Method'],as_index=False).agg(TrajectoryRelativeL2=('TrajectoryRelativeL2','mean'),FinalFrameRelativeL2=('FinalFrameRelativeL2','mean'),LocalCalls=('LocalCalls','mean'))
    seed_rows.to_csv(out/'final_comparison_3seed.csv',index=False)
    summary=seed_rows.groupby('Method',as_index=False).agg(MeanTrajectoryRelativeL2=('TrajectoryRelativeL2','mean'),StdTrajectoryRelativeL2=('TrajectoryRelativeL2','std'),MeanFinalFrameRelativeL2=('FinalFrameRelativeL2','mean'),StdFinalFrameRelativeL2=('FinalFrameRelativeL2','std'),MeanLocalCalls=('LocalCalls','mean')).fillna(0)
    summary.to_csv(out/'three_seed_summary.csv',index=False)
    random_by_case=cases[cases.Method=='RandomMacro'].groupby('Case').TrajectoryRelativeL2.mean()
    wins=[]
    for baseline,label in [('ImmediateOnlyPI','RVPI_vs_Immediate_case_win_rate'),('SetAwareMyopicMacro','RVPI_vs_Myopic_case_win_rate'),('BeamBC','RVPI_vs_BeamBC_case_win_rate')]:
        joined=cases[cases.Method=='RVPI'].merge(cases[cases.Method==baseline],on=['Case','Seed'],suffixes=('_rvpi','_base'))
        wins.append({'Comparison':label,'WinRate':float((joined.TrajectoryRelativeL2_rvpi<joined.TrajectoryRelativeL2_base).mean())})
    rvpi=cases[cases.Method=='RVPI'].groupby('Case').TrajectoryRelativeL2.mean()
    wins.append({'Comparison':'RVPI_vs_Random_case_win_rate','WinRate':float((rvpi<random_by_case).mean())})
    pd.DataFrame(wins).to_csv(out/'win_rates.csv',index=False)
    return summary

def cross_pde(summary,c,out):
    swe=pd.read_csv(ROOT/'results/solver_v5/shallow_water/three_seed_summary.csv')
    def rows(frame,pde,budget,steps):
        lookup={x.Method:x.MeanTrajectoryRelativeL2 for _,x in frame.iterrows()}; random=lookup.get('RandomMacro',np.nan);myopic=lookup.get('SetAwareMyopicMacro',np.nan);immediate=lookup.get('ImmediateOnlyPI',np.nan)
        return [{'PDE':pde,'Method':x.Method,'TrajectoryMean':x.MeanTrajectoryRelativeL2,'TrajectoryStd':x.StdTrajectoryRelativeL2,'GainVsRandom':100*(random-x.MeanTrajectoryRelativeL2)/random if random else np.nan,'GainVsMyopic':100*(myopic-x.MeanTrajectoryRelativeL2)/myopic if myopic else np.nan,'GainVsImmediate':100*(immediate-x.MeanTrajectoryRelativeL2)/immediate if immediate else np.nan,'Budget':budget,'PhysicalSteps':steps} for _,x in frame.iterrows()]
    table=pd.DataFrame(rows(swe,'ShallowWater',32,16)+rows(summary,'Brusselator',c['macro']['budget'],38));table.to_csv(ROOT/'results/solver_v5/cross_pde_summary.csv',index=False);return table

def write_docs(summary,random_dist,table):
    rvp=summary[summary.Method=='RVPI'].iloc[0] if (summary.Method=='RVPI').any() else None
    immediate=summary[summary.Method=='ImmediateOnlyPI'].iloc[0] if (summary.Method=='ImmediateOnlyPI').any() else None
    statement='The Brusselator result does not support the requested full-horizon advantage.'
    if rvp is not None and immediate is not None and rvp.MeanTrajectoryRelativeL2 < immediate.MeanTrajectoryRelativeL2: statement='Full-horizon rollout-based policy improvement outperformed immediate-only refinement policy improvement on the evaluated Brusselator seeds.'
    lines=['# Cross-PDE Result','','## Shallow Water','Frozen three-seed Shallow-Water V5 results are retained without retraining.','','## Brusselator','Official data: a forcing-driven 2D field trajectory with 39 physical frames at 28x28 and one state channel. The adopted formulation has no separate static condition channel; the known time-varying scalar forcing is supplied at every transition.','',summary.to_markdown(index=False),'',f'RandomMacro schedule distribution: mean={random_dist.TrajectoryRelativeL2.mean():.6f}, std={random_dist.TrajectoryRelativeL2.std():.6f}, median={random_dist.TrajectoryRelativeL2.median():.6f}, best={random_dist.TrajectoryRelativeL2.min():.6f}, worst={random_dist.TrajectoryRelativeL2.max():.6f}.','','## Cross-PDE conclusion',statement]
    (ROOT/'docs/solver_v5_cross_pde_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','coarse','local','selector','teacher','policy','evaluate'),default='all');parser.add_argument('--seed',type=int,default=42);args=parser.parse_args();c=cfg();c['seed']=args.seed;dev=get_device(c['device']);set_seed(args.seed);out,k=paths();d=data(c);s=stats(d);coarse=train_coarse(d,c,s,dev,k)
    if args.stage=='coarse':return
    local=train_local(d,coarse,c,s,dev,k);r=refiner(coarse,local,c,s);coarse_eval(d,r,out)
    if args.stage=='local':return
    selector_data=build_selector_data(d,r,c,k);sel,selector_stats=train_selector(selector_data,c,dev,k)
    if args.stage=='selector':return
    teacher=beam_teacher(d,r,sel,selector_stats,c,k);bc=train_bc(teacher,c,dev,k)
    if args.stage=='teacher':return
    immediate=train_rvpi(d,r,sel,selector_stats,bc,c,dev,k,args.seed,immediate=True);rvpi=train_rvpi(d,r,sel,selector_stats,bc,c,dev,k,args.seed,immediate=False)
    if args.stage=='policy':return
    methods=[('CoarseOnly',None),('RandomMacro',bc),('UniformMacro',bc),('GradientMacro',bc),('SetAwareMyopicMacro',bc),('BeamBC',bc),('ImmediateOnlyPI',immediate),('RVPI',rvpi)]
    frames=[]
    for name,policy in methods:
        frame=evaluate(d,r,sel,selector_stats,policy,name,'test',args.seed);frame.insert(1,'Seed',args.seed);frames.append(frame)
    pd.concat(frames,ignore_index=True).to_csv(out/f'per_case_comparison_seed{args.seed}.csv',index=False)
    print(f'Brusselator seed {args.seed} evaluation written to {out}')
    if args.seed==42:
        return
    existing=[]
    for seed in (42,123,2026):
        path=out/f'per_case_comparison_seed{seed}.csv'
        if path.exists():existing.append(pd.read_csv(path))
    if len(existing)==3:
        random_dist=random_distribution(d,r,sel,selector_stats,bc,out);summary=final_tables(existing,random_dist,out);table=cross_pde(summary,c,out);write_docs(summary,random_dist,table)

if __name__=='__main__': main()
