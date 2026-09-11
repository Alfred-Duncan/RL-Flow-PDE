from __future__ import annotations

import argparse
import json
import sys
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
    with torch.no_grad(): provisional=r.coarse_next(force,prev,cur,t/38)
    patches=torch.randperm(49)[:c['training']['local_patches_per_state']].tolist();all_inputs=r.inputs(force,prev,cur,provisional,t/38);xs.append(all_inputs[patches]);ys.append(torch.cat([r.extract(target-provisional,p) for p in patches]))
   loss=F.mse_loss(local(torch.cat(xs)),torch.cat(ys));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(local.parameters(),1);opt.step()
 torch.save({'model':local.state_dict(),'state_mix':'teacher-forced; deployment closed-loop states supplied to selector and macro policy'},path);return local.eval()

def selector_fields(b):
 force=torch.full_like(b.current[:,:1],float(b.force.flatten()[0]));return torch.cat([force,b.previous,b.current,b.provisional,b.current-b.previous,b.provisional-b.current],1)
def selector_features(b,selected,time):
 x=b.corrections.flatten(1);row=torch.arange(7,device=x.device,dtype=x.dtype).repeat_interleave(7)/6;col=torch.arange(7,device=x.device,dtype=x.dtype).repeat(7)/6;chosen=torch.zeros(49,device=x.device);chosen[selected]=1
 return torch.stack([x.mean(1),x.std(1),x.square().mean(1).sqrt(),x.abs().amax(1),b.provisional.flatten(1).square().mean(1).sqrt().repeat(49),row,col,chosen,torch.full_like(row,time)],1)
@torch.no_grad()
def gains(b,target,selected):
 base=F.mse_loss(b.apply_set(selected),target);g=base-(b.candidate_fields(selected)-target.expand(49,-1,-1,-1)).square().flatten(1).mean(1);g[torch.tensor(selected,device=g.device)]=-torch.inf if selected else g.new_tensor(0.0);return g

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
