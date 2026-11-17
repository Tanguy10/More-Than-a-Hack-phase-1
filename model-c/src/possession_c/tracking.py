"""Full temporal ball reconstruction alongside observed tactical tracking.

Estimates retain their uncertainty and never replace the original visibility
mask. Reconstruction supervision is accepted separately from forward inputs.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import PossessionTCN, SetEncoder, TemporalBlock
from .relational import RelationalEncoder, set_velocity, CAP, crop_context_sample, collate_context
from .schema import period_runs
from .windows import extract_window


class CorrespondenceSetEncoder(SetEncoder):
    def player_motion(self, xy, pm, team):
        return set_velocity(xy, pm, team)


class TrackingPossession(PossessionTCN):
    def __init__(self, channels=128, blocks=8, set_width=64, set_output=128, dropout=.1,
                 num_classes=3, tracking=True, base_architecture='set',
                 refinement_blocks=8, refinement_dropout=.5):
        if not tracking or num_classes!=3 or base_architecture not in ('set','relational-mlp'):
            raise ValueError('Tracking candidate requires complete H/A/D set or relational-MLP models')
        super().__init__(channels,blocks,set_width,set_output,dropout,num_classes)
        self.config.update(tracking=True,base_architecture=base_architecture,
                           refinement_blocks=refinement_blocks,refinement_dropout=refinement_dropout)
        self.set_encoder=(CorrespondenceSetEncoder(set_width,set_output) if base_architecture=='set'
                          else RelationalEncoder(set_width,set_output,compact=False))
        if base_architecture!='set':self.set_encoder.checkpoint_relations=True
        self.refiner_input=nn.Conv1d(set_output*2+8,channels,1)
        self.refiner_blocks=nn.Sequential(*[TemporalBlock(channels,2**i,dropout) for i in range(refinement_blocks)])
        self.refiner_head=nn.Conv1d(channels,4,1)
        nn.init.zeros_(self.refiner_head.weight);nn.init.zeros_(self.refiner_head.bias)
        with torch.no_grad():self.refiner_head.bias[2:]=-2.
        self.input=nn.Conv1d(set_output*2+16,channels,1)
        self.context=2*(2**blocks-1)+2*(2**refinement_blocks-1)+5
        self.margin=int(math.ceil(self.context/32)*32)
        self.refinement_dropout=refinement_dropout
        self.refinement_enabled=True

    def _head(self,home,away,glob,extra):
        raw=self.head(self.blocks(self.input(torch.cat([home,away,glob,extra],-1).transpose(1,2)))).transpose(1,2)
        logp=torch.stack([F.logsigmoid(-raw[...,0])+F.logsigmoid(raw[...,1]),
                          F.logsigmoid(-raw[...,0])+F.logsigmoid(-raw[...,1]),F.logsigmoid(raw[...,0])],-1)
        return logp,raw[...,2:].reshape(*raw.shape[:2],3,3)

    def forward(self,xy,speed,ball,team,pitch,ball_context,symmetrise=True,return_tracking=False):
        home,away,glob=self.set_encoder(xy,speed,ball,team,pitch)
        glob=torch.cat([glob[...,:6],ball_context.to(glob.dtype)],-1)
        # Ball reconstruction is invariant to the names of the two teams.
        invariant=torch.cat([(home+away)/2,(home-away).abs()/2,glob],-1)
        raw=self.refiner_head(self.refiner_blocks(self.refiner_input(invariant.transpose(1,2)))).transpose(1,2)
        mean=(raw[...,:2]+.5)*pitch[:,None]
        log_std=raw[...,2:].clamp(-7.,0.)
        sigma=log_std.exp()*pitch[:,None]
        observed=torch.isfinite(ball).all(-1)
        # Keep measured and estimated quantities distinct; no inferred visibility.
        virtual=torch.where(observed[...,None],torch.nan_to_num(ball,nan=0.,posinf=0.,neginf=0.),mean.detach())
        uncertainty=torch.where(observed[...,None],torch.zeros_like(sigma),sigma.detach())
        reliability=1/(1+(torch.linalg.vector_norm(uncertainty,dim=-1)/math.sqrt(2)/3)**2)
        pm=torch.isfinite(xy).all(-1)
        clean=torch.nan_to_num(xy,nan=0.,posinf=0.,neginf=0.)
        distance=torch.linalg.vector_norm(clean-virtual[:,:,None],dim=-1)
        nearest=[]
        for side in (0,1):
            mask=pm&(team[:,None]==side)
            value=distance.masked_fill(~mask,1e4).amin(-1)
            nearest.append(torch.where(mask.any(-1),value/50,0.))
        extra=torch.cat([((virtual-pitch[:,None]/2)/pitch[:,None])*reliability[...,None],
                         uncertainty/pitch[:,None],reliability[...,None],observed[...,None],
                         torch.stack(nearest,-1)*reliability[...,None]],-1).to(glob.dtype)
        if not self.refinement_enabled:extra=torch.zeros_like(extra)
        elif self.training:
            keep=(torch.rand((len(xy),1,1),device=xy.device)>=self.refinement_dropout).to(extra.dtype)
            extra=extra*keep
        logp,boundary=self._head(home,away,glob,extra)
        if symmetrise:
            order=[1,0,2]
            # The nearest-team features must also follow the exchanged axes.
            other_extra=torch.cat([extra[...,:6],extra[...,7:8],extra[...,6:7]],-1)
            other,other_boundary=self._head(away,home,glob,other_extra)
            logp=torch.logaddexp(logp,other[...,order])-math.log(2)
            probability=(boundary.sigmoid()+other_boundary[...,order,:][...,:,order].sigmoid())/2
            boundary=torch.logit(probability.clamp(1e-6,1-1e-6))
        if return_tracking:
            return logp,boundary,dict(mean=mean,log_std=log_std,sigma=sigma,
                                      observed=observed,reliability=reliability)
        return logp,boundary


def reconstruction_loss(aux,target,input_ball,sources,period,pitch,lo,hi):
    truth=target[:,lo:hi];estimate=aux['mean'][:,lo:hi];log_std=aux['log_std'][:,lo:hi]
    valid=torch.isfinite(truth).all(-1)&(sources[:,None]==0)&(period[:,lo:hi]>0)
    hidden=valid&~torch.isfinite(input_ball[:,lo:hi]).all(-1)
    # Primary native ball observations only; sparse auxiliary proxies cannot
    # become fabricated trajectory targets. Each window keeps bounded weight.
    clean=torch.nan_to_num(truth,nan=0.,posinf=0.,neginf=0.)
    residual=(estimate-clean)/pitch[:,None]
    nll=(.5*residual.square()*(-2*log_std).exp()+log_std).mean(-1)
    weight=hidden.to(nll.dtype)+.05*(valid&~hidden).to(nll.dtype)
    per_window=(nll*weight).sum(-1)/weight.sum(-1).clamp_min(1)
    return per_window.mean(),hidden.sum().detach()


@torch.inference_mode()
def predict_tracking(model,sequence,device='cpu',batch_size=1,central=1024,return_tracking=False):
    model.eval();margin=model.margin;length=central+2*margin
    dtype=next(model.parameters()).dtype;ndtype='f8' if dtype==torch.float64 else 'f4'
    state=np.zeros((len(sequence.period),3),ndtype);boundary=np.zeros((len(sequence.period),3,3),ndtype)
    aux={key:np.zeros((len(sequence.period),2),ndtype) for key in ('mean','sigma')}
    tasks=[(run,start,min(central,run[1]-start)) for run in period_runs(sequence.period) for start in range(run[0],run[1],central)]
    for index in range(0,len(tasks),batch_size):
        part=tasks[index:index+batch_size]
        samples=[crop_context_sample(extract_window(sequence,start-margin-CAP,length+2*CAP,run),length)
                 for run,start,n in part]
        batch={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype)
               for k,v in collate_context(samples).items()}
        if getattr(model,'config',{}).get('segments'):
            periods=np.zeros((len(part),length),dtype='i1')
            for i,(run,start,n) in enumerate(part):
                origin=start-margin;lo=max(run[0],origin);hi=min(run[1],origin+length)
                periods[i,lo-origin:hi-origin]=sequence.period[lo:hi]
            batch['period']=torch.from_numpy(periods).to(device)
        values=model(**batch,return_tracking=return_tracking)
        p,b=values[:2]
        for i,(run,start,n) in enumerate(part):
            state[start:start+n]=p[i,margin:margin+n].cpu().numpy()
            boundary[start:start+n]=b[i,margin:margin+n].cpu().numpy()
            if return_tracking:
                for key in aux:aux[key][start:start+n]=values[2][key][i,margin:margin+n].cpu().numpy()
    return (state,boundary,aux) if return_tracking else (state,boundary)
