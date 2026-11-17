"""Full-context possession queries alongside the complete tracking backbone."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .tracking import TrackingPossession


def rasterize(logits,intervals,length,endpoints=None):
    """Soft H/A occupancy and localized end-to-start evidence, without labels."""
    time=(torch.arange(length,device=logits.device,dtype=logits.dtype)+.5)/length
    start,end=intervals.unbind(-1);tau=1/length
    left=(time-start[...,None])/tau;right=(end[...,None]-time)/tau
    membership=left.sigmoid()*right.sigmoid()
    probability=logits.softmax(-1)[...,:2]
    mass=torch.einsum('bqt,bqc->btc',membership,probability)
    # A unit neutral prior prevents low-confidence/null queries from producing
    # a confident team ratio simply by cancelling their small occupancy mass.
    conditional=(mass+.5)/(mass.sum(-1,keepdim=True)+1)
    kinds=logits.new_ones((*logits.shape[:2],2)) if endpoints is None else endpoints.softmax(-1)[...,0]
    # State samples are frame centres; an interval [a,b) ends at edge b.
    # b==length lies outside this crop and is never clamped onto length-1.
    edges_time=torch.arange(length,device=logits.device,dtype=logits.dtype)/length
    start_distance=(edges_time-start[...,None])/tau;end_distance=(edges_time-end[...,None])/tau
    starts=torch.einsum('bqt,bqc->btc',(-.5*start_distance.square()).exp()*kinds[...,0,None],probability)
    ends=torch.einsum('bqt,bqc->btc',(-.5*end_distance.square()).exp()*kinds[...,1,None]*(end<1)[...,None],probability)
    edges=torch.stack([torch.log1p(100*ends[...,0]*starts[...,1]),
                       torch.log1p(100*ends[...,1]*starts[...,0])],-1)
    return conditional,edges,mass


class SegmentQueries(nn.Module):
    def __init__(self,channels=128,queries=128,layers=6,heads=8,feedforward=512,dropout=.1):
        super().__init__()
        if channels%2 or channels%heads or min(queries,layers)<=0:raise ValueError('Invalid segment attention dimensions')
        self.queries=nn.Parameter(torch.randn(queries,channels)*.02)
        layer=nn.TransformerDecoderLayer(channels,heads,feedforward,dropout,activation='gelu',
                                         batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,layers,norm=nn.LayerNorm(channels))
        self.classes=nn.Linear(channels,3)
        self.endpoints=nn.Linear(channels,6)
        self.geometry=nn.Sequential(nn.Linear(channels,channels),nn.GELU(),nn.Linear(channels,2))
        reference=(torch.arange(queries,dtype=torch.float32)+.5)/queries
        self.register_buffer('reference_logits',torch.logit(reference))
        with torch.no_grad():self.classes.bias[2]=2

    def forward(self,memory,central=1024,padding_mask=None):
        if memory.ndim!=3 or memory.shape[1]<central:raise ValueError('Complete central memory required')
        length=memory.shape[1];channels=memory.shape[-1]
        position=torch.arange(length,device=memory.device,dtype=memory.dtype)[:,None]
        frequencies=torch.exp(torch.arange(0,channels,2,device=memory.device,dtype=memory.dtype)*(-math.log(10000)/channels))
        angles=position*frequencies
        positional=torch.stack((angles.sin(),angles.cos()),-1).reshape(length,channels)
        encoded=memory+positional[None]
        if padding_mask is not None:
            if padding_mask.shape!=memory.shape[:2] or padding_mask.dtype!=torch.bool:raise ValueError('Invalid observed-period memory mask')
            encoded=encoded.masked_fill(padding_mask[...,None],0)
            # Safe null memory for a fully padded software fixture; no NaN attention.
            # Keep a separate key mask: masked_fill retains its original mask
            # for backward, so even an empty in-place assignment would break it.
            padding_mask=padding_mask.clone()
            padding_mask[padding_mask.all(1),0]=False
        output=self.decoder(self.queries[None].expand(len(memory),-1,-1),encoded,memory_key_padding_mask=padding_mask)
        raw=self.geometry(output)
        centre=(raw[...,0]+self.reference_logits).sigmoid()
        width=1/central+(1-1/central)*(raw[...,1]-3.5).sigmoid()
        start=(centre-width/2).clamp(0,1-1/central)
        end=torch.maximum((centre+width/2).clamp(1/central,1),start+1/central)
        intervals=torch.stack([start,end],-1)
        logits=self.classes(output)
        endpoints=self.endpoints(output).reshape(*output.shape[:2],2,3)
        conditional,edges,mass=rasterize(logits,intervals,central,endpoints)
        return dict(logits=logits,intervals=intervals,endpoints=endpoints,conditional=conditional,
                    occupancy_mass=mass,edge_evidence=edges)


class SegmentPossession(TrackingPossession):
    """One complete recipe: 128 queries, six attention layers, native central1024."""
    def __init__(self,segments=True,segment_queries=128,segment_layers=6,segment_heads=8,
                 segment_feedforward=512,segment_dropout=.1,**options):
        if (not segments or segment_queries!=128 or segment_layers!=6 or segment_heads!=8 or
                segment_feedforward!=512 or options.get('channels',128)!=128 or
                options.get('blocks',8)!=8 or options.get('refinement_blocks',8)!=8):
            raise ValueError('Only the registered full segment architecture is a quality model')
        super().__init__(**options)
        self.config.update(segments=True,segment_queries=segment_queries,segment_layers=segment_layers,
            segment_heads=segment_heads,segment_feedforward=segment_feedforward,segment_dropout=segment_dropout)
        self.segment_decoder=SegmentQueries(128,segment_queries,segment_layers,segment_heads,segment_feedforward,segment_dropout)
        self.segment_state_gain=nn.Parameter(torch.zeros(()))
        self.segment_boundary_gain=nn.Parameter(torch.zeros(()))
        self.segments_enabled=True
        self._segment_outputs=[]
        self._segment_period=None

    def _head(self,home,away,glob,extra):
        features=self.blocks(self.input(torch.cat([home,away,glob,extra],-1).transpose(1,2)))
        raw=self.head(features).transpose(1,2)
        if self.segments_enabled:
            if raw.shape[1]<1024:raise ValueError('Segment model requires the native central1024 window')
            period=self._segment_period
            query=self.segment_decoder(features.transpose(1,2),1024,None if period is None else period<=0)
            lo=(raw.shape[1]-1024)//2;hi=lo+1024
            logratio=(query['conditional'][...,0].log()-query['conditional'][...,1].log()).clamp(-8,8)
            delta=F.pad(logratio,(lo,raw.shape[1]-hi))*self.segment_state_gain.tanh()
            raw=torch.cat([raw[...,:1],raw[...,1:2]+delta[...,None],raw[...,2:]],-1)
            boundary=raw[...,2:].reshape(*raw.shape[:2],3,3)
            correction=torch.zeros_like(boundary)
            correction[:,lo:hi,0,1]=query['edge_evidence'][...,0]
            correction[:,lo:hi,1,0]=query['edge_evidence'][...,1]
            if period is not None:
                valid=torch.zeros_like(period,dtype=torch.bool)
                valid[:,1:]=(period[:,1:]>0)&(period[:,1:]==period[:,:-1])
                correction=correction*valid[...,None,None]
            boundary=boundary+self.segment_boundary_gain.tanh()*correction
            self._segment_outputs.append(query)
        else:boundary=raw[...,2:].reshape(*raw.shape[:2],3,3)
        logp=torch.stack([F.logsigmoid(-raw[...,0])+F.logsigmoid(raw[...,1]),
                          F.logsigmoid(-raw[...,0])+F.logsigmoid(-raw[...,1]),F.logsigmoid(raw[...,0])],-1)
        return logp,boundary

    def forward(self,*args,period=None,return_tracking=False,**kwargs):
        if period is None:raise ValueError('Observed native periods are required for segment memory masking')
        self._segment_outputs=[]
        self._segment_period=period
        values=super().forward(*args,return_tracking=return_tracking,**kwargs)
        if return_tracking:values[2]['segments']=tuple(self._segment_outputs)
        self._segment_outputs=[]
        self._segment_period=None
        return values


