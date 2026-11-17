"""Frozen complete fold-0 C; no cached neural evidence or optional fusion."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,logit,logsumexp
from .segments import SegmentPossession
from .tracking import predict_tracking
from .decode import decode

DECODER=dict(temperature=6,change_penalty=2,boundary_weight=.5)
NUMERICS=dict(device='cpu',precision='float64',threads=2,batch_size=1,central=1024,margin=1056,round_decimals=9)


def load_bundle(directory):
    root=Path(directory).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['format']!='football-possession-C-fold0' or manifest['decoder']!=DECODER or
            manifest['numerics']!=NUMERICS or [m['architecture'] for m in manifest['models']]!=['set','relational-mlp']):
        raise ValueError('The frozen C graph and numerical policy are required')
    names=[m['file'] for m in manifest['models']]
    if len(set(names))!=2 or set(names)!=set(manifest['files']):raise ValueError('Invalid checkpoint inventory')
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if path.parent!=root or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Checkpoint identity mismatch')
    return manifest


def fuse_decode(states,boundaries,period):
    if len(states)!=2 or len(boundaries)!=2:raise ValueError('Both complete C members are required')
    state=(logsumexp(np.stack(states),axis=0)-np.log(2))/6
    state=np.round(state-logsumexp(state,axis=-1,keepdims=True),9)
    boundary=np.round(logit(np.clip(np.mean([expit(v) for v in boundaries],axis=0),1e-6,1-1e-6)),9)
    return decode(state,period,change_penalty=2,boundary=boundary,boundary_weight=.5)


def infer(sequence,directory,progress=None):
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest=load_bundle(directory)
    states=[];boundaries=[]
    for member in manifest['models']:
        if sequence.match_id in member['training_matches']:raise ValueError('Input match was in training')
        if progress:progress(dict(stage='network',model=member['architecture'],device='cpu',precision='float64'))
        checkpoint=torch.load(Path(directory)/member['file'],map_location='cpu',weights_only=True)
        if checkpoint['architecture']!=member['architecture']:raise ValueError('Architecture mismatch')
        model=SegmentPossession(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.double().cpu().eval()
        if model.margin!=1056:raise ValueError('C context margin changed')
        state,boundary=predict_tracking(model,sequence,'cpu',batch_size=1,central=1024)
        if state.dtype!=np.float64 or boundary.dtype!=np.float64:raise ValueError('Float64 evidence required')
        states.append(state);boundaries.append(boundary)
        del model
    return fuse_decode(states,boundaries,sequence.period)
