"""Release-specific identities and the independent decoder objective."""
from itertools import product
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from scipy.special import expit,logit,logsumexp
from possession_c.pipeline import fuse_decode,load_bundle

ROOT=Path(__file__).resolve().parents[1]


def test_manifest_and_completed_pair():
    manifest=load_bundle(ROOT/'weights')
    assert [m['parameters'] for m in manifest['models']]==[3308445,3375873]
    for member in manifest['models']:
        assert member['fold']==0
        assert member['selected_epoch']==100 and member['selected_updates']==2400
        assert len(member['training_matches'])==29  # 3 dense + 14 auxiliary + 12 halves
        assert not set(member['training_matches'])&set(member['held_out_inner']+member['held_out_outer'])


def test_checksum_and_path_escape_rejected(tmp_path):
    manifest=json.loads((ROOT/'weights/manifest.json').read_text())
    for entry in manifest['models']:
        (tmp_path/entry['file']).write_bytes(b'wrong')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):load_bundle(tmp_path)
    member=manifest['models'][0]
    original=member['file'];member['file']='../escape.pt'
    manifest['files']['../escape.pt']=hashlib.sha256(b'outside').hexdigest()
    del manifest['files'][original]
    (tmp_path.parent/'escape.pt').write_bytes(b'outside')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):load_bundle(tmp_path)


def test_fusion_decoder_matches_exhaustive_path_objective():
    rng=np.random.default_rng(913)
    states=[rng.normal(size=(6,3)) for _ in range(2)]
    edges=[rng.normal(size=(6,3,3)) for _ in range(2)]
    scores=logsumexp(np.stack(states),axis=0)-np.log(2)
    scores=scores/6;scores=np.round(scores-logsumexp(scores,axis=-1,keepdims=True),9)
    boundary=np.round(logit(np.clip(np.mean([expit(b) for b in edges],axis=0),1e-6,1-1e-6)),9)
    def objective(path):
        value=sum(scores[i,k] for i,k in enumerate(path))
        for t,(a,b) in enumerate(zip(path,path[1:]),1):
            if a!=b:value+=-2+.5*boundary[t,a,b]
        return value
    best=max(product(range(3),repeat=6),key=objective)
    actual=fuse_decode(states,edges,np.ones(6,'i1'))
    assert objective(actual)==pytest.approx(objective(best),abs=1e-12)


def test_fusion_resets_at_published_period_breaks():
    state=np.full((8,3),-30.)
    state[:3,0]=0;state[5:,1]=0
    actual=fuse_decode([state,state],[np.zeros((8,3,3))]*2,np.array([1,1,1,0,0,2,2,2]))
    np.testing.assert_array_equal(actual,[0,0,0,3,3,1,1,1])
