"""Reproduce the C submission from the participant EPTS XML/TXT bundle."""
import argparse,hashlib,json,platform,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Participant bundle root containing tactical/')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from .resources import restrict_resources
    resources=restrict_resources()
    import numpy as np
    import torch
    from .io.tactical import load_tactical
    from .pipeline import infer
    from .submission import write_submission
    started=time.monotonic()
    sequence=load_tactical(args.data)
    print(json.dumps(dict(stage='input',rows=len(sequence.period),resources=resources)),flush=True)
    prediction=infer(sequence,args.weights,lambda row:print(json.dumps(row),flush=True))
    digest=write_submission(prediction,args.output)
    metadata=dict(csv_sha256=digest,rows=len(prediction),weights_manifest_sha256=hashlib.sha256((args.weights/'manifest.json').read_bytes()).hexdigest(),
        runtime=dict(torch=str(torch.__version__),numpy=np.__version__,platform=platform.system(),precision='float64',elapsed_seconds=time.monotonic()-started,**resources))
    args.output.with_suffix('.generation.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(csv_valid=True,rows=len(prediction),sha256=digest)),flush=True)


if __name__=='__main__':main()
