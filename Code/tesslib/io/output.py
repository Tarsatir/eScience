import pathlib
import os
import datetime

def set_output_dir(path,outdirname=None):
    timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if outdirname is None:
        outdir=timestamp
    else:
        outdir=outdirname+'_'+timestamp
    outpath = os.path.join(path,outdir)
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    return outpath
    

