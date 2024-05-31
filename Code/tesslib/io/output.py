import pathlib
import os
import datetime

def set_output_dir(path,outdirname=None,use_timestamp=False):
    if use_timestamp:
        timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if outdirname is None:
            outdir=timestamp
        else:
            outdir=outdirname+'_'+timestamp
    else:
        if outdirname is None:
            raise ValueError('no output directory has been specified')
        else:
            outdir=outdirname
    outpath = os.path.join(path,outdir)
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)          
    return outpath
    

