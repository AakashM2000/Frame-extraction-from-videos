"""Sample native-resolution frames using video presentation timestamps."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image

def frames(path, fps_override=None, time_scale=1.0):
    import av
    if time_scale<=0 or not math.isfinite(time_scale): raise ValueError('time_scale must be positive')
    if fps_override is not None and (fps_override<=0 or not math.isfinite(fps_override)):
        raise ValueError('fps override must be positive')
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        start=None; previous=None
        for i,f in enumerate(container.decode(stream)):
            if fps_override:
                t=i/fps_override; source='explicit_cfr'
            elif f.pts is not None and f.time_base is not None:
                t=float(f.pts*f.time_base); source='pts'
            else: raise ValueError('Missing timestamps. Supply --fps only if true constant capture FPS is known')
            if start is None: start=t
            t=(t-start)*time_scale
            if previous is not None and t<=previous: raise ValueError('Non-increasing video timestamps')
            previous=t
            yield i,t,f.to_ndarray(format='rgb24'),source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',required=True,nargs='+',help='Files or folders (recursive)')
    p.add_argument('--out',required=True)
    p.add_argument('--every-sec',type=float,default=.5)
    p.add_argument('--fps',type=float,help='Explicit constant FPS override; never guessed')
    p.add_argument('--time-scale',type=float,default=1.)
    a=p.parse_args()
    if a.every_sec<=0: p.error('--every-sec must be positive')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Use an empty output folder to avoid stale frames')
    out.mkdir(parents=True,exist_ok=True)
    extensions={'.mp4','.avi','.mov','.mkv','.m4v','.mpeg','.wmv'}
    paths=[]
    for item in a.videos:
        q=Path(item)
        paths.extend(sorted(x for x in q.rglob('*') if x.suffix.lower() in extensions) if q.is_dir() else [q])
    paths=sorted(set(x.resolve() for x in paths))
    if not paths: raise ValueError('No videos found')
    with (out/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['file','video_id','frame_idx','time_s','time_source','video_path','width','height'])
        writer.writeheader()
        for path in paths:
            vid=re.sub(r'[^A-Za-z0-9_-]','_',path.stem)+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:8]
            next_t=0.; count=0
            for idx,t,rgb,source in frames(path,a.fps,a.time_scale):
                if t+1e-9<next_t: continue
                name=f'{vid}_f{idx:08d}.png'; Image.fromarray(rgb).save(out/name)
                writer.writerow(dict(file=name,video_id=vid,frame_idx=idx,time_s=t,time_source=source,
                                     video_path=str(path),width=rgb.shape[1],height=rgb.shape[0]))
                count+=1; next_t=t+a.every_sec
            print(path.name, count, 'frames')
    (out/'extraction.json').write_text(json.dumps(vars(a),indent=2))
if __name__=='__main__': main()
