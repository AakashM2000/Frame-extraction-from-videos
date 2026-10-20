"""Interactively select ROIs on the first video frame. Run on a desktop, not headless Colab."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .video_preprocessing import frames
from .measurements import roi_slice

def pick(rgb,title,count=2):
    fig,ax=plt.subplots(figsize=(14,7));ax.imshow(rgb);ax.set_title(title+' | click two points; close window to skip')
    points=plt.ginput(count,timeout=0);plt.close(fig);return np.asarray(points)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fps',type=float);a=p.parse_args()
    iterator=frames(a.video,a.fps)
    try:_,_,rgb,_=next(iterator)
    finally:iterator.close()
    cfg={'image_size':[rgb.shape[1],rgb.shape[0]],'view':'unknown','mm_per_pixel':None,
         'uniform_scale_confirmed':False,'time_scale':1.,'motion_axis':[1,0],
         'stationary_substrate_confirmed':False,'film_interfaces_confirmed':False,
         'horizontal_substrate_confirmed':False}
    cfg['view']=input('View (side/top/unknown): ').strip()
    for key,title in [('tool_template_roi','Tight TEXTURED patch on rigid tool: opposite corners'),
                      ('substrate_template_roi','TEXTURED patch fixed to substrate/stage: opposite corners'),
                      ('left_meniscus_roi','Left liquid-air interface: opposite corners of search area'),
                      ('right_meniscus_roi','Right liquid-air interface: opposite corners of search area'),
                      ('film_roi','Flat film, away from meniscus; include air above and substrate below')]:
        points=pick(rgb,title)
        cfg[key]=None
        if len(points)==2:
            lo=np.floor(points.min(0)).astype(int);hi=np.ceil(points.max(0)).astype(int)
            roi=[*lo,*(hi-lo)];roi=[int(v) for v in roi];roi_slice(roi,rgb.shape);cfg[key]=roi
    points=pick(rgb,'Optional calibration: two endpoints of a known length IN THE MEASUREMENT PLANE')
    if len(points)==2:
        length=float(input('Known distance in mm: '));distance=np.linalg.norm(points[1]-points[0])
        if length<=0 or distance<=0:raise ValueError('Calibration length must be positive')
        cfg['mm_per_pixel']=float(length/distance)
        cfg['uniform_scale_confirmed']=input('Is this an undistorted view with uniform scale in the measurement plane? yes/no: ').lower()=='yes'
    if cfg['tool_template_roi'] and not cfg['substrate_template_roi']:
        cfg['stationary_substrate_confirmed']=input('Are BOTH camera and substrate stationary? yes/no: ').lower()=='yes'
    if cfg['film_roi']:
        cfg['film_interfaces_confirmed']=input('Are the film top and film/substrate interface independently visible? yes/no: ').lower()=='yes'
        cfg['horizontal_substrate_confirmed']=input('Is the substrate horizontal in this side view? yes/no: ').lower()=='yes'
    path=Path(a.out)
    if path.exists():raise ValueError('Output exists; choose another filename')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(cfg,indent=2));print(path)
if __name__=='__main__':main()
