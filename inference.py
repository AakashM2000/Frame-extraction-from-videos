"""Tiled native-resolution segmentation and calibrated offline video measurements."""
import argparse,csv,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from . import CLASS_NAMES,CLASS_PALETTE
from .model import load_model
from .video_preprocessing import frames
from .measurements import TemplateTracker,meniscus,film_thickness,kinematics,roi_slice

class Segmenter:
    def __init__(self,checkpoint,device=None,tile=512,overlap=64):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model,self.ck=load_model(checkpoint,self.device);self.channels=self.ck['model_config']['channels']
        self.tile=tile;self.overlap=overlap
        if tile<32 or not 0<=overlap<tile:raise ValueError('Require tile>=32 and 0<=overlap<tile')
    @torch.no_grad()
    def predict(self,rgb):
        im=Image.fromarray(rgb).convert('L' if self.channels==1 else 'RGB')
        arr=np.asarray(im,dtype=np.float32)/255.
        if arr.ndim==2:arr=arr[...,None]
        H,W=arr.shape[:2];tile=self.tile;step=tile-self.overlap
        def starts(n):return list(range(0,max(n-tile,0),step))+[max(n-tile,0)]
        total=np.zeros((len(CLASS_NAMES),H,W),np.float32);den=np.zeros((H,W),np.float32)
        win=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.01).astype(np.float32)
        for y in starts(H):
            for x in starts(W):
                patch=arr[y:y+tile,x:x+tile];h,w=patch.shape[:2]
                patch=np.pad(patch,((0,tile-h),(0,tile-w),(0,0)),mode='edge')
                inp=torch.from_numpy(patch.transpose(2,0,1).copy())[None].to(self.device)
                prob=self.model(inp).float().softmax(1)[0].cpu().numpy()[:,:h,:w]
                total[:,y:y+h,x:x+w]+=prob*win[:h,:w];den[y:y+h,x:x+w]+=win[:h,:w]
        probs=total/den
        return probs.argmax(0).astype(np.uint8),probs.max(0)

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    size=cfg['image_size']
    if len(size)!=2 or any(not isinstance(v,int) or v<16 for v in size):raise ValueError('image_size must be [width,height]')
    scale=cfg.get('mm_per_pixel')
    if scale is not None and (not np.isfinite(scale) or scale<=0):raise ValueError('Invalid mm_per_pixel')
    if scale is not None and not cfg.get('uniform_scale_confirmed',False):
        raise ValueError('Confirm uniform in-plane scale before physical measurements')
    if cfg.get('view') not in ['side','top','unknown']:raise ValueError('view must be side, top or unknown')
    threshold=cfg.get('min_confidence',.7)
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('min_confidence must be in [0,1]')
    for key in ['left_meniscus_roi','right_meniscus_roi','film_roi','tool_template_roi','substrate_template_roi']:
        if cfg.get(key) is not None:roi_slice(cfg[key],(size[1],size[0]))
    axis=np.asarray(cfg.get('motion_axis',[1,0]),float)
    if axis.shape!=(2,) or not np.isfinite(axis).all() or np.linalg.norm(axis)==0:raise ValueError('Invalid motion axis')
    return cfg

class Analyzer:
    def __init__(self,cfg):
        self.cfg=cfg;self.tool=None;self.substrate=None;self.origin=None
        settings=cfg.get('tracking',{})
        for key,attr in [('tool_template_roi','tool'),('substrate_template_roi','substrate')]:
            if cfg.get(key):setattr(self,attr,TemplateTracker(cfg[key],**settings))
        if self.tool and not self.substrate and not cfg.get('stationary_substrate_confirmed',False):
            raise ValueError('Motion needs a substrate feature OR confirmed stationary substrate and fixed camera')
        self.axis=np.array(cfg.get('motion_axis',[1,0]),float);self.axis/=np.linalg.norm(self.axis)
    def measure(self,rgb,labels,confidence):
        c=self.cfg;scale=c.get('mm_per_pixel');threshold=c.get('min_confidence',.7)
        if [rgb.shape[1],rgb.shape[0]]!=c['image_size']:raise ValueError('Video dimensions differ from calibration')
        row={f'area_frac_{name}':float(np.mean(labels==i)) for i,name in enumerate(CLASS_NAMES)}
        row.update(position_px=np.nan,position_mm=np.nan,tracking_status='disabled',tool_ncc=np.nan,substrate_ncc=np.nan)
        geometry=[]
        if self.tool:
            tool,score,status=self.tool.update(rgb);row['tool_ncc']=score
            substrate=np.zeros(2)
            if self.substrate:
                substrate,s_score,s_status=self.substrate.update(rgb);row['substrate_ncc']=s_score
                if substrate is None:status='substrate_'+s_status
            if tool is not None and substrate is not None:
                relative=tool-substrate
                if self.origin is None:self.origin=relative.copy()
                row['position_px']=float((relative-self.origin)@self.axis)
                if scale is not None:row['position_mm']=row['position_px']*scale
                row['tracking_status']='ok'
            else:row['tracking_status']=status
        for side in ['left','right']:
            roi=c.get(side+'_meniscus_roi') if c['view']=='side' else None
            fit,points=meniscus(labels,confidence,roi,threshold,**c.get('circle',{}))
            for key in ['radius_px','rmse_px','span_deg','status']:row[side+'_'+key]=fit[key]
            row[side+'_radius_mm']=fit['radius_px']*scale if scale is not None else np.nan
            if c['view']!='side':row[side+'_status']='requires_side_view'
            geometry.append((fit,points))
        film_enabled=c['view']=='side' and c.get('film_interfaces_confirmed',False) and c.get('horizontal_substrate_confirmed',False)
        film=film_thickness(labels,confidence,c.get('film_roi') if film_enabled else None,threshold,
                            **c.get('thickness',{}))
        if c.get('film_roi') and not film_enabled:film['status']='requires_visible_horizontal_interfaces'
        row.update({('film_'+k if k=='status' else k):v for k,v in film.items()})
        row['thickness_mm']=film['thickness_px']*scale if scale is not None else np.nan
        return row,geometry

def overlay(rgb,labels,geometry,t,config):
    palette=np.asarray(CLASS_PALETTE,np.uint8);out=rgb.copy();fg=labels!=0
    out[fg]=(.65*rgb[fg]+.35*palette[labels[fg]]).astype(np.uint8)
    im=Image.fromarray(out);draw=ImageDraw.Draw(im)
    for color,(fit,points) in zip(['yellow','magenta'],geometry):
        for x,y in points[::max(1,len(points)//100)]:draw.point((int(x),int(y)),fill=color)
        if fit['valid']:
            x,y,r=fit['cx'],fit['cy'],fit['radius_px']
            # Only draw fitted arc near observed points to avoid huge extrapolated circles.
            vec=points-[x,y];pred=np.array([x,y])+r*vec/np.linalg.norm(vec,axis=1)[:,None]
            for px,py in pred:draw.point((int(px),int(py)),fill=color)
    for key,color in [('film_roi','white'),('left_meniscus_roi','yellow'),('right_meniscus_roi','magenta')]:
        if config.get(key):
            x,y,w,h=config[key];draw.rectangle((x,y,x+w-1,y+h-1),outline=color)
    draw.text((5,5),f't={t:.6f} s | preview only; timestamps in CSV',fill='white',stroke_width=1,stroke_fill='black')
    return np.asarray(im)

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:'' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--config',required=True)
    p.add_argument('--out',required=True);p.add_argument('--every',type=int,default=1)
    p.add_argument('--fps',type=float,help='Known constant original FPS; overrides video timestamps')
    p.add_argument('--tile',type=int,default=512);p.add_argument('--overlap',type=int,default=64);p.add_argument('--device')
    p.add_argument('--save-masks',action='store_true');p.add_argument('--no-preview',action='store_true')
    p.add_argument('--preview-fps',type=int,default=15,help='Preview playback rate only; does not change measured times')
    a=p.parse_args()
    if a.every<1 or a.preview_fps<1:p.error('--every and --preview-fps must be positive')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    if a.save_masks:(out/'masks').mkdir()
    cfg=load_config(a.config);seg=Segmenter(a.checkpoint,a.device,a.tile,a.overlap);analyzer=Analyzer(cfg)
    rows=[];container=None;stream=None
    try:
        for idx,t,rgb,time_source in frames(a.input,a.fps,cfg.get('time_scale',1.)):
            if idx%a.every:continue
            labels,confidence=seg.predict(rgb);measurement,geo=analyzer.measure(rgb,labels,confidence)
            row={'frame_idx':idx,'time_s':t,'time_source':time_source,**measurement};rows.append(row)
            if a.save_masks:Image.fromarray(labels).save(out/'masks'/f'{idx:08d}.png')
            rendered=overlay(rgb,labels,geo,t,cfg)
            if len(rows)==1:Image.fromarray(rendered).save(out/'first_overlay.png')
            if not a.no_preview:
                import av
                if container is None:
                    container=av.open(str(out/'preview.mp4'),'w');stream=container.add_stream('libx264',rate=a.preview_fps)
                    # Pad odd dimensions for codec; measurements remain at original coordinates.
                    stream.width=rgb.shape[1]+rgb.shape[1]%2;stream.height=rgb.shape[0]+rgb.shape[0]%2;stream.pix_fmt='yuv420p'
                padded=np.pad(rendered,((0,stream.height-rgb.shape[0]),(0,stream.width-rgb.shape[1]),(0,0)))
                frame=av.VideoFrame.from_ndarray(padded,format='rgb24')
                frame.pts=len(rows)-1;frame.time_base=Fraction(1,a.preview_fps)
                for packet in stream.encode(frame):container.mux(packet)
            if len(rows)%20==0:print(f'Analyzed {len(rows)} frames; video time {t:.3f} s',flush=True)
        if not rows:raise ValueError('No decoded frames')
        unit='mm' if cfg.get('mm_per_pixel') is not None else 'px'
        s,v,acc,status=kinematics([r['time_s'] for r in rows],[r['position_'+unit] for r in rows],**cfg.get('kinematics',{}))
        for i,row in enumerate(rows):
            row['smoothed_position_'+unit]=s[i];row['velocity_'+unit+'_s']=v[i]
            row['speed_'+unit+'_s']=abs(v[i]);row['acceleration_'+unit+'_s2']=acc[i];row['kinematics_status']=status[i]
        write_csv(out/'measurements.csv',rows)
        (out/'analysis_config.json').write_text(json.dumps(cfg,indent=2))
        (out/'provenance.json').write_text(json.dumps({'arguments':vars(a),'checkpoint_classes':seg.ck['class_names'],
            'model_config':seg.ck['model_config'],'note':'Estimated 2D quantities; calibration/quality thresholds need experimental validation'},indent=2))
    finally:
        if container is not None:
            try:
                for packet in stream.encode():container.mux(packet)
            finally:container.close()
    print(out/'measurements.csv')
if __name__=='__main__':main()
