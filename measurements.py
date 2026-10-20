"""View-dependent estimates in ORIGINAL image coordinates; no thickness-from-color heuristic."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

NAN=float('nan')

def roi_slice(roi,shape):
    if roi is None:raise ValueError('ROI missing')
    if len(roi)!=4 or any(int(v)!=v for v in roi):raise ValueError('ROI must be integer [x,y,width,height]')
    x,y,w,h=map(int,roi);H,W=shape[:2]
    if x<0 or y<0 or w<1 or h<1 or x+w>W or y+h>H:raise ValueError(f'ROI {roi} outside image {W}x{H}')
    return slice(y,y+h),slice(x,x+w)

def largest(mask):
    labels,n=ndi.label(mask,structure=np.ones((3,3),dtype=bool))
    if n==0:return np.zeros_like(mask,dtype=bool)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    return labels==sizes.argmax()

def circle_fit(points,min_span_deg=35,max_rmse_px=2.,max_radius_px=10000.):
    if not 0<min_span_deg<=360 or max_rmse_px<=0 or max_radius_px<=0:
        raise ValueError('Invalid circle-fit thresholds')
    p=np.asarray(points,float)
    fail={'valid':False,'radius_px':NAN,'cx':NAN,'cy':NAN,'rmse_px':NAN,'span_deg':NAN,'status':'too_few_points'}
    if len(p)<12:return fail
    mean=p.mean(0);q=p-mean
    a=np.c_[2*q,np.ones(len(q))]
    init=np.linalg.lstsq(a,(q*q).sum(1),rcond=None)[0]
    rad2=init[2]+sum(init[:2]**2)
    if rad2<=0:return dict(fail,status='degenerate')
    result=least_squares(lambda v:np.linalg.norm(q-v[:2],axis=1)-v[2],
                         [*init[:2],np.sqrt(rad2)],bounds=([-np.inf,-np.inf,.1],[np.inf,np.inf,np.inf]),
                         loss='soft_l1',f_scale=1.,max_nfev=500)
    cx,cy=result.x[:2]+mean;r=result.x[2]
    errors=np.linalg.norm(p-[cx,cy],axis=1)-r;rmse=float(np.sqrt(np.mean(errors**2)))
    angles=np.sort(np.mod(np.arctan2(p[:,1]-cy,p[:,0]-cx),2*np.pi))
    span=float(np.degrees(2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))))
    valid=result.success and rmse<=max_rmse_px and span>=min_span_deg and r<=max_radius_px
    return {'valid':bool(valid),'radius_px':float(r) if valid else NAN,'cx':float(cx),'cy':float(cy),
            'rmse_px':rmse,'span_deg':span,'status':'ok' if valid else 'poor_circle_fit'}

def meniscus(labels,confidence,roi,threshold=.7,contact_margin=3,**fit_args):
    fail={'valid':False,'radius_px':NAN,'cx':NAN,'cy':NAN,'rmse_px':NAN,'span_deg':NAN,'status':'disabled'}
    if roi is None:return fail,np.empty((0,2))
    sy,sx=roi_slice(roi,labels.shape)
    # Only a coating-to-background free interface, excluding contact with tool/substrate.
    liquid=largest(labels==2)
    air=ndi.binary_dilation(labels==0)
    solid=ndi.binary_dilation((labels==1)|(labels==3),iterations=max(1,contact_margin))
    edge=liquid & air & ~solid & (confidence>=threshold)
    edge[[0,-1],:]=False;edge[:,[0,-1]]=False
    chosen=np.zeros_like(edge);chosen[sy,sx]=edge[sy,sx]
    selected=largest(chosen);y,x=np.nonzero(selected);pts=np.c_[x,y]
    if len(pts)<12:return dict(fail,status='insufficient_free_interface'),pts
    # Truncating the arc at ROI edges is allowed: we fit the visible arc, not a full circle.
    return circle_fit(pts,**fit_args),pts

def film_thickness(labels,confidence,roi,threshold=.7,min_pixels=3,min_columns=10,min_fraction=.7):
    if min_pixels<1 or min_columns<1 or not 0<min_fraction<=1:
        raise ValueError('Invalid thickness thresholds')
    result={'thickness_px':NAN,'thickness_p10_px':NAN,'thickness_p90_px':NAN,'valid_columns':0,'status':'disabled'}
    if roi is None:return result
    sy,sx=roi_slice(roi,labels.shape);values=[]
    for x in range(sx.start,sx.stop):
        col=labels[sy,x];y=np.flatnonzero(col==2)
        if not len(y) or np.any(np.diff(y)!=1):continue
        lo,hi=y[0]+sy.start,y[-1]+sy.start
        # Both interfaces must be inside the ROI, with air above and substrate below.
        if lo<=sy.start or hi+1>=sy.stop:continue
        if labels[lo-1,x]!=0 or labels[hi+1,x]!=1:continue
        if np.min(confidence[lo-1:hi+2,x])<threshold:continue
        if hi-lo+1<min_pixels:continue
        values.append(hi-lo+1)
    result['valid_columns']=len(values)
    if len(values)<min_columns or len(values)/(sx.stop-sx.start)<min_fraction:
        return dict(result,status='unresolved_or_incomplete_interfaces')
    return dict(result,thickness_px=float(np.median(values)),thickness_p10_px=float(np.percentile(values,10)),
                thickness_p90_px=float(np.percentile(values,90)),status='ok')

class TemplateTracker:
    """Fixed-template NCC for a textured rigid feature; no mask-centroid speed estimate.

    Search around last accepted location. Ambiguous/weak matches are rejected. A rejected
    observation is not used for updating the tracking position. Translation only.
    """
    def __init__(self,roi,search_radius=40,min_score=.8,min_peak_margin=.04,max_jump_px=25):
        self.roi=roi;self.radius=int(search_radius);self.min_score=min_score;self.margin=min_peak_margin
        self.max_jump=max_jump_px;self.template=None;self.pos=None
        if self.radius<1 or self.max_jump<=0:raise ValueError('Invalid tracking search/jump settings')
        if not -1<=min_score<=1 or not 0<=min_peak_margin<=2:raise ValueError('Invalid NCC thresholds')
    def update(self,rgb):
        gray=np.asarray(rgb,dtype=float)@np.array([.299,.587,.114])
        if self.template is None:
            sy,sx=roi_slice(self.roi,gray.shape);t=gray[sy,sx].copy()
            if min(t.shape)<5 or t.std()<3:raise ValueError('Tracking template needs a larger textured feature')
            self.template=t-t.mean();self.pos=np.array([sx.start,sy.start],float)
            return self.pos+[(t.shape[1]-1)/2,(t.shape[0]-1)/2],1.,'initialized'
        t=self.template;h,w=t.shape;H,W=gray.shape
        x,y=np.rint(self.pos).astype(int)
        x0=max(0,x-self.radius);y0=max(0,y-self.radius)
        x1=min(W,x+w+self.radius);y1=min(H,y+h+self.radius)
        s=gray[y0:y1,x0:x1];ones=np.ones(t.shape)
        numerator=fftconvolve(s,t[::-1,::-1],mode='valid')
        sums=fftconvolve(s,ones,mode='valid');sums2=fftconvolve(s*s,ones,mode='valid')
        denom=np.sqrt(np.maximum(sums2-sums*sums/t.size,0)*np.sum(t*t))
        score=np.divide(numerator,denom,out=np.full_like(numerator,-1.),where=denom>1e-8)
        iy,ix=np.unravel_index(np.argmax(score),score.shape);best=float(score[iy,ix])
        other=score.copy();other[max(0,iy-3):iy+4,max(0,ix-3):ix+4]=-1
        if best<self.min_score or best-float(other.max())<self.margin:
            return None,best,'weak_or_ambiguous_match'
        if (ix==0 and x0>0) or (iy==0 and y0>0) or (ix==score.shape[1]-1 and x1<W) or (iy==score.shape[0]-1 and y1<H):
            return None,best,'search_boundary'
        offset=np.zeros(2)
        # Subpixel parabolic interpolation of NCC peak.
        for axis,idx,limit in [(0,ix,score.shape[1]),(1,iy,score.shape[0])]:
            if 0<idx<limit-1:
                v=score[iy,ix-1:ix+2] if axis==0 else score[iy-1:iy+2,ix]
                den=v[0]-2*v[1]+v[2]
                if abs(den)>1e-8:offset[axis]=np.clip(.5*(v[0]-v[2])/den,-.5,.5)
        new=np.array([x0+ix,y0+iy])+offset
        if np.linalg.norm(new-self.pos)>self.max_jump:return None,best,'jump_rejected'
        self.pos=new
        return new+[(w-1)/2,(h-1)/2],best,'ok'

def kinematics(times,positions,window=9,max_gap_s=.2,max_speed=None):
    """Offline centred quadratic local fits on continuous VALID runs. No gap bridging."""
    t=np.asarray(times,float);s=np.asarray(positions,float);n=len(t)
    if window<5 or window%2==0 or max_gap_s<=0:raise ValueError('Use odd window>=5 and positive max gap')
    if len(s)!=n or not np.isfinite(t).all():raise ValueError('Invalid time/position arrays')
    if max_speed is not None and (not np.isfinite(max_speed) or max_speed<=0):raise ValueError('max_speed must be positive')
    if n>1 and np.any(np.diff(t)<=0):raise ValueError('Times must increase')
    v=np.full(n,np.nan);a=v.copy();smooth=v.copy();quality=np.full(n,'insufficient_window',dtype=object)
    good=np.isfinite(s)
    if max_speed is not None:
        # Reject both ends of a jump; never quietly fit an extreme tracking jump.
        jumps=np.flatnonzero(np.isfinite(np.diff(s)) & (np.abs(np.diff(s)/np.diff(t))>max_speed))
        good[jumps]=False;good[jumps+1]=False
    quality[~good]='invalid_position'
    runs=[];current=[]
    for i in range(n):
        if not good[i] or (current and t[i]-t[current[-1]]>max_gap_s):
            if current:runs.append(current);current=[]
        if good[i]:current.append(i)
    if current:runs.append(current)
    half=window//2
    for run in runs:
        for j in range(half,len(run)-half):
            i=run[j];idx=run[j-half:j+half+1];dt=t[idx]-t[i]
            coefficients=np.polynomial.polynomial.polyfit(dt,s[idx],2)
            smooth[i],v[i],a[i]=coefficients[0],coefficients[1],2*coefficients[2];quality[i]='ok'
    return smooth,v,a,quality
