"""Train U-Net on reviewed, run-separated CVAT masks; save best and resumable last checkpoints."""
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import CLASS_NAMES,CLASS_PALETTE,IGNORE
from .dataset import load_records,split_records,SegmentationDataset
from .model import UNet

def loss_fn(logits,target,weights=None):
    valid=target!=IGNORE
    if not valid.any():return logits.sum()*0
    ce=F.cross_entropy(logits.float(),target,weight=weights,ignore_index=IGNORE)
    probs=logits.float().softmax(1)*valid[:,None]
    truth=F.one_hot(target.masked_fill(~valid,0),len(CLASS_NAMES)).permute(0,3,1,2).float()*valid[:,None]
    dice=(2*(probs*truth).sum((0,2,3))+1e-6)/(probs.sum((0,2,3))+truth.sum((0,2,3))+1e-6)
    return ce+1-dice.mean()

@torch.no_grad()
def evaluate(model,loader,device,weights):
    k=len(CLASS_NAMES);model.eval();cm=torch.zeros((k,k),device=device,dtype=torch.int64);total=0.;n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); pred=model(x); total+=float(loss_fn(pred,y,weights));n+=1
        valid=y!=IGNORE; code=k*y[valid]+pred.argmax(1)[valid]
        cm+=torch.bincount(code,minlength=k*k).reshape(k,k)
    c=cm.cpu().numpy();inter=c.diagonal();union=c.sum(0)+c.sum(1)-inter
    iou=np.divide(inter,union,out=np.full(k,np.nan),where=union>0)
    return total/n,iou,float(np.nanmean(iou)),c.tolist()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--val-groups',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--crop',type=int,nargs=2,default=[384,512])
    p.add_argument('--channels',type=int,choices=[1,3],default=3);p.add_argument('--base',type=int,default=32)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--patience',type=int,default=15)
    p.add_argument('--resume');p.add_argument('--class-weights',choices=['auto','none'],default='auto')
    p.add_argument('--no-amp',action='store_true');p.add_argument('--device',default=None)
    a=p.parse_args()
    if min(a.crop)<32 or a.epochs<1 or a.batch_size<1 or a.lr<=0 or a.workers<0 or a.patience<1:
        p.error('Invalid training arguments; crop dimensions must be >=32')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    rows=load_records(a.data);tr,va=split_records(rows,a.val_groups)
    signature=hashlib.sha256(json.dumps({'rows':rows,'val':sorted(a.val_groups)},sort_keys=True).encode()).hexdigest()
    out=Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume: raise ValueError('Use a new --out or --resume')
    out.mkdir(parents=True,exist_ok=True)
    config={'channels':a.channels,'classes':len(CLASS_NAMES),'base':a.base}
    model=UNet(**config).to(device)
    train_loader=DataLoader(SegmentationDataset(tr,True,a.crop,a.channels),batch_size=a.batch_size,
                            shuffle=True,num_workers=a.workers,drop_last=False)
    val_loader=DataLoader(SegmentationDataset(va,False,channels=a.channels),batch_size=1,num_workers=a.workers)
    counts=np.zeros(len(CLASS_NAMES),dtype=np.int64)
    from PIL import Image
    for r in tr:
        m=np.asarray(Image.open(r['mask']));counts+=np.bincount(m[m!=IGNORE],minlength=len(CLASS_NAMES))
    if np.any(counts==0):raise ValueError(f'Every class needs training pixels: {dict(zip(CLASS_NAMES,counts.tolist()))}')
    weight=None
    if a.class_weights=='auto':
        freq=counts/counts.sum();w=np.clip(np.median(freq)/freq,.1,10.)
        weight=torch.tensor(w,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=4,factor=.5)
    amp=device.type=='cuda' and not a.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start=0;best=-1.;bad=0;history=[]
    settings={'crop':a.crop,'channels':a.channels,'base':a.base,'batch_size':a.batch_size,'class_weights':a.class_weights}
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck.get('format')!='blade-unet-v1' or ck['signature']!=signature or ck['settings']!=settings:
            raise ValueError('Resume dataset/split/settings mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler']);start=ck['epoch']+1;best=ck['best_miou'];bad=ck['bad_epochs'];history=ck['history']
        torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng'])
        if device.type=='cuda' and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    (out/'split.json').write_text(json.dumps({'train':tr,'validation':va},indent=2))
    (out/'run.json').write_text(json.dumps({'arguments':vars(a),'class_pixels':counts.tolist()},indent=2))
    for epoch in range(start,a.epochs):
        model.train();total=0.
        for x,y in train_loader:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):loss=loss_fn(model(x),y,weight)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();total+=float(loss.detach())
        val_loss,ious,miou,cm=evaluate(model,val_loader,device,weight);scheduler.step(miou)
        record={'epoch':epoch+1,'train_loss':total/len(train_loader),'val_loss':val_loss,'miou':miou,
                **{f'iou_{n}':None if np.isnan(v) else float(v) for n,v in zip(CLASS_NAMES,ious)}}
        history.append(record);print(record,flush=True)
        improved=miou>best
        if improved:best=miou;bad=0
        else:bad+=1
        ck={'format':'blade-unet-v1','model':model.state_dict(),'model_config':config,'class_names':CLASS_NAMES,
            'palette':CLASS_PALETTE,'epoch':epoch,'best_miou':best,'signature':signature,'settings':settings,
            'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'bad_epochs':bad,'history':history,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        if improved:torch.save(ck,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False))
        (out/'confusion_matrix.json').write_text(json.dumps(cm))
        if bad>=a.patience:break
if __name__=='__main__':main()
