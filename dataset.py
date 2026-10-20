"""Strict CVAT color-mask conversion, explicit reviewed samples and run-level splits."""
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
from . import CLASS_NAMES,IGNORE

ALIASES={'bg':'background','glass':'substrate','film':'coating','wet film':'coating','wet_film':'coating',
         'liquid':'coating','ink':'coating','meniscus':'coating','blade':'tool','applicator':'tool','doctor blade':'tool'}
EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def labelmap(path):
    mapping={}; names=set()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        fields=line.split(':'); name=fields[0].strip().lower()
        name=ALIASES.get(name,name)
        if name not in CLASS_NAMES and name!='ignore': raise ValueError(f'Unknown CVAT label: {name}')
        color=tuple(map(int,fields[1].split(',')))
        if len(color)!=3 or min(color)<0 or max(color)>255: raise ValueError('Invalid RGB color')
        if color in mapping: raise ValueError('Duplicate label color')
        mapping[color]=IGNORE if name=='ignore' else CLASS_NAMES.index(name); names.add(name)
    if 'background' not in names:
        if (0,0,0) in mapping: raise ValueError('Black foreground conflicts with implicit background')
        mapping[(0,0,0)]=0
    return mapping

def decode_mask(path,mapping):
    image=Image.open(path)
    if image.mode not in ('RGB','RGBA','P'):
        raise ValueError(f'{path}: unpaletted index/grayscale mask requires explicit conversion; do not guess label IDs')
    rgb=np.asarray(image.convert('RGB')); result=np.full(rgb.shape[:2],IGNORE,np.uint8)
    known=np.zeros(result.shape,bool)
    for color,idx in mapping.items():
        match=np.all(rgb==color,axis=-1); result[match]=idx; known|=match
    if not known.all():
        bad=np.unique(rgb[~known],axis=0)[:8].tolist()
        raise ValueError(f'{path}: unmapped colors {bad}; check labelmap.txt; use lossless PNG masks')
    if np.all(result==IGNORE): raise ValueError(f'{path}: every pixel ignored')
    return result

def build(export,images,reviewed,out,group):
    export,images,out=map(Path,(export,images,out))
    if out.exists(): raise ValueError('Use a new output manifest filename')
    mapping=labelmap(export/'labelmap.txt')
    # reviewed.txt explicitly certifies that each named image has been fully annotated.
    wanted=[x.strip() for x in Path(reviewed).read_text().splitlines() if x.strip()]
    if not wanted or len(wanted)!=len(set(wanted)): raise ValueError('Reviewed list empty or duplicated')
    def index(root,extensions):
        result={}
        for path in root.rglob('*'):
            if path.suffix.lower() in extensions:
                if path.stem in result: raise ValueError(f'Duplicate stem {path.stem}; use unique extracted names')
                result[path.stem]=path
        return result
    imgs=index(images,EXT); masks=index(export/'SegmentationClass',{'.png'})
    records=[]; cache=out.parent/(out.stem+'_masks'); cache.mkdir(parents=True,exist_ok=False)
    for name in wanted:
        stem=Path(name).stem
        if stem not in imgs or stem not in masks: raise ValueError(f'Missing image or mask for {name}')
        mask=decode_mask(masks[stem],mapping)
        with Image.open(imgs[stem]) as im:
            if im.size!=(mask.shape[1],mask.shape[0]): raise ValueError(f'Image/mask size mismatch for {name}')
        target=cache/(stem+'.png');Image.fromarray(mask).save(target)
        records.append({'image':str(imgs[stem].resolve()),'mask':str(target.resolve()),'group':group,
                        'id':stem,'image_sha256':hashlib.sha256(imgs[stem].read_bytes()).hexdigest(),
                        'mask_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    out.write_text(json.dumps(records,indent=2));print(f'{len(records)} reviewed samples -> {out}')

def load_records(manifests):
    rows=[]
    for manifest in manifests:
        rows+=json.loads(Path(manifest).read_text())
    if len(rows)<2: raise ValueError('At least two reviewed samples required')
    hashes=[r['image_sha256'] for r in rows]
    if len(set(hashes))!=len(hashes): raise ValueError('Duplicate source images across manifests; remove duplicates')
    for r in rows:
        for field in ('image','mask'):
            if hashlib.sha256(Path(r[field]).read_bytes()).hexdigest()!=r[field+'_sha256']:
                raise ValueError('Source files changed after preparation: '+r[field])
    return rows

def split_records(rows,val_groups):
    groups=set(val_groups)
    if not groups<=set(r['group'] for r in rows): raise ValueError('Unknown validation group')
    train=[r for r in rows if r['group'] not in groups]; val=[r for r in rows if r['group'] in groups]
    if not train or not val: raise ValueError('Need separate acquisition runs for training and validation')
    return train,val

class SegmentationDataset:
    def __init__(self,records,training=False,crop=(384,512),channels=3):
        self.records=records;self.training=training;self.crop=tuple(crop);self.channels=channels
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        import torch
        r=self.records[index];image=Image.open(r['image']).convert('RGB');mask=Image.open(r['mask'])
        if self.training:
            h,w=self.crop;dw=max(0,w-image.width);dh=max(0,h-image.height)
            image=ImageOps.expand(image,(0,0,dw,dh),fill=0);mask=ImageOps.expand(mask,(0,0,dw,dh),fill=IGNORE)
            # Avoid crops containing only ignored padding. Keep valid background-only examples.
            for _ in range(20):
                x=random.randint(0,image.width-w);y=random.randint(0,image.height-h)
                box=(x,y,x+w,y+h)
                if np.any(np.asarray(mask.crop(box))!=IGNORE):break
            else: raise ValueError('Could not select a valid training crop')
            image,mask=image.crop(box),mask.crop(box)
            if random.random()<.5:image,mask=ImageOps.mirror(image),ImageOps.mirror(mask)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
        if self.channels==1:image=image.convert('L')
        arr=np.array(image,dtype=np.float32)/255.
        if arr.ndim==2:arr=arr[...,None]
        return torch.from_numpy(arr.transpose(2,0,1).copy()),torch.from_numpy(np.array(mask,dtype=np.int64))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['export','images','reviewed','out','group']:p.add_argument('--'+arg,required=True)
    a=p.parse_args();build(a.export,a.images,a.reviewed,a.out,a.group)
if __name__=='__main__':main()
