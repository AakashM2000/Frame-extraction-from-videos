"""Independent U-Net implementation. No spin-coater package dependency."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False), nn.GroupNorm(4,b), nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.GroupNorm(4,b), nn.ReLU())

class UNet(nn.Module):
    def __init__(self, channels=3, classes=4, base=32):
        super().__init__()
        if channels not in (1,3) or base < 4 or base % 4:
            raise ValueError('channels must be 1 or 3; base a positive multiple of 4')
        sizes=[base*2**i for i in range(5)]
        self.enc=nn.ModuleList([Block(channels,sizes[0])]+[Block(a,b) for a,b in zip(sizes,sizes[1:])])
        self.dec=nn.ModuleList([Block(sizes[i+1]+sizes[i],sizes[i]) for i in range(3,-1,-1)])
        self.head=nn.Conv2d(base,classes,1)
    def forward(self,x):
        if min(x.shape[-2:])<16:
            raise ValueError('Both image dimensions must be at least 16')
        skips=[]
        for i,block in enumerate(self.enc):
            x=block(F.max_pool2d(x,2) if i else x)
            skips.append(x)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat([x,skip],dim=1))
        return self.head(x)

def load_model(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    from . import CLASS_NAMES
    if ck.get('format')!='blade-unet-v1' or ck.get('class_names')!=CLASS_NAMES:
        raise ValueError('Use a checkpoint trained by this project; old/spin-coater weights are incompatible')
    model=UNet(**ck['model_config']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    return model.eval(),ck
