import glob, json, os, sys
import numpy as np
sys.path.insert(0,'/mnt/e/10/p0_sim/train')
from real_eval import load_rgb, slice_boxes
from lopo_sen2ships import find_ports, scene_of

def tile_stats(H,W,side,ov,boxes):
    bs=slice_boxes(H,W,side,ov); n=0; pos=0; cnts=[]
    for (x0,y0,x1,y1) in bs:
        n+=1; c=0
        for (cx,cy,w,h) in boxes:
            if abs(cx-(x0+x1)/2)<=side/2 and abs(cy-(y0+y1)/2)<=side/2: c+=1
        cnts.append(c); pos += (c>0)
    return n,pos,cnts

print('== 真实 sen2ships (side 640)')
tn=tp=0; allc=[]
for port,cf in find_ports():
    p,boxes=scene_of(cf)
    if not os.path.isfile(p): continue
    rgb=load_rgb(p,'pct'); H,W=rgb.shape[:2]
    n,pos,c=tile_stats(H,W,640,0.2,boxes); tn+=n; tp+=pos; allc+=c
print('  瓦片=%d 含目标比例=%.1f%% 每正瓦片均值=%.1f' % (tn, 100*tp/max(tn,1), np.mean([x for x in allc if x>0]) if tp else 0))

print('== 仿真 v4 test (side 130, 前60帧)')
L='/mnt/e/10/p0_sim/data/p0_dota_sim_v4/train/layout/masked_all/test'
tn=tp=0; allc=[]
for ip in sorted(glob.glob(L+'/images/*.png'))[:60]:
    import cv2
    bgr=cv2.imread(ip); H,W=bgr.shape[:2]
    stem=os.path.splitext(os.path.basename(ip))[0]
    boxes=[]
    lp=os.path.join(L,'labelTxt',stem+'.txt')
    if os.path.isfile(lp):
        for line in open(lp,errors='ignore'):
            t=line.split()
            if len(t)<9: continue
            pts=np.array(t[:8],float).reshape(4,2); c=pts.mean(0)
            e1,e2=pts[1]-pts[0],pts[2]-pts[1]
            boxes.append((c[0],c[1],np.linalg.norm(e1),np.linalg.norm(e2)))
    n,pos,c=tile_stats(H,W,130,0.2,boxes); tn+=n; tp+=pos; allc+=c
print('  瓦片=%d 含目标比例=%.1f%% 每正瓦片均值=%.1f' % (tn, 100*tp/max(tn,1), np.mean([x for x in allc if x>0]) if tp else 0))
