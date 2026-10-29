import glob, os, sys, zlib
import numpy as np, cv2
sys.path.insert(0,'/mnt/e/10/p0_sim/train')
from real_eval import slice_boxes

L='/mnt/e/10/p0_sim/data/p0_dota_sim_v4/train/layout/masked_all/test'
def var_score(g, side):
    g=g.astype(np.float32); mu=cv2.blur(g,(9,9)); mu2=cv2.blur(g*g,(9,9))
    return cv2.blur(np.maximum(mu2-mu*mu,0.0),(side,side),borderType=cv2.BORDER_CONSTANT)

for side in (130,196):
    tot=dict(tiles=0,pos=0,gt=0,cp=0.0,cr=0.0)
    for ip in sorted(glob.glob(L+'/images/*.png')):
        bgr=cv2.imread(ip); H,W=bgr.shape[:2]
        g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        stem=os.path.splitext(os.path.basename(ip))[0]
        boxes=[]
        lp=os.path.join(L,'labelTxt',stem+'.txt')
        if os.path.isfile(lp):
            for line in open(lp,errors='ignore'):
                t=line.split()
                if len(t)<9: continue
                pts=np.array(t[:8],float).reshape(4,2); c=pts.mean(0)
                boxes.append((c[0],c[1]))
        bs=slice_boxes(H,W,side,0.2)
        cent=[((b[0]+b[2])/2.0,(b[1]+b[3])/2.0) for b in bs]
        if not cent or not boxes: continue
        sm=var_score(g,side); sh,sw=sm.shape
        val=[float(sm[int(min(max(c[1],0),sh-1)),int(min(max(c[0],0),sw-1))]) for c in cent]
        order=sorted(range(len(cent)),key=lambda i:-val[i]); kk=min(4,len(cent))
        def cov(sel):
            n=0
            for (cx,cy) in boxes:
                if any(abs(cx-tx)<=side/2 and abs(cy-ty)<=side/2 for (tx,ty) in sel): n+=1
            return n
        pos=sum(1 for (tx,ty) in cent if any(abs(cx-tx)<=side/2 and abs(cy-ty)<=side/2 for (cx,cy) in boxes))
        rng=np.random.default_rng(zlib.crc32(stem.encode())); nd=50; ra=0.0
        for _ in range(nd):
            idx=rng.permutation(len(cent))[:kk]; ra+=cov([cent[i] for i in idx])
        ra/=nd
        tot['tiles']+=len(cent); tot['pos']+=pos; tot['gt']+=len(boxes)
        tot['cp']+=cov([cent[i] for i in order[:kk]]); tot['cr']+=ra
    print('sim side=%d prev=%.1f%%  prior=%.1f%%  rand=%.1f%%  gain=%+.1f pt' % (
        side, 100*tot['pos']/tot['tiles'], 100*tot['cp']/tot['gt'], 100*tot['cr']/tot['gt'],
        100*(tot['cp']-tot['cr'])/tot['gt']))
