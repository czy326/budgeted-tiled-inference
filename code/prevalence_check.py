import glob, json, os, sys, zlib
import numpy as np, cv2
sys.path.insert(0,'/mnt/e/10/p0_sim/train')
from real_eval import load_rgb, slice_boxes
from lopo_sen2ships import find_ports, scene_of

def var_score(rgb, side):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu=cv2.blur(g,(9,9)); mu2=cv2.blur(g*g,(9,9))
    return cv2.blur(np.maximum(mu2-mu*mu,0.0),(side,side),borderType=cv2.BORDER_CONSTANT)

def run(side, k, ports):
    tot=dict(tiles=0,pos=0,gt=0,cov_r=0.0,cov_a=0.0)
    for port,cf in ports:
        p,boxes=scene_of(cf)
        if not os.path.isfile(p): continue
        rgb=load_rgb(p,'pct'); H,W=rgb.shape[:2]
        bs=slice_boxes(H,W,side,0.2)
        cent=[((b[0]+b[2])/2.0,(b[1]+b[3])/2.0) for b in bs]
        if not cent: continue
        smap=var_score(rgb,side); sh,sw=smap.shape
        val=[float(smap[int(min(max(c[1],0),sh-1)),int(min(max(c[0],0),sw-1))]) for c in cent]
        order=sorted(range(len(cent)),key=lambda i:-val[i])
        kk=min(k,len(cent))
        def cov(sel):
            n=0
            for (cx,cy,w,h) in boxes:
                for (tx,ty) in sel:
                    if abs(cx-tx)<=side/2 and abs(cy-ty)<=side/2: n+=1; break
            return n
        # prevalence
        pos=0
        for (tx,ty) in cent:
            if any(abs(cx-tx)<=side/2 and abs(cy-ty)<=side/2 for (cx,cy,w,h) in boxes): pos+=1
        rng=np.random.default_rng(zlib.crc32(port.encode())); nd=50
        ra=0.0
        for _ in range(nd):
            idx=rng.permutation(len(cent))[:kk]
            ra+=cov([cent[i] for i in idx])
        ra/=nd
        tot['tiles']+=len(cent); tot['pos']+=pos; tot['gt']+=len(boxes)
        tot['cov_r']+=cov([cent[i] for i in order[:kk]]); tot['cov_a']+=ra
    if not tot['gt']: return None
    return dict(side=side,k=k,prev=100*tot['pos']/tot['tiles'],
                gt=tot['gt'], cov_prior=100*tot['cov_r']/tot['gt'],
                cov_rand=100*tot['cov_a']/tot['gt'],
                gain=100*(tot['cov_r']-tot['cov_a'])/tot['gt'])

ports=find_ports()
print('%-6s %5s %8s %10s %10s %8s' % ('side','k','prev%','prior%','rand%','gain pt'))
out=[]
for side in (240,320,480,640,960):
    r=run(side,4,ports)
    if r:
        out.append(r)
        print('%-6d %5d %7.1f%% %9.1f%% %9.1f%% %+7.1f' % (r['side'],r['k'],r['prev'],r['cov_prior'],r['cov_rand'],r['gain']))
json.dump(out,open('/mnt/e/10/p0_sim/data/support/prevalence_real.json','w'),indent=2)
