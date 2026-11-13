"""Contrasting painted-divider proposals, independent of marking hue.

Proposals require two visible, near-parallel dividers with compatible endpoints.
They are estimates, not an exhaustive inventory of spaces hidden by vehicles.
"""
import cv2
import numpy as np
from .markings import paint_response


def discover_spaces(image):
    h, w = image.shape[:2]
    scale = min(1., 1400 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale)
    sh, sw = small.shape[:2]
    value=paint_response(small)
    paint=value>8
    mask=paint.astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 15, 45)
    raw = cv2.HoughLinesP(cv2.bitwise_or(mask, edges), 1, np.pi/360, threshold=20,
                          minLineLength=max(30, min(sh, sw)*.08),
                          maxLineGap=max(8, min(sh, sw)*.018))
    lsd = cv2.createLineSegmentDetector().detect(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))[0]
    lines = list([] if raw is None else raw[:, 0])
    lines += list([] if lsd is None else lsd[:, 0])
    segments = []
    for line in lines:
        a, b = line[:2].astype(float), line[2:].astype(float)
        direction = b-a
        length = np.linalg.norm(direction)
        if length < max(30, min(sh,sw)*.08): continue
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        samples = a[None,:] + np.linspace(.05,.95,40)[:,None]*(b-a)
        def brightness(offset,paint_only=False):
            xy = np.rint(samples + normal*offset).astype(int)
            xy[:,0] = np.clip(xy[:,0],0,sw-1); xy[:,1] = np.clip(xy[:,1],0,sh-1)
            values=value[xy[:,1],xy[:,0]].astype(float)
            if paint_only:values=np.where(paint[xy[:,1],xy[:,0]],values,0)
            return values
        centre = np.maximum.reduce([brightness(t,True) for t in (-4,0,4)])
        stripe = (centre>12) & (centre-brightness(-12)>6) & (centre-brightness(12)>6)
        if np.mean(stripe) < .35: continue
        if direction[1] < 0 or (abs(direction[1]) < .001 and direction[0] < 0):
            direction = -direction
        segments.append((a, b, direction, length))
    # Merge duplicate Hough edges before pairing neighbouring dividers.
    groups = []
    for a, b, u, length in sorted(segments, key=lambda s: -s[3]):
        for g in groups:
            v = g['u']; n = np.array([-v[1], v[0]])
            lo, hi = sorted([a@v, b@v])
            if (abs(u@v) > .99 and abs(((a+b)/2-g['mid'])@n) < max(6, min(sh, sw)*.012)
                    and min(hi, g['hi'])-max(lo, g['lo']) > -10):
                g['lo'] = min(g['lo'], lo); g['hi'] = max(g['hi'], hi)
                break
        else:
            lo, hi = sorted([a@u, b@u])
            groups.append(dict(u=u, mid=(a+b)/2, lo=lo, hi=hi))
    candidates = []
    for i, g in enumerate(groups):
        u = g['u']; n = np.array([-u[1], u[0]])
        for j in range(i+1, len(groups)):
            other = groups[j]
            if abs(u@other['u']) < .985: continue
            # Project the second divider into the first divider's coordinates.
            v = other['u']; m = other['mid']
            ends = [m+v*(other[k]-m@v) for k in ('lo', 'hi')]
            lo2, hi2 = sorted([p@u for p in ends])
            lo, hi = max(g['lo'], lo2), min(g['hi'], hi2)
            depth = hi-lo
            width = abs((m-g['mid'])@n)
            if depth < min(sh, sw)*.20 or not .4 < width/depth < .95: continue
            if abs(g['lo']-lo2) > depth*.3 or abs(g['hi']-hi2) > depth*.3: continue
            # A visible intermediate divider means these are multiple spaces.
            left, right = sorted([g['mid']@n, m@n])
            if any(k not in (i,j) and abs(t['u']@u) > .985
                   and left+6 < t['mid']@n < right-6
                   and abs(t['lo']-lo)<depth*.15 and abs(t['hi']-hi)<depth*.15
                   and min(t['hi'],hi)-max(t['lo'],lo) > depth*.85
                   for k,t in enumerate(groups)): continue
            points = np.array([u*lo+n*(g['mid']@n), u*lo+n*(m@n),
                               u*hi+n*(m@n), u*hi+n*(g['mid']@n)])
            points[:,0] = np.clip(points[:,0]/sw, 0, 1)
            points[:,1] = np.clip(points[:,1]/sh, 0, 1)
            if cv2.contourArea(points.astype(np.float32)) < .22:
                candidates.append(points.tolist())
    # Repeated bays provide stronger evidence than an isolated rectangle on a
    # car roof or sign. Keep candidates supported by a neighbouring similar bay.
    def compatible(a,b):
        a=np.array(a)*[sw,sh];b=np.array(b)*[sw,sh]
        da=a[3]-a[0];db=b[3]-b[0]
        la=np.linalg.norm(da);lb=np.linalg.norm(db)
        wa=np.linalg.norm(a[1]-a[0]);wb=np.linalg.norm(b[1]-b[0])
        u=da/la
        return (abs(da@db/(la*lb))>.985 and min(la,lb)/max(la,lb)>.75
                and min(wa,wb)/max(wa,wb)>.65
                and abs((a.mean(axis=0)-b.mean(axis=0))@u)<max(la,lb)*.18
                and np.linalg.norm(a.mean(axis=0)-b.mean(axis=0))<max(wa,wb)*2.3)
    candidates=[a for i,a in enumerate(candidates)
                if any(i!=j and compatible(a,b) for j,b in enumerate(candidates))]
    unique=[]
    for p in sorted(candidates,key=lambda p:-cv2.contourArea(np.array(p,np.float32))):
        a=np.array(p,np.float32)
        if any(cv2.intersectConvexConvex(a,np.array(q,np.float32))[0]
               / max(cv2.contourArea(a),1e-9)>.5 for q in unique):continue
        unique.append(p)
    candidates=unique
    rows=[]
    for p in sorted(candidates,key=lambda p:np.mean(p,axis=0)[1]):
        if not rows or abs(np.mean(p,axis=0)[1]-np.mean(rows[-1][0],axis=0)[1])>.1:
            rows.append([])
        rows[-1].append(p)
    candidates=[p for row in rows for p in sorted(row,key=lambda p:np.mean(p,axis=0)[0])]
    spaces = [{'id':f'P{i:02d}', 'points':p} for i,p in enumerate(candidates,1)]
    return {'version':1, 'source':'white_lines', 'review_required':True, 'spaces':spaces,
            'diagnostics':{'line_segments':len(segments), 'merged_dividers':len(groups),
                'proposed_spaces':len(spaces)},
            'note':'Painted-line proposals: colour and local contrast provide evidence, while repeated bay geometry validates it. Hidden or distorted markings may be missed.'}
