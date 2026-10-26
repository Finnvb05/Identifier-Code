import numpy as np, time
import gauge as G
from pi_dual import Cam, Role

role = Role('side', 300.0, None, 3.0, 'calib_side.json', 0.01,
            'ref_side.npy', 'below')
cam = Cam(0, exposure=None)
role.cam = cam
time.sleep(1.5)
raw = cam.grab()
cam.close()

flat, touch = role.ref.apply(raw)
ys = np.where(role.ref.mask)[0]
print('\nmask rows %d..%d   opaque %.1f%%   touch %d'
      % (ys.min(), ys.max(), role.ref.masked_frac * 100, touch))

def runs(a, label):
    p = a[:, 1500:2500].mean(axis=1)
    b = np.percentile(p, 95)
    idx = np.flatnonzero(p < 0.5 * b)
    print('%s bright %.0f DN' % (label, b))
    if idx.size == 0:
        print('    NO DARK ROWS AT ALL')
        return
    s = idx[0]
    for x, y in zip(idx, idx[1:]):
        if y != x + 1:
            print('    dark rows %d..%d  (%d tall)' % (s, x, x - s + 1))
            s = y
    print('    dark rows %d..%d  (%d tall)' % (s, idx[-1], idx[-1] - s + 1))

runs(raw,  'RAW  ')
runs(flat, 'FLAT ')
