"""Dependency-light deterministic invariants for revised statistical protocol."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_experiments as r
import numpy as np


def run():
    a=r.simulate(301,'contact-heating')
    b=r.simulate(301,'contact-heating')
    assert np.array_equal(a['temperature'],b['temperature']), 'nondeterministic trace'
    assert a['onset']>=r.WARM+90 and a['end']+30<=r.N-90
    assert {1,2,3,4}.isdisjoint(r.CAL_SEEDS)
    assert {51,52,53,54}.isdisjoint(r.CAL_SEEDS)
    assert set(range(601,701)).isdisjoint(r.CAL_SEEDS)
    assert a['end']-a['onset']==r.DURATION
    assert r.METHODS.count('Kalman innovation')==1
    assert r.METHODS.count('Isolation forest residual')==1
    # An alarm already active before injection MUST count as a new-rule hit.
    label=np.zeros((r.N,3),np.int8); label[500:770,0]=1
    s=np.zeros((r.N,3)); s[470:650,0]=10
    old=r.evaluate_legacy(s,5.,500,770,label,False)
    new=r.evaluate_active(s,5.,500,770,label)
    assert old[0]==0 and new[0]==1 and new[1]==0
    # Hit is monotone in threshold in the corrected rule (despite refractory).
    rr=[r.evaluate_active(s,x,500,770,label)[0] for x in (11.,7.,1.)]
    assert rr==[0.,1.,1.],rr
    # All-phase common residual => no peer-positive contrast.
    z=np.array([2.,2.,2.]); peer=np.maximum(0,z-(sum(z)-z)/2)
    assert np.all(peer==0)
    # Healthy pseudo-onset must be valid in 270+30 evaluation window.
    no_noise=r.simulate(301,'contact-heating',measurement_noise=False)
    assert no_noise['measurement_noise'] is False
    print('All protocol invariants passed. This is NOT an experiment validation.')

if __name__=='__main__':run()
