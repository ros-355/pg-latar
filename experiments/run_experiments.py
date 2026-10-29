#!/usr/bin/env python3
"""PG-LATAR reproduction / extended synthetic-only MCC anomaly benchmark.

No physical MCC measurements. Seed domains are deliberately disjoint:
 training [1,2,3,4], calibration [51,52,53,54], tests start 301,
 healthy evaluation starts 601 (100 sequences in the archived full run), shifted tests start 801. Never tune on test traces.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from scipy.stats import binomtest
from sklearn.ensemble import IsolationForest
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]; OUT=BASE/'results'; FIG=BASE/'figures'
for p in (OUT,FIG): p.mkdir(exist_ok=True,parents=True)
N=5760; WARM=180; DURATION=270; DT=1.; TARGETS=(.05,.1,.25,.5,1.0)
FAMILIES=['contact-heating','resistance-drift','shared ambient/enclosure heating','sensor-offset']
PHYSICAL=set(FAMILIES[:3]); DEFAULT_SEVERITY=7.
METHODS=['Fixed temperature','Ambient-corrected','Load regression','Load regression + accumulator',
         'Phase delta-T','Phase delta-T + accumulator','RC residual','PG-LATAR (no peer)','PG-LATAR',
         'PG-LATAR two-sided','RC multi-step rollout','PG-LATAR observer',
         'Kalman innovation','Isolation forest residual']

@njit(cache=True)
def lowpass(v,b):
    out=np.empty_like(v); out[0]=v[0]
    for k in range(1,len(v)):out[k]=(1-b)*out[k-1]+b*v[k]
    return out

@njit(cache=True)
def thermal_process(T0,ambient,current,tau,alpha,source,noise,structural,door):
    n=len(ambient); out=np.empty((n,3));out[0]=T0
    if structural:
        core=np.copy(T0)
        for t in range(1,n):
            for j in range(3):
                # two-node internal core and measured surface, with temperature-dependent R
                teq=ambient[t-1]+door[t-1]+alpha[j]*(1+.003*(core[j]-30))*current[t-1,j]**2+source[t-1,j]
                core[j]+= (teq-core[j])/(tau[j]*1.45)
                out[t,j]=out[t-1,j]+(core[j]-out[t-1,j])/(tau[j]*.37)+noise[t,j]
    else:
        for t in range(1,n):
            for j in range(3):
                teq=ambient[t-1]+alpha[j]*current[t-1,j]**2+source[t-1,j]
                out[t,j]=out[t-1,j]+(teq-out[t-1,j])/tau[j]+noise[t,j]
    return out


def simulate(seed:int,fault=None,shift=False,structural=False,severity=7.,measurement_noise=True):
    rng=np.random.default_rng(seed); t=np.arange(N); hour=t/60
    ambient=27+3.2*np.sin(2*np.pi*(hour-5)/24)+.7*np.sin(2*np.pi*hour/72)+lowpass(rng.normal(size=N),.025)
    demand=102+20*np.sin(2*np.pi*(hour-6)/24)+7*lowpass(rng.normal(size=N),.025)
    for start in range(0,N,360):demand[start:start+360]+=rng.normal(0,5)
    current=np.clip(demand[:,None]*np.array([.93,1.,1.08])+np.column_stack([lowpass(rng.normal(size=N),.12)*4 for _ in range(3)]),45,160)
    tau=np.array([34.,37.,40.]);alpha=np.array([.00170,.00158,.00152])
    if shift:
        tau*=rng.uniform(.88,1.12,3);alpha*=rng.uniform(.94,1.06,3)
    onset=int(rng.integers(WARM+90,N-DURATION-120)) if fault else -1; end=onset+DURATION if fault else -1
    phase=seed%3; source=np.zeros((N,3)); label=np.zeros((N,3),dtype=np.int8)
    if fault:
        ramp=np.minimum(1.,np.arange(DURATION)/(75 if fault!='resistance-drift' else 140))
        if fault=='contact-heating':source[onset:end,phase]=severity*ramp
        elif fault=='resistance-drift':source[onset:end,phase]=(.00070*severity/7)*current[onset:end,phase]**2*ramp
        elif fault=='shared ambient/enclosure heating':source[onset:end,:]=severity
        elif fault=='cooling-event':source[onset:end,phase]=-severity*ramp
        elif fault not in ('sensor-offset','negative-sensor-offset'):raise ValueError(fault)
        label[onset:end,:]=1 if fault=='shared ambient/enclosure heating' else label[onset:end,:]
        if fault!='shared ambient/enclosure heating':label[onset:end,phase]=1
    door=np.zeros(N)
    if structural:
        # ambient disturbance and door-opening/free-convection surrogate (non-fault nuisance)
        # door signal is NOT in measured ambient, challenging first-order assumed dynamics.
        for s in [930,2280,3890]:
            door[s:s+100]+=rng.choice([-1.,1.])*2.5
        ambient[2600:3000]+=2.5
    T0=ambient[0]+alpha*current[0]**2
    T=thermal_process(T0,ambient,current,tau,alpha,source,rng.normal(0,.055,size=(N,3)),structural,door)
    obs=T.copy()
    if measurement_noise:
        obs+=rng.normal(0,.25,size=(N,3))+rng.normal(0,.08,size=(N,1))
    if fault=='sensor-offset':obs[onset:end,phase]+=(severity/7)*np.minimum(7.,np.arange(DURATION)*.05)
    if fault=='negative-sensor-offset':obs[onset:end,phase]-=(severity/7)*np.minimum(7.,np.arange(DURATION)*.05)
    return {'temperature':obs,'current':current,'ambient':ambient+(rng.normal(0,.2,N) if measurement_noise else np.zeros(N)),
            'label':label,'fault':fault or 'healthy','onset':onset,'end':end,'seed':int(seed),
            'structural':structural,'shift':shift,'severity':severity,'measurement_noise':measurement_noise}

@njit(cache=True)
def rolling_prediction(T,I,Ta,dyn,observer=False,scales=None):
    if scales is None:scales=np.ones(3)
    n=len(T); pred=np.empty_like(T);pred[0]=T[0]
    quiet=np.zeros(3,dtype=np.int64)
    for k in range(1,n):
        for j in range(3):
            value=pred[k-1,j]+dyn[j,0]*(Ta[k-1]-pred[k-1,j])+dyn[j,1]*I[k-1,j]**2+dyn[j,2]
            if observer and k%360==0 and k>180 and quiet[j]>=30:
                value=T[k,j] # reset only after a window of small residuals
            pred[k,j]=value
            if observer:
                if abs(T[k,j]-value)<1.5*scales[j]:quiet[j]+=1
                else:quiet[j]=0
    return pred

@njit(cache=True)
def multirollout_residual(b, temperatures, currents, ambient, chunk=60):
    accum=np.empty(len(temperatures)*((N//chunk)-1)*(chunk-1), dtype=np.float64)
    at=0
    for q in range(len(temperatures)):
        t=temperatures[q];i=currents[q];a=ambient[q]
        for start in range(0,N-chunk,chunk):
            pr=t[start]
            for k in range(start+1,start+chunk):
                pr=pr+b[0]*(a[k-1]-pr)+b[1]*i[k-1]**2+b[2]
                accum[at]=pr-t[k]; at+=1
    return accum[:at]

def fit(train,with_multi=True):
    sta=[];dyn=[];mult=[]
    for j in range(3):
        x1=np.concatenate([np.column_stack((np.ones(N),d['ambient'],d['current'][:,j]**2)) for d in train]); y1=np.concatenate([d['temperature'][:,j] for d in train]);sta.append(np.linalg.lstsq(x1,y1,rcond=None)[0])
        x2=np.concatenate([np.column_stack((d['ambient'][:-1]-d['temperature'][:-1,j],d['current'][:-1,j]**2,np.ones(N-1))) for d in train]);y2=np.concatenate([np.diff(d['temperature'][:,j]) for d in train]);
        res=lsq_linear(x2,y2,bounds=([.008,0.,-np.inf],[.085,np.inf,np.inf]),max_iter=100)
        if not res.success: raise RuntimeError('constrained least squares failed')
        dyn.append(res.x)
        if with_multi:
            tt=np.array([d['temperature'][:,j] for d in train]);ii=np.array([d['current'][:,j] for d in train]); aa=np.array([d['ambient'] for d in train]);
            fit2=least_squares(multirollout_residual,res.x,args=(tt,ii,aa),bounds=([.008,0.,-np.inf],[.085,np.inf,np.inf]),max_nfev=45,ftol=1e-4,xtol=1e-4,gtol=1e-4)
            mult.append(fit2.x)
        else:mult.append(res.x)
    return np.array(sta),np.array(dyn),np.array(mult)

def raw_features(d,fit_coeff):
    sta,dyn,mult=fit_coeff;I=d['current'];Ta=d['ambient'];T=d['temperature']
    sta_pred=np.column_stack([sta[j,0]+sta[j,1]*Ta+sta[j,2]*I[:,j]**2 for j in range(3)])
    return {'Fixed temperature':T,'Ambient-corrected':T-Ta[:,None],'Load regression':T-sta_pred,
           'RC residual':T-rolling_prediction(T,I,Ta,dyn),'RC multi-step rollout':T-rolling_prediction(T,I,Ta,mult)}

def fit_state(train,with_multi=True):
    coeff=fit(train,with_multi=with_multi)
    res=np.concatenate([raw_features(d,coeff)['RC residual'][WARM:] for d in train]);sig=np.std(res,axis=0)
    static_res=np.concatenate([raw_features(d,coeff)["Load regression"][WARM:] for d in train]);static_sig=np.std(static_res,axis=0)
    return coeff,sig,static_sig

@njit(cache=True)
def accum(v,rho,k):
    n=len(v); y=np.zeros_like(v)
    for t in range(1,n):
        for j in range(3):y[t,j]=max(0.,rho*y[t-1,j]+v[t,j]-k)
    return y

def score_all(d,state,gamma=.45,rho=.95,k=1.1,floor=.12):
    coeff,sigma,static_sigma=state; raw=raw_features(d,coeff)
    T=d['temperature']; phase=(3*T-np.sum(T,axis=1)[:,None])/2
    z=np.maximum(0,raw['RC residual']/np.maximum(sigma,floor)[None,:]); peers=(np.sum(z,axis=1)[:,None]-z)/2
    peer=np.maximum(0,z-peers)
    scored={name:raw[name] for name in ('Fixed temperature','Ambient-corrected','Load regression','RC residual','RC multi-step rollout')}
    scored['Load regression + accumulator']=accum(np.maximum(0,raw['Load regression']/np.maximum(static_sigma,.12)),rho,k)
    scored['Phase delta-T']=phase;scored['Phase delta-T + accumulator']=accum(np.maximum(0,phase/3.),rho,k)
    scored['PG-LATAR (no peer)']=accum(z,rho,k)
    scored['PG-LATAR']=accum(z+gamma*peer,rho,k)
    z2=np.abs(raw['RC residual']/np.maximum(sigma,floor)[None,:]); peer2=np.maximum(0,z2-(np.sum(z2,axis=1)[:,None]-z2)/2)
    scored['PG-LATAR two-sided']=accum(z2+gamma*peer2,rho,k)
    observer=rolling_prediction(T,d['current'],d['ambient'],coeff[1],True,np.maximum(sigma,floor))
    zobs=np.maximum(0,(T-observer)/np.maximum(sigma,floor)[None,:]);peerobs=np.maximum(0,zobs-(np.sum(zobs,axis=1)[:,None]-zobs)/2)
    scored['PG-LATAR observer']=accum(zobs+gamma*peerobs,rho,k)
    return scored


# ---------------------------------------------------------------------------
# 2026-10-23 protocol revision. The original simulator, constrained fitting,
# accumulator, and seed domains above remain. Old onset-only score is retained
# explicitly to expose, not hide, the original non-monotonic recall pathology.
# ---------------------------------------------------------------------------

CAL_SEEDS = tuple(range(10001, 10151))
SHIFT_CAL_SEEDS = tuple(range(20001, 20151))
RECAL_BOOT_SEED = 447711
HYPERPARAMS = dict(gamma=.45, rho=.95, k=1.1, sigma_floor=.12)
# These values are pre-existing defaults, NOT optimized against held-out data.
# Historical influence of pilot seeds 601--606 cannot be independently proven.
ALL_PAIRS = (
    [('PG-LATAR',m) for m in METHODS if m!='PG-LATAR'] +
    [('Load regression',m) for m in METHODS if m not in ('PG-LATAR','Load regression')] +
    [('RC multi-step rollout',m) for m in ('Kalman innovation','Isolation forest residual')]
)

@njit(cache=True)
def kalman_innovations(T, I, Ta, dyn, q_var, r_var):
    """Genuine scalar KF predict/innovation/update separately for each phase.
    x^-_t=(1-b1)x_(t-1)+b1*Ta_(t-1)+b2*I^2+b3; P^- = a²P+Q;
    innov=(measured-predicted)/sqrt(P^-+R); K=P^-/(P^-+R).
    Q/R are estimated exclusively from 1--4 healthy training traces.
    """
    n=len(T); y=np.zeros_like(T)
    x=T[0].copy(); P=r_var.copy()
    for t in range(1,n):
        for j in range(3):
            a=1.-dyn[j,0]
            pre=a*x[j]+dyn[j,0]*Ta[t-1]+dyn[j,1]*I[t-1,j]**2+dyn[j,2]
            pp=a*a*P[j]+q_var[j]
            S=pp+r_var[j]
            y[t,j]=(T[t,j]-pre)/np.sqrt(S)
            gain=pp/S
            x[j]=pre+gain*(T[t,j]-pre)
            P[j]=(1.-gain)*pp
    return y


def train_extra(train,state):
    coeff,sigma,_=state
    dyn=coeff[1]
    residuals=[raw_features(d,coeff)['RC residual'] for d in train]
    innovation_e=[]
    for d in train:
        T=d['temperature']; I=d['current']; Ta=d['ambient']
        per=np.column_stack([T[:-1,j]+dyn[j,0]*(Ta[:-1]-T[:-1,j])+
                             dyn[j,1]*I[:-1,j]**2+dyn[j,2] for j in range(3)])
        innovation_e.append(T[1:]-per)
    raw=np.concatenate(innovation_e,axis=0)
    # Training-only componentwise decomposition of observed innovation variance.
    # An approximation rather than oracle access to generator noise variances.
    rv=np.maximum(.005,.5*np.var(np.diff(raw,axis=0),axis=0))
    qv=np.maximum(.0005,np.var(raw,axis=0)-rv*(1+(1-dyn[:,0])**2))
    X=[]
    for d,r in zip(train,residuals):
        norm=r[WARM:]/np.maximum(sigma,.12)
        X.append(np.column_stack([norm.ravel(),(d['current'][WARM:]/100).ravel(),
                   np.repeat(((d['ambient'][WARM:]-27)/5)[:,None],3,axis=1).ravel()]))
    X=np.concatenate(X,axis=0)
    iso=IsolationForest(n_estimators=48,max_samples=256,contamination='auto',
                        random_state=2139,n_jobs=1).fit(X)
    return {'Q':qv,'R':rv,'iso':iso,'sigma':sigma}


def score_extended(d,state,extra,**kwargs):
    sc=score_all(d,state,**kwargs)
    T=d['temperature']; I=d['current']; Ta=d['ambient']
    sc['Kalman innovation']=kalman_innovations(T,I,Ta,state[0][1],extra['Q'],extra['R'])
    residual=raw_features(d,state[0])['RC residual']/np.maximum(extra['sigma'],.12)
    X=np.column_stack([residual.ravel(),(I/100).ravel(),
                       np.repeat(((Ta-27)/5)[:,None],3,axis=1).ravel()])
    sc['Isolation forest residual']=(-extra['iso'].decision_function(X)).reshape(N,3)
    return sc

@njit(cache=True)
def evaluate_legacy(score,threshold,onset,end,label,healthy=False):
    """Original onset-only / per-phase refractory counting, kept unchanged."""
    fp=0; hit=0; delay=np.nan
    n=score.shape[0]
    for j in range(3):
        last=-50; prev=False
        for t in range(WARM,n):
            above=score[t,j]>threshold
            if above and not prev and t-last>=40:
                last=t
                if not healthy and onset<=t<end+30 and label[min(t,end-1),j]>0:
                    if hit==0 or t-onset<delay:hit=1;delay=float(t-onset)
                if healthy or t<onset-90 or t>=end+90:fp+=1
            prev=above
    clean=(n-WARM) if healthy else (n-WARM-(end-onset)-180)
    return float(hit),delay,float(fp),clean/1440.

@njit(cache=True)
def false_system_episodes(score,threshold,onset=-1,end=-1):
    """One OR-combined system alert: phase-overlapping alarms are one episode.
    A new system episode requires a low-to-high crossing; refractory=40 min.
    Faulted traces exclude [-90,+90] guard bands around the event.
    """
    last=-40; was=False; fp=0
    for t in range(WARM,score.shape[0]):
        active=False
        for j in range(score.shape[1]):
            if score[t,j]>threshold:
                active=True
                break
        if active and not was and t-last>=40:
            last=t
            if onset<0 or t<onset-90 or t>=end+90:
                fp+=1
        was=active
    return fp

@njit(cache=True)
def evaluate_active(score, threshold, onset,end,label):
    """Monotone-in-threshold EVENT recall, evaluated independently of episodes.
    Hit if any affected-phase alarm is ACTIVE at any sample within
    [onset, end+30), i.e., the 270+30-minute event-evaluation window.
    Pre-existing active alarms count as hits; independent healthy matched
    windows quantify chance hits. Delay is from onset and capped at 300.
    For healthy traces pass onset=end=-1. FP = system-level OR episodes.
    """
    n=score.shape[0]; clean=((n-WARM) if onset<0 else (n-WARM-(end-onset)-180))/1440.
    fp=false_system_episodes(score,threshold,onset,end)
    if onset<0:
        return (np.nan,np.nan,float(fp),clean,np.nan)
    tt=-1
    for t in range(onset,min(n,end+30)):
        for j in range(score.shape[1]):
            if label[min(t,end-1),j]>0 and score[t,j]>threshold:
                tt=t;break
        if tt>=0:break
    hit=1. if tt>=0 else 0.
    d=float(tt-onset) if tt>=0 else np.nan
    return hit,d,float(fp),clean,(d if tt>=0 else 300.)


def evaluate_thresholds(d,sc,thr,old_rule=False):
    if old_rule:
        return {m:evaluate_legacy(sc[m],thr[m],d['onset'],d['end'],d['label'],d['fault']=='healthy') for m in METHODS}
    return {m:evaluate_active(sc[m],thr[m],d['onset'],d['end'],d['label']) for m in METHODS}


def summarize_method(arr, n_boot=2000, seed=1967, old_rule=False):
    a=np.asarray(arr,float); n=len(a)
    if a.ndim!=2:return {}
    event_mask=np.isfinite(a[:,0]); ne=int(np.sum(event_mask)); hits=int(np.nansum(a[:,0]))
    d=a[:,1]; detected=d[np.isfinite(d)]
    false_rate=float(a[:,2].sum()/a[:,3].sum())
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,n,size=(n_boot,n)) if n_boot else np.empty((0,n),dtype=int)
    if n_boot:
        rates=np.sum(a[idx,2],axis=1)/np.sum(a[idx,3],axis=1)
        rate_ci=np.quantile(rates,[.025,.975]).tolist()
    else:rate_ci=[None,None]
    if old_rule:
        return {'n':ne,'hits':hits,'recall':hits/ne if ne else None,
                'wilson95':wilson(hits,ne) if ne else [None,None],
                'fp_episodes_per_system_day':false_rate,'fp_bootstrap95':rate_ci,
                'detected_delay_mean':float(np.mean(detected)) if len(detected) else None}
    rmst=float(np.nanmean(a[:,4])) if ne else None
    med=float(np.median(detected)) if len(detected) else None
    if n_boot and ne:
        b=a[idx]; rmsts=np.nanmean(b[:,:,4],axis=1)
        # Avoid empty detected-only bootstrap samples without warning spam.
        median_samples=np.array([np.median(row[np.isfinite(row)]) if np.isfinite(row).any()
                                 else np.nan for row in b[:,:,1]])
        rm_ci=np.nanquantile(rmsts,[.025,.975]).tolist()
        med_ci=np.nanquantile(median_samples,[.025,.975]).tolist() if np.isfinite(median_samples).any() else [None,None]
    else:rm_ci=[None,None];med_ci=[None,None]
    return {'n':ne,'hits':hits,'recall':hits/ne if ne else None,
            'wilson95':wilson(hits,ne) if ne else [None,None],
            'fp_episodes_per_system_day':false_rate,'fp_bootstrap95':rate_ci,
            'median_delay_detected_min':med,'median_delay_bootstrap95':med_ci,
            'restricted_mean_ttd_300_min':rmst,'rmtd_bootstrap95':rm_ci,
            'censoring_minutes':300,'episode_unit':'system (OR of 3 phases)'}


def wilson(hits,n):
    if n==0:return [None,None]
    z=1.959963984540054;p=hits/n; d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0.,mid-half)),float(min(1.,mid+half))]


def savecsv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def simulate_scores(seeds,faults,state,extra,**kwargs):
    # Seeds and family assignments come from caller; generator never consults tests.
    out=[]
    for sd,ft in zip(seeds,faults):
        d=simulate(int(sd),ft,**kwargs)
        out.append((d,score_extended(d,state,extra)))
    return out


def make_calibration(seeds,state,extra,shift=False):
    traces=[]
    for sd in seeds:
        d=simulate(sd,shift=shift)
        traces.append(score_extended(d,state,extra))
    return traces


def grid_for(method, scores, n_grid=100):
    """Score quantile grid; avoid incorrectly assuming FP episodes monotone.
    Includes very high quantiles to accommodate 0.05/system-day budgets.
    """
    flat=np.concatenate([s[method][WARM:].ravel() for s in scores])
    quant=np.r_[np.linspace(.65,.99,30),np.linspace(.9905,.999,30),
                np.linspace(.99901,.9999999,max(50,n_grid-60))]
    values=np.unique(np.r_[np.min(flat)-1e-7,np.quantile(flat,quant),np.max(flat)+1e-7])
    return values


def calibrate_method(method,cal_scores,targets=TARGETS,grid_size=100):
    thresholds=grid_for(method,cal_scores,grid_size)
    # Matrix [healthy calibration sequence, threshold].
    counts=np.array([[false_system_episodes(sc[method],float(th)) for th in thresholds]
                     for sc in cal_scores],dtype=np.int32)
    days=len(cal_scores)*(N-WARM)/1440.
    rates=np.sum(counts,axis=0)/days
    decision={}
    for target in targets:
        # Conservative: among observed rates <= target, choose the closest.
        eligible=np.flatnonzero(rates<=target+1e-12)
        if len(eligible):k=int(eligible[np.argmin(target-rates[eligible])])
        else:k=int(np.argmin(rates))
        decision[str(target)]={'threshold':float(thresholds[k]),
                               'calibration_fp_day':float(rates[k]),
                               'target_fp_day':float(target),'index':k}
    return thresholds,counts,decision


def cal_uncertainty(thresholds,counts,targets=TARGETS,repetitions=200,seed=RECAL_BOOT_SEED):
    rng=np.random.default_rng(seed);n=counts.shape[0];days=(N-WARM)/1440
    res=[]
    for target in targets:
        vals=[];rates=[]
        for _ in range(repetitions):
            which=rng.integers(0,n,n)
            rr=counts[which].sum(axis=0)/(n*days)
            valid=np.flatnonzero(rr<=target+1e-12)
            k=int(valid[np.argmin(target-rr[valid])]) if len(valid) else int(np.argmin(rr))
            vals.append(float(thresholds[k]));rates.append(float(rr[k]))
        res.append({'target':target,'threshold_p025':float(np.quantile(vals,.025)),
                    'threshold_median':float(np.median(vals)),
                    'threshold_p975':float(np.quantile(vals,.975)),
                    'rate_p025':float(np.quantile(rates,.025)),
                    'rate_p975':float(np.quantile(rates,.975)),
                    'resamples':repetitions})
    return res


def calibration_propagation(cal_tables, nominal_pairs, repeats=200, targets=(.05,.1,.25,.5,1.)):
    """Carry calibration bootstrap threshold variability onto the SAME fixed test
    traces. Outputs are calibration-induced variation conditional on this test
    cohort, NOT combined calibration/test uncertainty. No threshold is selected
    based on test labels; all grid locations come from healthy calibration.
    """
    rows=[]
    for m,(thresholds,counts,decisions) in cal_tables.items():
        rng=np.random.default_rng(RECAL_BOOT_SEED)
        n=len(counts); exposure=(N-WARM)/1440.
        indices={}
        for target in targets:
            ids=[]
            for _ in range(repeats):
                sample=rng.integers(0,n,n)
                rr=np.sum(counts[sample],axis=0)/(n*exposure)
                valid=np.flatnonzero(rr<=target+1e-12)
                k=int(valid[np.argmin(target-rr[valid])]) if len(valid) else int(np.argmin(rr))
                ids.append(k)
            indices[target]=np.array(ids,dtype=int)
        required=sorted(set(int(k) for seq in indices.values() for k in seq))
        # Compute each held-out evaluation at a particular calibration-derived
        # threshold exactly once, then look up replicated calibration choices.
        evaluated={}
        for j in required:
            observations=[evaluate_active(sc[m],float(thresholds[j]),d['onset'],d['end'],d['label'])
                          for d,sc in nominal_pairs]
            a=np.array(observations)
            evaluated[j]=(float(np.nanmean(a[:,0])),float(np.sum(a[:,2])/np.sum(a[:,3])),
                          float(np.nanmean(a[:,4])))
        for target,ids in indices.items():
            sample=np.array([evaluated[int(k)] for k in ids],dtype=float)
            rows.append({'method':m,'target':target,'calibration_resamples':repeats,
              'recall_median_conditional':float(np.median(sample[:,0])),
              'recall_p025_conditional':float(np.quantile(sample[:,0],.025)),
              'recall_p975_conditional':float(np.quantile(sample[:,0],.975)),
              'fp_median_conditional':float(np.median(sample[:,1])),
              'fp_p025_conditional':float(np.quantile(sample[:,1],.025)),
              'fp_p975_conditional':float(np.quantile(sample[:,1],.975)),
              'rmtd_median_conditional':float(np.median(sample[:,2])),
              'rmtd_p025_conditional':float(np.quantile(sample[:,2],.025)),
              'rmtd_p975_conditional':float(np.quantile(sample[:,2],.975)),
              'interpretation':'calibration resampling, test sequences fixed'})
    return rows


def make_figures(folder):
    """Use only outputs produced by a FULL re-run; never pilot/smoke metrics."""
    import pandas as pd
    summary=pd.read_csv(folder/'metrics_summary.csv')
    sev=pd.read_csv(folder/'severity_per_family_wilson.csv')
    curve=pd.read_csv(folder/'threshold_curves.csv')
    chosen=['Load regression','RC multi-step rollout','PG-LATAR (no peer)',
            'PG-LATAR','Kalman innovation','Isolation forest residual']
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for m in chosen:
        part=curve[curve.method==m].sort_values('fp_episodes_per_system_day')
        ax[0].plot(part.fp_episodes_per_system_day,part.recall,'.-',lw=.8,ms=2,label=m)
        ax[1].plot(part.fp_episodes_per_system_day,part.rmtd_300_min,'.-',lw=.8,ms=2,label=m)
    for m in chosen:
        pts=summary[(summary.setting=='nominal')&(summary.calibration_target.isin([.1,.5,1.]))&(summary.method==m)]
        for _,r in pts.iterrows():
            if pd.notna(r.recall):
                # Uncertainty bounds are stored as textual numeric pairs.
                import ast
                ci=ast.literal_eval(r.wilson95); fi=ast.literal_eval(r.fp_bootstrap95)
                x=r.fp_episodes_per_system_day;y=r.recall
                ax[0].errorbar(x,y,xerr=[[max(0,x-fi[0])],[max(0,fi[1]-x)]],
                          yerr=[[max(0,y-ci[0])],[max(0,ci[1]-y)]],fmt='o',ms=2,capsize=1)
    for a in ax:a.set_xlabel('System alarm episodes / clean system-day');a.grid(alpha=.2)
    ax[0].set_ylabel('Event recall');ax[1].set_ylabel('Restricted mean time to detection (min; 300-min cap)')
    ax[1].legend(fontsize=6)
    fig.savefig(folder/'calibrated_operating_curves.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for panel,family in zip(axs.ravel(),FAMILIES):
        for method in chosen:
            d=sev[(sev.family==family)&(sev.calibration_target==.5)&
                   (sev.method==method)&(sev.severity_C_equivalent<=2)].sort_values('severity_C_equivalent')
            if len(d):
                import ast
                yy=d.recall.to_numpy(dtype=float); x=d.severity_C_equivalent.to_numpy(dtype=float)
                cis=np.array([ast.literal_eval(z) for z in d.wilson95])
                panel.errorbar(x,yy,yerr=np.vstack([yy-cis[:,0],cis[:,1]-yy]),marker='o',ms=3,lw=1,capsize=2,label=method)
        panel.set_title(family);panel.set_xlabel('Injected C-equivalent magnitude');panel.set_ylabel('Recall (Wilson 95% CI)');panel.set_ylim(-.03,1.04);panel.grid(alpha=.2)
    axs[0,0].legend(fontsize=6)
    fig.savefig(folder/'low_severity_wilson_figure.pdf');plt.close(fig)


def paired_statistics(name,arrays,thresholds,n_boot=2000,seed=91713):
    """Paired at identical sequences; exact McNemar = binomial discordance test.
    Bootstrap comparisons over whole sequences, NOT individual minutes.
    """
    rng=np.random.default_rng(seed)
    rows=[]
    for a,b in ALL_PAIRS:
        if not arrays or a not in arrays[0]:continue
        arr_a=np.array([r[a] for r in arrays],float)
        arr_b=np.array([r[b] for r in arrays],float)
        mask=np.isfinite(arr_a[:,0]) & np.isfinite(arr_b[:,0])
        x=arr_a[mask];y=arr_b[mask]
        n10=int(np.sum((x[:,0]==1)&(y[:,0]==0)))
        n01=int(np.sum((x[:,0]==0)&(y[:,0]==1)))
        p=float(binomtest(min(n10,n01),n10+n01,p=.5).pvalue) if n10+n01 else 1.
        n=len(x)
        if not n:continue
        take=rng.integers(0,n,(n_boot,n))
        recall_delta=(x[take,0]-y[take,0]).mean(axis=1)
        fp_delta=np.sum(x[take,2],axis=1)/np.sum(x[take,3],axis=1)-np.sum(y[take,2],axis=1)/np.sum(y[take,3],axis=1)
        rm_delta=np.mean(x[take,4]-y[take,4],axis=1)
        rows.append({'setting':name,'A':a,'B':b,'mcnemar_n_Aonly':n10,'mcnemar_n_Bonly':n01,
                     'mcnemar_exact_p':p,'recall_diff_AminusB':float(np.mean(x[:,0]-y[:,0])),
                     'recall_delta_bootstrap95':np.quantile(recall_delta,[.025,.975]).tolist(),
                     'fp_diff_AminusB':float(x[:,2].sum()/x[:,3].sum()-y[:,2].sum()/y[:,3].sum()),
                     'fp_delta_bootstrap95':np.quantile(fp_delta,[.025,.975]).tolist(),
                     'rmtd_diff_AminusB':float(np.mean(x[:,4]-y[:,4])),
                     'rmtd_delta_bootstrap95':np.quantile(rm_delta,[.025,.975]).tolist(),
                     'sequences':n,'bootstrap_sequences':n_boot})
    return rows


def chance_rate(healthy_score_pairs,thresholds,seed=742):
    """Healthy controls with pseudo-onsets and masks matching each family.
    Identical pseudo windows are reused across detector methods and budgets,
    enabling fair paired chance-hit comparisons without test-label tuning.
    """
    rng=np.random.default_rng(seed); rows=[]
    windows={}
    for family_index,family in enumerate(FAMILIES):
        windows[family]=[]
        for d,sc in healthy_score_pairs:
            onset=int(rng.integers(WARM+90,N-DURATION-120))
            affected=(0,1,2) if family=='shared ambient/enclosure heating' else (int(d['seed']%3),)
            windows[family].append((onset,affected))
    for target,limits in thresholds.items():
        for family in FAMILIES:
            for method in METHODS:
                hits=0
                for (d,sc),(onset,affected) in zip(healthy_score_pairs,windows[family]):
                    hits+=int(np.any(sc[method][onset:onset+DURATION+30,list(affected)]>limits[method]))
                rows.append({'target':target,'family':family,'method':method,
                             'n_pseudo_events':len(healthy_score_pairs),
                             'healthy_chance_hit':hits,
                             'healthy_chance_hit_rate':hits/len(healthy_score_pairs),
                             'wilson95':wilson(hits,len(healthy_score_pairs))})
    return rows

def robustness_table(rows):
    """Consolidated Table IV candidates from produced outputs, no estimates."""
    by={(r['setting'],r.get('calibration_target'),r['method']):r for r in rows}
    settings=['nominal','parameter_shift','shifted_recalibrated','structural_mismatch']
    result=[]
    for method in METHODS:
        line={'method':method,'target':.5}
        for setting in settings:
            q=by.get((setting,.5,method),{})
            line[setting+'_recall']=q.get('recall')
            line[setting+'_fp_system_day']=q.get('fp_episodes_per_system_day')
            line[setting+'_rmtd_300']=q.get('restricted_mean_ttd_300_min')
        result.append(line)
    return result


def holm_adjust(rows):
    """Holm correction of predeclared paired recall contrasts per setting."""
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['setting'],[]).append(i)
    for ids in groups.values():
        order=sorted(ids,key=lambda i:rows[i]['mcnemar_exact_p'])
        previous=0.
        for position,j in enumerate(order):
            adj=min(1.,(len(order)-position)*rows[j]['mcnemar_exact_p'])
            previous=max(previous,adj)
            rows[j]['mcnemar_holm_p']=previous
    return rows

def outcome_rows(name,pairs,thr,methods=METHODS,old=False,bootstrap=2000,seed=3321):
    ev=[evaluate_thresholds(d,sc,thr,old) for d,sc in pairs]
    rows=[]
    for m in methods:
        sm=summarize_method([v[m] for v in ev],bootstrap,seed,old)
        rows.append({'setting':name,'rule':'legacy_onset' if old else 'active_270plus30',
                     'method':m,**sm})
    return rows,ev


def threshold_sweep(name,pairs,cal_tables,reference_targets=(.05,.1,.25,.5,1.),bootstrap=0):
    rows=[]
    for m in METHODS:
        thresholds=cal_tables[m][0]
        # Use fixed calibration-derived thresholds; no fitting to test labels.
        sel=np.unique(np.r_[np.linspace(0,len(thresholds)-1,30,dtype=int),
                               [cal_tables[m][2][str(k)]['index'] for k in reference_targets]])
        for id in sel:
            th=float(thresholds[id]); observations=[evaluate_active(sc[m],th,d['onset'],d['end'],d['label']) for d,sc in pairs]
            r=summarize_method(observations,bootstrap)
            rows.append({'setting':name,'method':m,'threshold':th,
                         'fp_episodes_per_system_day':r['fp_episodes_per_system_day'],
                         'recall':r['recall'],'rmtd_300_min':r['restricted_mean_ttd_300_min'],
                         'median_delay_detected':r['median_delay_detected_min']})
    return rows


def realized_interpolation(rows, budgets=TARGETS):
    """Descriptive post-hoc curve interpolation. NOT deployable threshold choice.
    It uses held-out FP on the x-axis; do not report as calibrated performance.
    """
    by={}
    for r in rows:by.setdefault((r['setting'],r['method']),[]).append(r)
    out=[]
    for (setting,method),rr in by.items():
        rr=sorted(rr,key=lambda x:x['fp_episodes_per_system_day'])
        xx=np.array([v['fp_episodes_per_system_day'] for v in rr])
        for b in budgets:
            if b < xx.min() or b > xx.max():
                out.append({'setting':setting,'method':method,'realized_target':b,
                            'interpolated_recall':None,'interpolated_rmtd':None,
                            'note':'outside measured curve range'})
                continue
            rec=np.array([v['recall'] for v in rr]); rm=np.array([v['rmtd_300_min'] for v in rr])
            out.append({'setting':setting,'method':method,'realized_target':b,
                        'interpolated_recall':float(np.interp(b,xx,rec)),
                        'interpolated_rmtd':float(np.interp(b,xx,rm)),
                        'note':'descriptive only: post-hoc interpolated held-out curve'})
    return out


def save_experiment(path,content):
    with open(path,'w') as f:json.dump(content,f,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))


def full_main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    methods=METHODS
    print('Training on seeds 1--4, fixed legacy hyperparameters',flush=True)
    train=[simulate(s) for s in (1,2,3,4)]
    state=fit_state(train,with_multi=True)
    extra=train_extra(train,state)
    cal_seeds=CAL_SEEDS[:args.calibration_n]
    cal=make_calibration(cal_seeds,state,extra)
    cal_tables={};cal_unc=[];cal_rows=[];thresholds={str(t):{} for t in TARGETS}
    for m in METHODS:
        th,cnt,dec=calibrate_method(m,cal)
        cal_tables[m]=(th,cnt,dec)
        for target in TARGETS:
            q=dec[str(target)]
            thresholds[str(target)][m]=q['threshold']
            cal_rows.append({'method':m,**q})
        for r in cal_uncertainty(th,cnt,repetitions=args.calibration_resamples):
            cal_unc.append({'method':m,**r})
    savecsv(out/'calibration_thresholds.csv',cal_rows)
    savecsv(out/'calibration_uncertainty.csv',cal_unc)
    print('Calibration on',len(cal),'new nonoverlapping healthy sequences complete',flush=True)
    legacy_cal=[score_extended(simulate(s),state,extra) for s in (51,52,53,54)]
    thresholds_999={m:float(np.quantile(np.concatenate([s[m][WARM:].ravel() for s in legacy_cal]),.999)) for m in METHODS}
    regimes=[('nominal',301,args.n,False,False),('parameter_shift',801,args.shifted_n,True,False),
             ('structural_mismatch',300001,args.structural_n,False,True)]
    summary_rows=[];family_rows=[];paired_rows=[];curves=[];outcomes_legacy=[];chance=[]
    for name,base,n,shift,structural in regimes:
        seeds=[int(base+20*j+1000*i) for j in range(4) for i in range(n)]
        faults=[FAMILIES[j] for j in range(4) for i in range(n)]
        pairs=simulate_scores(seeds,faults,state,extra,shift=shift,structural=structural)
        if name=='nominal':
            old,old_ev=outcome_rows(name,pairs,thresholds_999,old=True,bootstrap=args.bootstrap)
            new,new_ev=outcome_rows(name,pairs,thresholds_999,bootstrap=args.bootstrap)
            outcomes_legacy.extend(old+new)
        for target in TARGETS:
            rows,ev=outcome_rows(name,pairs,thresholds[str(target)],bootstrap=args.bootstrap)
            for r in rows:r['calibration_target']=target
            summary_rows+=rows
            if target in (.1,.5,1.):
                paired_rows+=paired_statistics(name+'_'+str(target),ev,thresholds[str(target)],args.bootstrap)
            if target==.5:
                for j,family in enumerate(FAMILIES):
                    for m in METHODS:
                        mat=[v[m] for v in ev[j*n:(j+1)*n]]
                        fam=summarize_method(mat,args.bootstrap,seed=400+j)
                        family_rows.append({'setting':name,'family':family,
                                             'category':'sensor artifact' if family=='sensor-offset' else 'physical heating',
                                             'method':m,**fam})
        if name=='nominal':
            curves=threshold_sweep(name,pairs,cal_tables)
            calprop=calibration_propagation(cal_tables,pairs,args.calibration_resamples)
            savecsv(out/'calibration_propagated_test_uncertainty.csv',calprop)
        print('Finished',name,len(pairs),'event sequences',flush=True)
        del pairs
    savecsv(out/'metrics_summary.csv',summary_rows)
    savecsv(out/'per_family_metrics.csv',family_rows)
    savecsv(out/'mcnemar_paired_bootstrap.csv',holm_adjust(paired_rows))
    savecsv(out/'legacy_vs_corrected.csv',outcomes_legacy)
    savecsv(out/'threshold_curves.csv',curves)
    savecsv(out/'realized_fp_interpolation_DESCRIPTIVE_ONLY.csv',realized_interpolation(curves))
    healthy_pairs=simulate_scores(range(601,601+args.healthy_n),[None]*args.healthy_n,state,extra)
    for target in TARGETS:
        rows,_=outcome_rows('healthy_only',healthy_pairs,thresholds[str(target)],bootstrap=args.bootstrap)
        for r in rows:r['calibration_target']=target
        summary_rows+=rows
    chance=chance_rate(healthy_pairs,thresholds)
    savecsv(out/'healthy_chance_hits.csv',chance)
    savecsv(out/'metrics_summary.csv',summary_rows)
    print('Healthy-only chance hits and FP complete',flush=True)
    del healthy_pairs
    # Shifted-domain calibration is NEW and disjoint; compare both domains.
    shifted_cal=make_calibration(SHIFT_CAL_SEEDS[:args.calibration_n],state,extra,shift=True)
    shifted_th={}
    for m in METHODS:
        _,_,dec=calibrate_method(m,shifted_cal)
        shifted_th[m]=dec['0.5']['threshold']
    shift_pairs=simulate_scores(
        [801+20*j+1000*i for j in range(4) for i in range(args.shifted_n)],
        [FAMILIES[j] for j in range(4) for _ in range(args.shifted_n)],
        state,extra,shift=True)
    shifted_rows,shift_ev=outcome_rows('shifted_recalibrated',shift_pairs,shifted_th,bootstrap=args.bootstrap)
    for r in shifted_rows:r['calibration_target']=.5
    summary_rows+=shifted_rows
    savecsv(out/'metrics_summary.csv',summary_rows)
    savecsv(out/'robustness_table_IV_candidates.csv',robustness_table(summary_rows))
    # New clean/unclean measurement-noise ablation, same seeds/operating cases.
    noise_rows=[]
    for noise in (True,False):
        part=simulate_scores([400001+20*j+1000*i for j in range(4) for i in range(args.noise_n)],
               [FAMILIES[j] for j in range(4) for _ in range(args.noise_n)],state,extra,measurement_noise=noise)
        for m in METHODS:
            r=summarize_method([evaluate_active(sc[m],thresholds['0.5'][m],d['onset'],d['end'],d['label']) for d,sc in part],args.bootstrap)
            noise_rows.append({'measurement_noise':noise,'method':m,**r})
        del part
    savecsv(out/'measurement_noise_ablation.csv',noise_rows)
    # Paired severity experiments >= 100 events/family/level, independent of the other cohorts.
    severity_rows=[]
    for size_idx,size in enumerate((.5,1.,2.,4.,7.,10.)):
        sd=[500001+100000*size_idx+20*j+1000*i for j in range(4) for i in range(args.severity_n)]
        ff=[FAMILIES[j] for j in range(4) for _ in range(args.severity_n)]
        items=simulate_scores(sd,ff,state,extra,severity=size)
        for target in (.1,.5):
            lim=thresholds[str(target)]
            for j,family in enumerate(FAMILIES):
                sub=items[j*args.severity_n:(j+1)*args.severity_n]
                for m in METHODS:
                    v=[evaluate_active(sc[m],lim[m],d['onset'],d['end'],d['label']) for d,sc in sub]
                    stats=summarize_method(v,args.bootstrap)
                    severity_rows.append({'severity_C_equivalent':size,'family':family,
                                          'calibration_target':target,'method':m,**stats})
        del items
        print('Severity',size,'complete',flush=True)
    savecsv(out/'severity_per_family_wilson.csv',severity_rows)
    savecsv(out/'main_event_amplitude_distribution.csv',[{'cohort':'nominal_main',
       'family':f,'magnitude_C_equiv':DEFAULT_SEVERITY,'sequences':args.n} for f in FAMILIES])
    # Explicitly separate negative artifacts / cooling surrogates from positive heating.
    negative=[]
    for j,ff in enumerate(('cooling-event','negative-sensor-offset')):
        pairs=simulate_scores([1501+20*j+1000*i for i in range(args.n)],
                              [ff]*args.n,state,extra)
        for m in METHODS:
            v=[evaluate_active(sc[m],thresholds['0.5'][m],d['onset'],d['end'],d['label']) for d,sc in pairs]
            negative.append({'family':ff,'method':m,**summarize_method(v,args.bootstrap)})
    savecsv(out/'negative_extension.csv',negative)
    save_experiment(out/'run_manifest.json',{
        'title':'PG-LATAR corrected simulation-only statistical benchmark',
        'seed_partitions':{'training':[1,2,3,4],'legacy_calibration':[51,52,53,54],
          'new_healthy_calibration':[min(cal_seeds),max(cal_seeds)],
          'shifted_healthy_calibration':[min(SHIFT_CAL_SEEDS),max(SHIFT_CAL_SEEDS[:args.calibration_n])],
          'nominal_test_starts':[301,321,341,361], 'healthy_evaluation':[601,601+args.healthy_n-1],
          'shifted_test_starts':[801,821,841,861],
          'structural_test_starts':[300001,300021,300041,300061]},
        'hyperparameters_frozen_before_run':HYPERPARAMS,
        'hyperparameter_provenance':'pre-existing draft defaults; original historical selection unverified',
        'method_count':len(METHODS),'methods':METHODS,
        'generator':{'N_minutes':N,'warmup_minutes':WARM,'event_minutes':DURATION,
          'process_noise_sigma':.055,'per_phase_observation_noise_sigma':.25,
          'common_observation_noise_sigma':.08,'ambient_sensor_sigma':.2,
          'base_tau_minutes':[34,37,40],'base_alpha':[.00170,.00158,.00152],
          'tau_shift_multiplier_uniform':[.88,1.12],'alpha_shift_multiplier_uniform':[.94,1.06],
          'structural_core_tau_multiplier':1.45,'structural_surface_tau_multiplier':.37,
          'structural_temp_coefficient_per_C':.003,'door_steps_minute':[930,2280,3890],
          'door_step_duration_min':100,'door_step_degC_magnitude':2.5,
          'ambient_step_minutes':[2600,3000],'ambient_step_degC':2.5,
          'phase_current_factors':[.93,1.,1.08],
          'load_profile':'102 + 20*sin(2*pi*(hour-6)/24) + 7*lowpass(white noise,0.025) + six-hour demand offsets; clip current to [45,160]',
          'ambient_profile':'27 + 3.2*sin(2*pi*(hour-5)/24) + 0.7*sin(2*pi*hour/72) + lowpass(white noise,0.025)',
          'load_noise':'per-phase lowpass(white noise,0.12)*4',
          'resistance_drift_source':'(0.00070 * severity/7) * current^2 * ramp',
          'nominal_predictor_state':'recursive predicted state initialized from first measured temperature, without subsequent reinitialization; observer comparator differs'},
        'evaluation':{'system_episode_refractory_minutes':40,'window_minutes':300,
          'hit_rule':'any active alarm on affected phase within event + 30 min',
          'fp':'system OR episodes per system-day, 90-minute fault guard',
          'posthoc_equal_realized_fp':'descriptive interpolated held-out curves, not threshold selection'},
        'bootstrap':args.bootstrap,'calibration_resamples':args.calibration_resamples,
        'args':vars(args)})
    make_figures(out)
    print('Protocol revision complete. Do NOT update manuscript before reviewing full re-run outputs.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default=str(BASE/'revised_results'))
    ap.add_argument('--n',type=int,default=100)
    ap.add_argument('--healthy-n',type=int,default=100)
    ap.add_argument('--shifted-n',type=int,default=100)
    ap.add_argument('--structural-n',type=int,default=100)
    ap.add_argument('--calibration-n',type=int,default=150)
    ap.add_argument('--severity-n',type=int,default=100)
    ap.add_argument('--noise-n',type=int,default=100)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--calibration-resamples',type=int,default=200)
    ap.add_argument('--smoke',action='store_true',help='Tiny plumbing check only, never manuscript evidence.')
    args=ap.parse_args()
    if args.smoke:
        args.n=2;args.healthy_n=3;args.shifted_n=2;args.structural_n=2
        args.calibration_n=4;args.severity_n=2;args.noise_n=2
        args.bootstrap=12;args.calibration_resamples=5
    if min(args.n,args.healthy_n,args.calibration_n,args.severity_n,args.noise_n)<=0:raise ValueError('All cohort counts must be positive')
    full_main(args)
