"""Reproduce all manuscript results (E1-E5). Outputs go to an explicit directory (default: results/)."""
from pathlib import Path
import json,time,platform,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy,sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score,confusion_matrix
from .simulation import *
HERE=Path(__file__).resolve().parent
OUT=Path('results'); FIG=OUT/'figures'; TAB=OUT/'tables'
def set_output_dir(path):
    """Direct results.json, figures/ and tables/ to `path` (created if absent)."""
    global OUT,FIG,TAB
    OUT=Path(path);FIG=OUT/'figures';TAB=OUT/'tables'
    for d in (OUT,FIG,TAB):d.mkdir(parents=True,exist_ok=True)
    return OUT
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
CHANNELS={'R0':(0,0,0),'H05':(.05,0,0),'H50':(.5,0,0),'V05':(0,0,.05),'V10':(0,0,.1),'O':(.5,0,.1),'Hminus':(-.5,0,0),'L5':None}
CONFIGS={'Single':[],'H 0.05':[1],'H 0.50':[2],'V 0.05':[3],'V 0.10':[4],'Oblique':[5],'H+V':[2,4],'H+H':[2,6],'L1-L5':[7]}
ERRORS=[(0,0),(0,.25),(.25,.25),(.5,.25),(1,.25),(.5,1)]
def truths(s):return np.column_stack([signal(s,b) if b is not None else signal(s,f=F_L5) for b in CHANNELS.values()])
def features(s,m,indices):return np.column_stack([np.rad2deg(s['el']),m[:,0]]+[np.abs(m[:,i]-m[:,0]) for i in indices])
def metrics(y,p):return dict(ba=float(balanced_accuracy_score(y,p)),f1=float(f1_score(y,p,average='macro')),recall=recall_score(y,p,average=None).tolist(),confusion=confusion_matrix(y,p).tolist())
def fit(s,m,ids):return DecisionTreeClassifier(max_depth=5,class_weight='balanced',random_state=42).fit(features(s,m,ids),s['y'])
def savefig(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=220);plt.close(fig)
def aggregate(rows):
    out={}
    for name in CONFIGS:
        for tag in sorted(set(r['tag'] for r in rows)):
            a=[r for r in rows if r['config']==name and r['tag']==tag]
            if not a:continue
            b=[next(r['ba'] for r in rows if r['config']=='Single' and r['tag']==tag and r['seed']==x['seed']) for x in a]
            gains=np.array([r['ba'] for r in a])-b
            out.setdefault(name,{})[tag]={'mean':float(np.mean([r['ba'] for r in a])),'sd':float(np.std([r['ba'] for r in a],ddof=1)), 'f1':float(np.mean([r['f1'] for r in a])),'gain_pp':float(100*gains.mean()),'gain_sd_pp':float(100*gains.std(ddof=1)), 'recall':np.mean([r['recall'] for r in a],axis=0).tolist()}
    return out

def run_all(n=12000,seeds=(0,1,2,3,4),output_dir=None):
    set_output_dir(output_dir or OUT);start=time.time();self_checks();R={'protocol':{'n_train':n,'n_test':n,'seeds':list(seeds),'errors':ERRORS,'class_counts_per_split':[n//4]*4,'channels':CHANNELS,'configs':CONFIGS},'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
    el=np.deg2rad(np.linspace(10,80,200));u=np.c_[np.cos(el)/np.sqrt(2),np.cos(el)/np.sqrt(2),np.sin(el)];g=np.array([0,0,1]);w=np.array([1,0,0])
    fig,ax=plt.subplots(figsize=(5,3))
    for b,normal,label in [((.5,0,0),g,'Ground / H 0.50 m'),((0,0,.1),g,'Ground / V 0.10 m'),((.5,0,0),w,'Wall / H 0.50 m')]:ax.plot(np.rad2deg(el),abs(phase_step(b,u,normal))/np.pi,label=label)
    ax.set(xlabel='Elevation (degrees)',ylabel=r'$|\Delta\phi|/\pi$');ax.legend(frameon=False);savefig(fig,'fig_dphi')
    s=make_scene(12000,900);lengths=[.02,.05,.1,.2,.5,1,2];R['E2']={}
    fig,axs=plt.subplots(1,2,figsize=(7,3),sharey=True)
    for ax,(orient,vec) in zip(axs, [('horizontal',np.array([1,0,0])),('vertical',np.array([0,0,1]))]):
        rows=[]
        for L in lengths:
            d=abs(signal(s,L*vec)-signal(s));rows.append([float(np.median(d[s['y']==i])) for i in range(4)])
        R['E2'][orient]=dict(lengths=lengths,medians=rows)
        for i in [1,2]:ax.plot(lengths,np.array(rows)[:,i],'-o',label=CLASSES[i])
        ax.axhline(.25,color='gray',ls='--',label='q = 0.25 dB (reference)');ax.set(xscale='log',xlabel=orient+' baseline (m)')
    axs[0].set_ylabel('Median absolute difference (dB)');axs[1].legend(frameon=False,fontsize=7);savefig(fig,'fig_delta_vs_baseline')
    rows=[];stress=[]
    for seed in seeds:
        train=make_scene(n,1000+seed);test=make_scene(n,2000+seed);a=truths(train);b=truths(test)
        for j,(sig,q) in enumerate(ERRORS):
            mt=measure(a,3000+100*seed+j,sig,q);mv=measure(b,4000+100*seed+j,sig,q)
            for name,ids in CONFIGS.items():
                clf=fit(train,mt,ids);res=metrics(test['y'],clf.predict(features(test,mv,ids)))
                rows.append(dict(seed=seed,tag=f'{sig}/{q}',config=name,**res))
        # Frozen nominal model; domain and receiver shifts never used for training/selection.
        mt=measure(a,5000+seed,.5,.25)
        shifted=make_scene(n,6000+seed,SceneConfig(height=.8,wall_min=5,wall_max=60,ground_eps=8,wall_eps=3,back_db=10,xpol_db=15,leakage_phase=np.pi/2,level_sd=2))
        mv=measure(b,7000+seed,.5,.25)
        bs=np.array([0,.3,.6,.2,.4,.7,-.4,2.]);sl=np.array([0,.2,.4,.3,.4,.6,-.2,.8])
        bias=bs+np.sin(test['el'])[:,None]*sl
        biased=measure(b,7000+seed,.5,.25,offsets=bias)
        cal=make_scene(2000,8000+seed);cal['y'][:]=0;ct=truths(cal)
        cm=measure(ct,9000+seed,.5,.25,offsets=bs+np.sin(cal['el'])[:,None]*sl)
        basis=np.c_[np.ones(len(cal['y'])),np.sin(cal['el'])];coef=np.linalg.lstsq(basis,cm-cm[:,[0]],rcond=None)[0]
        corrected=biased-np.c_[np.ones(n),np.sin(test['el'])]@coef
        cases={'nominal':(test,mv),'receiver_bias':(test,biased),'calibrated':(test,corrected),'correlated_noise':(test,measure(b,7000+seed,.5,.25,rho=.8)),'shifted_scene':(shifted,measure(truths(shifted),7000+seed,.5,.25))}
        for name,ids in CONFIGS.items():
            clf=fit(train,mt,ids)
            for tag,(sc,obs) in cases.items():stress.append(dict(seed=seed,tag=tag,config=name,**metrics(sc['y'],clf.predict(features(sc,obs,ids)))))
        print('Completed paired seed',seed,flush=True)
    R['E3_raw']=rows;R['E3']=aggregate(rows);R['E5_raw']=stress;R['E5']=aggregate(stress)
    fig,ax=plt.subplots(figsize=(7,4))
    for name in ['Single','H 0.50','V 0.10','Oblique','H+V','H+H','L1-L5']:
        dat=[R['E3'][name][f'{s}/{q}'] for s,q in ERRORS]
        ax.errorbar(range(len(ERRORS)),[100*d['mean'] for d in dat],yerr=[100*d['sd'] for d in dat],marker='o',ms=3,capsize=2,label=name)
    ax.set_xticks(range(len(ERRORS)),[f'{s}/{q}' for s,q in ERRORS]);ax.set(xlabel='Noise SD / quantization step (dB)',ylabel='Balanced accuracy (%)',ylim=(20,100));ax.legend(ncol=3,fontsize=8,frameon=False);savefig(fig,'fig_bal_acc')
    # Binary oracle: nonlinear two-ray model, unknown uniform phase, fixed known mu.
    alphas=[0,.01,.025,.05,.1,.2,.4];R['E4']=[]
    fig,ax=plt.subplots(figsize=(5,3))
    for q in [.25,1.]:
        ys=[]
        for alpha in alphas:
            e=float(bayes_error(alpha,np.pi,.5,q));D=float(rms_delta_db(alpha,np.pi)/np.sqrt(.5+q*q/6));ys.append(e)
            R['E4'].append(dict(alpha=alpha,q=q,sigma=.5,phase_step=float(np.pi),D=D,bayes_error=e))
        ax.plot(alphas,ys,'-o',label=f'q = {q} dB')
    ax.axhline(.5,color='gray',ls='--');ax.set(xlabel='Reflected/direct voltage ratio',ylabel='Equal-prior Bayes error',ylim=(0,.52));ax.legend(frameon=False);savefig(fig,'fig_bayes')
    # Convergence and independent Monte Carlo audit of difference PMF.
    e1=bayes_error(.1,np.pi,.5,.25,nphase=512);e2=bayes_error(.1,np.pi,.5,.25,nphase=2048)
    x,p0,p1=difference_pmf(.1,np.pi,.5,.25);rng=np.random.default_rng(12345);ph=rng.uniform(0,2*np.pi,200000)
    raw=40.125+10*np.log10(1+.1**2+2*.1*np.cos(np.c_[ph,ph+np.pi]));obs=measure(raw,777,.5,.25);diff=obs[:,1]-obs[:,0]
    counts=np.array([(diff==v).mean() for v in x]);tv=float(.5*abs(counts-p1).sum())
    assert abs(e1-e2)<1e-6 and tv<.02
    R['validation']={'phase_quadrature_difference':abs(e1-e2),'monte_carlo_pmf_total_variation':tv}
    # Weak-signal design tensor and equal-length geometric conditioning.
    nrm=np.eye(3);layouts={'horizontal_pair':np.array([[.5,0,0]]),'oblique_pair':np.array([[.5,0,.1]]),'horizontal_vertical':np.array([[.5,0,0],[0,0,.1]]),'three_axes':np.eye(3)*.1}
    R['design_rank']={k:int(np.linalg.matrix_rank(v)) for k,v in layouts.items()}
    R['runtime_s']=time.time()-start
    R['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'simulation.py',HERE/'run_experiments.py']}
    (OUT/'results.json').write_text(json.dumps(R,indent=2));make_tables(R)
    return R

def make_tables(R):
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Features & $0/0$ & $0/0.25$ & $0.25/0.25$ & $0.5/0.25$ & $1/0.25$ & $0.5/1$ \\',r'\midrule']
    for name in CONFIGS:
        vals=[R['E3'][name][f'{s}/{q}'] for s,q in ERRORS]
        lines.append(name.replace('+',r'$+$')+' & '+' & '.join(f"{100*v['mean']:.1f}" for v in vals)+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(TAB/'table_e3.tex').write_text('\n'.join(lines))
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Features & Nominal & Bias & Calibrated & $\rho=0.8$ & Scene shift \\',r'\midrule']
    for name in ['Single','H 0.50','V 0.10','Oblique','H+V','H+H','L1-L5']:
        vals=[R['E5'][name][tag]['mean'] for tag in ['nominal','receiver_bias','calibrated','correlated_noise','shifted_scene']]
        lines.append(name.replace('+',r'$+$')+' & '+' & '.join(f'{100*v:.1f}' for v in vals)+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(TAB/'table_e5.tex').write_text('\n'.join(lines))

    lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Baseline (m) & 0.02 & 0.05 & 0.10 & 0.20 & 0.50 & 1.00 & 2.00 \\',r'\midrule']
    for orient in ['horizontal','vertical']:
        lines.append(orient.title()+' & '+' & '.join(f'{v[1]:.3f}/{v[2]:.3f}'.replace('0.', '.') for v in R['E2'][orient]['medians'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(TAB/'table_e2.tex').write_text('\n'.join(lines))
    lines=[r'\begin{tabular}{lccc}',r'\toprule',r'Features & $0/0$ & $0/0.25$ & $0.5/0.25$ \\',r'\midrule']
    for name in ['Single','H 0.50','V 0.10','H+V','L1-L5']:
        vals=[' / '.join(f'{100*r:.0f}' for r in R['E3'][name][tag]['recall']) for tag in ['0/0','0/0.25','0.5/0.25']]
        lines.append(name.replace('+',r'$+$')+' & '+' & '.join(vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(TAB/'table_recall.tex').write_text('\n'.join(lines))

def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n',type=int,default=12000,help='scenes per split (multiple of 4)')
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    ap.add_argument('--out',default='results')
    a=ap.parse_args(argv);run_all(a.n,tuple(a.seeds),a.out)
if __name__=='__main__':main()
