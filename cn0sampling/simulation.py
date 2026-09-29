"""Vectorized, relative-phase GNSS simulator. Synthetic sensitivity model, not a receiver emulator."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
C0=299792458.; F_L1=1575.42e6; F_L5=1176.45e6; LAM_L1=C0/F_L1; DB=20/np.log(10)
CLASSES=['LOS','MPG','MPB','NLOS']
@dataclass(frozen=True)
class SceneConfig:
    height:float=1.5
    wall_min:float=3.
    wall_max:float=40.
    ground_eps:float=4.
    wall_eps:float=5.
    back_db:float=15.
    xpol_db:float=10.
    leakage_phase:float=0.
    level_sd:float=0.

def fresnel_circular(eps_r,sigma,f,grazing):
    eta=eps_r-1j*sigma/(2*np.pi*f*8.8541878128e-12)
    s=np.sin(grazing);root=np.sqrt(eta-np.cos(grazing)**2)
    gh=(s-root)/(s+root);gv=(eta*s-root)/(eta*s+root)
    return (gv+gh)/2,(gv-gh)/2

def make_scene(n,seed=0,cfg=SceneConfig()):
    if n%4:raise ValueError('n must be divisible by four for exact class balance')
    rng=np.random.default_rng(seed);y=np.tile(np.arange(4),n//4);rng.shuffle(y)
    el=np.deg2rad(rng.uniform(10,80,n));az=rng.uniform(0,2*np.pi,n)
    u=np.c_[np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el)]
    wa=rng.uniform(0,2*np.pi,n);wn=np.c_[np.sin(wa),np.cos(wa),np.zeros(n)]
    # Antenna and satellite are on the same side of the reflecting wall.
    bad=np.einsum('ij,ij->i',u,wn)>0;wn[bad]*=-1
    return dict(y=y,el=el,u=u,wn=wn,wd=rng.uniform(cfg.wall_min,cfg.wall_max,n),
                level=rng.normal(0,cfg.level_sd,n),cfg=cfg)

def excess(p,u,n,d):
    """Positive geometric excess for an illuminated plane with receiver/source on same side."""
    return 2*(np.sum(p*n,axis=-1)-d)*np.sum(u*n,axis=-1)

def signal(scene,baseline=(0,0,0),f=F_L1):
    cfg=scene['cfg'];u=scene['u'];el=scene['el'];y=scene['y'];n=len(y)
    p=np.tile(np.array([0.,0.,cfg.height])+baseline,(n,1));k=2*np.pi*f/C0
    g=np.array([0.,0.,1.]);lg=excess(p,u,g,0.);lw=excess(p,u,scene['wn'],scene['wd'])
    if np.any(lg<0) or np.any(lw<0):raise ValueError('baseline crosses reflector; invalid geometry')
    leakage=10**(-cfg.xpol_db/20)*np.exp(1j*cfg.leakage_phase)
    co,cross=fresnel_circular(cfg.ground_eps,1e-3,f,el)
    ag=(co+leakage*cross)*10**(-cfg.back_db/20)
    co,cross=fresnel_circular(cfg.wall_eps,1e-2,f,np.arcsin(np.clip(-np.sum(u*scene['wn'],axis=1),0,1)))
    aw=co+leakage*cross
    # Relative phases avoid cancellation of large absolute phases; direct voltage is normalized to 1.
    E=(y!=3).astype(complex)
    E+=np.isin(y,[1,2])*ag*np.exp(-1j*k*lg)
    E+=np.isin(y,[2,3])*aw*np.exp(-1j*k*lw)
    power=np.maximum(np.abs(E)**2,1e-24)
    return 35+13*np.sin(el)+scene['level']+10*np.log10(power)

def measure(truth,seed,sigma=.5,q=.25,rho=0.,offsets=None):
    """Correlated Gaussian channel error, optional biases, then exact round-to-nearest quantization."""
    if sigma<0 or q<0 or not 0<=rho<=1:raise ValueError('invalid receiver parameters')
    rng=np.random.default_rng(seed)
    common=rng.normal(size=(len(truth),1));ind=rng.normal(size=truth.shape)
    x=truth+sigma*(np.sqrt(rho)*common+np.sqrt(1-rho)*ind)
    if offsets is not None:x=x+offsets
    return np.round(x/q)*q if q else x

def rms_delta_db(alpha,dphi):return np.sqrt(2)*DB*alpha*np.abs(np.sin(dphi/2))
def phase_step(b,u,n,f=F_L1):return -4*np.pi*f/C0*np.sum(np.asarray(b)*n,axis=-1)*np.sum(u*n,axis=-1)

def difference_pmf(alpha,dphi,sigma,q,mu=40.125,nphase=512):
    """Numerical phase-marginalized PMF of Q(Y1)-Q(Y0), independent Gaussian channel noise.
    Exact nonlinear two-ray power. Known alpha, phase step and free level. Gaussian tails truncated
    beyond 9 sigma; midpoint periodic quadrature. This is a binary oracle, not a four-class bound.
    """
    if not 0<=alpha<1 or sigma<=0 or q<=0:raise ValueError('require alpha in [0,1), sigma,q > 0')
    ph=(np.arange(nphase)+.5)*2*np.pi/nphase
    a=mu+10*np.log10(1+alpha**2+2*alpha*np.cos(ph))
    b=mu+10*np.log10(1+alpha**2+2*alpha*np.cos(ph+dphi))
    low=min(a.min(),b.min(),mu)-9*sigma-q;high=max(a.max(),b.max(),mu)+9*sigma+q
    m=np.arange(np.floor(low/q),np.ceil(high/q)+1)
    def bins(v):return ndtr(((m+.5)*q-v[:,None])/sigma)-ndtr(((m-.5)*q-v[:,None])/sigma)
    p=bins(a);r=bins(b);joint=p.T@r/nphase
    z=bins(np.array([mu]))[0];null=np.outer(z,z)
    offsets=np.arange(-len(m)+1,len(m))
    h1=np.array([np.diagonal(joint,offset=i).sum() for i in offsets]);h0=np.array([np.diagonal(null,offset=i).sum() for i in offsets])
    assert abs(h1.sum()-1)<1e-10 and abs(h0.sum()-1)<1e-10
    return offsets*q,h0,h1

def bayes_error(alpha,dphi,sigma,q,**kw):
    _,p0,p1=difference_pmf(alpha,dphi,sigma,q,**kw)
    return .5*np.minimum(p0,p1).sum()

def self_checks():
    s=make_scene(400,4);p=np.array([0,0,1.5]);g=np.array([0,0,1.])
    assert np.allclose(excess(p,s['u'],g,0),3*np.sin(s['el']))
    u=s['u'];b=np.array([.3,0,.07]);ell0=excess(p,u,g,0);ell1=excess(p+b,u,g,0)
    assert np.allclose(-2*np.pi*F_L1/C0*(ell1-ell0),phase_step(b,u,g))
    d=signal(s,(.5,0,0))-signal(s)
    assert np.max(abs(d[s['y']==1]))<1e-12
    assert np.max(abs(d[np.isin(s['y'],[0,3])]))<1e-12
    assert np.max(abs(signal(s)-signal(s,(0,0,0))))==0
    phi=np.linspace(0,2*np.pi,100000,endpoint=False);a=1e-5;t=.7
    exact=10*np.log10((1+a*a+2*a*np.cos(phi+t))/(1+a*a+2*a*np.cos(phi)))
    assert np.isclose(np.sqrt(np.mean(exact**2)),rms_delta_db(a,t),rtol=1e-5)
    assert np.isclose(bayes_error(0,np.pi,.5,.25),.5)
    assert bayes_error(.1,np.pi,.5,.25)<.5
    constant=np.full((1000,2),40.125)
    assert np.all(measure(constant,1,0,.25)[:,0]==measure(constant,1,0,.25)[:,1])
    obs=measure(np.zeros((100000,2)),22,.5,0,rho=.8)
    assert abs(np.var(obs[:,1]-obs[:,0])-.1)<.003
    print('Passed: geometry, phase convention, blind cases, paired identity, RMS approximation, PMF normalization, binary null.')
