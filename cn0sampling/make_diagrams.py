"""Deterministic scientific diagrams; both PNG (300 dpi) and editable SVG/PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path('results/figures')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
def box(ax,x,y,w,h,title,body,color='#eaf2f8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',fc=color,ec='#27465b',lw=1))
 ax.text(x+w/2,y+h*.76,title,ha='center',va='center',weight='bold',fontsize=9.5)
 ax.text(x+w/2,y+h*.37,body,ha='center',va='center',fontsize=8.5,linespacing=1.5)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#27465b',lw=1.2))
def save(fig,n):
 for ext in ['png','pdf','svg']:fig.savefig(ROOT/(n+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def main(output_dir='results/figures'):
 global ROOT
 ROOT=Path(output_dir);ROOT.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(figsize=(10,5.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 box(ax,.02,.60,.28,.32,'Propagation and geometry','LOS / ground / wall / NLOS\nWavelength and baseline vectors\nRelative complex fields')
 box(ax,.36,.60,.28,.32,'Measurement formation','Antenna voltage response\nChannel-specific level and bias\nCorrelated noise → quantization')
 box(ax,.70,.60,.28,.32,'Observable differences','Spatial or L1–L5 samples\nCalibration before differencing\nShared reference channel')
 arrow(ax,(.30,.76),(.36,.76));arrow(ax,(.64,.76),(.70,.76))
 box(ax,.02,.08,.28,.32,'E1–E2: physical checks','Phase convention and null cases\nBaseline orientation and length\nExact vs weak-signal response','#f0f4ec')
 box(ax,.36,.08,.28,.32,'E3 and E5: empirical tests','Paired train/test scenes; five seeds\nEqual-channel layout controls\nBias, calibration and scene shifts','#f0f4ec')
 box(ax,.70,.08,.28,.32,'E4: binary detection limit','Phase-marginalized discrete PMFs\nEqual-prior Bayes error\nExplicit nuisance assumptions','#f0f4ec')
 for x in [.16,.5,.84]:arrow(ax,(x,.60),(x,.40))
 save(fig,'Framework_overview')
 fig,axs=plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios':[1.1,1]})
 ax=axs[0];ax.set(xlim=(-.1,1.1),ylim=(-.05,1.08));ax.axis('off');ax.set_title('(a) Complementary spatial samples',fontsize=11,weight='bold')
 ax.plot([0,1],[.12,.12],color='#777',lw=3);ax.text(.06,.03,'Ideal horizontal ground',fontsize=9)
 ax.plot([.99,.99],[.12,.95],color='#777',lw=4);ax.text(.98,.99,'Vertical wall',ha='right',fontsize=9)
 pts={'R0':(.27,.46),'RH':(.72,.46),'RV':(.27,.76)}
 for label,(x,y) in pts.items():ax.scatter(x,y,s=90,c='#176ba0',zorder=4);ax.text(x-.025,y+.045,label,ha='right',weight='bold')
 ax.annotate('',xy=pts['RH'],xytext=pts['R0'],arrowprops=dict(arrowstyle='<->',color='#176ba0'));ax.text(.49,.39,r'$b_h$',ha='center')
 ax.annotate('',xy=pts['RV'],xytext=pts['R0'],arrowprops=dict(arrowstyle='<->',color='#176ba0'));ax.text(.34,.62,r'$b_v$')
 for target in ['R0','RH']:
  x,y=pts[target];ax.annotate('',xy=(x,y),xytext=(x-.18,y+.22),arrowprops=dict(arrowstyle='->',color='#777'))
 ax.text(.02,.94,'Incident plane waves',fontsize=9)
 ax.text(.53,.21,'Ground phase changes with height;\nwall phase changes along its normal.',ha='center',fontsize=9)
 ax.text(.5,-.01,'Schematic only; not to scale',ha='center',fontsize=8,color='#555')
 ax=axs[1];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('(b) Calibration and evaluation sequence',fontsize=11,weight='bold')
 steps=[('1  Independent open-sky calibration','Fit channel offsets versus elevation'),('2  Synchronized C/N₀ observations','Keep missing/invalid reports explicit'),('3  Calibrate and form differences','Use a common R0 measurement'),('4  Apply frozen decision rule','Classifier or specified binary likelihood'),('5  Report evidence and limits','Class metrics, paired gains and shift tests')]
 for i,(t,b) in enumerate(steps):
  y=.82-i*.185;box(ax,.04,y,.92,.14,t,b)
  if i<4:arrow(ax,(.5,y),(.5,y-.045))
 fig.tight_layout(w_pad=3);save(fig,'Operation_scheme')
if __name__=='__main__':main()
