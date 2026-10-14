"""Publication figures from saved results; executed after analysis completes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path('.analysisdeps').resolve()))
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAMES={'small':'Small only','large':'Stronger only','rules':'Recovery rules','learned':'Joint controller','learned_model_only':'Model selection only',
       'learned_no_effort':'No effort penalty','learned_no_history':'History masked'}


def build(directory='results/main'):
    root=Path(directory); out=Path('output/figures'); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'summary.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    order=['small','large','rules','learned','learned_model_only']
    for fault in ['clean','transient']:
        rows={r['policy']:r for r in summary['test'] if r['fault']==fault}
        policies=[p for p in order if p in rows]
        fig,axes=plt.subplots(1,2,figsize=(6,4.5),layout='constrained')
        for ax,metric,interval,title in [(axes[0],'success_rate','success_cluster_interval','Task success'),
                                        (axes[1],'mean_effort_seconds','effort_cluster_interval','Mean deployment effort')]:
            vals=[rows[p][metric]*(100 if metric=='success_rate' else 1) for p in policies]
            cis=[rows[p][interval] for p in policies]; factor=100 if metric=='success_rate' else 1
            errs=[[max(0,v-ci[0]*factor) for v,ci in zip(vals,cis)], [max(0,ci[1]*factor-v) for v,ci in zip(vals,cis)]]
            ax.bar(range(len(vals)),vals,color=['#536b85' if p!='learned' else '#bd6a32' for p in policies],yerr=errs,capsize=3)
            ax.set_xticks(range(len(vals)),[NAMES[p] for p in policies],rotation=35,ha='right')
            ax.set_title(title); ax.set_ylabel('Percent' if metric=='success_rate' else 'Seconds')
            if metric=='success_rate': ax.set_ylim(0,105)
        fig.savefig(out/f'{fault}.png',dpi=220); fig.savefig(out/f'{fault}.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.4)); ax.set_xlim(0,8); ax.set_ylim(0,3.5); ax.axis('off')
    boxes=[(.1,2.2,2.1,1,'Public request\nand task contract'),(2.8,2.2,2.1,1,'Small model\nproposes SQL'),(5.5,2.2,2.1,1,'Execute query\nand collect evidence'),
           (5.5,.2,2.1,1,'Predict success\nand remaining effort'),(2.8,.2,2.1,1,'Verify, repair,\nescalate, retry or stop'),(.1,.2,2.1,1,'Format result\nand score offline')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor='#f3f5f8',edgecolor='#536b85'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center')
    for start,end in [((2.2,2.7),(2.8,2.7)),((4.9,2.7),(5.5,2.7)),((6.55,2.2),(6.55,1.2)),((5.5,.7),(4.9,.7)),((2.8,.7),(2.2,.7))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#536b85'})
    fig.tight_layout(); fig.savefig(out/'architecture.png',dpi=220); fig.savefig(out/'architecture.pdf'); plt.close(fig)
    return out


if __name__=='__main__': print(build())
