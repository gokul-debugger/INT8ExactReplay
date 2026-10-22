"""Generate the article figure from saved/recomputed target values."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze import BASE,compute
from arithmetic import unpack


def render(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    h=compute()[1][-1]['classes'][7]
    values=[('Sequential scales S2',h['scale_paths']['S2']['maps']['exact']),
            ('Folded scales S3 / S4',h['scale_paths']['S3']['rounding']),
            ('Exact scaled reference',h['ideal_r_map']),
            ('Observed float-output score\n(exact division)',h['native_pre_maps']['exact'])]
    fig,ax=plt.subplots(figsize=(9,4.4));fig.subplots_adjust(left=.34,right=.96,bottom=.22,top=.79)
    ax.axvspan(-4,0,color='#f7ead6');ax.axvspan(0,11,color='#e5f1fa')
    ax.axvline(0,color='#424242',linestyle='--',linewidth=1.2)
    for i,(label,mapping) in enumerate(values):
        distance=float(unpack(mapping['signed_threshold_distance_code_units']))*1e6
        color='#a95811' if mapping['code']==16 else '#1266a6'
        ax.scatter(distance,i,s=80,color=color,zorder=3)
        ax.annotate(f'code {mapping["code"]}',(distance,i),xytext=(8,9),textcoords='offset points',fontsize=10,color=color)
    ax.set_yticks(range(4),[v[0] for v in values]);ax.invert_yaxis();ax.set_xlim(-4,11);ax.set_ylim(3.55,-.65)
    ax.set_xlabel('Distance from the rounding midpoint (millionths of a code unit)',labelpad=10)
    ax.set_xticks([-4,-2,0,2,4,6,8,10]);ax.grid(axis='x',alpha=.2)
    ax.tick_params(axis='y',length=0,pad=12)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    fig.suptitle('A small arithmetic difference can change the output code',fontsize=15,y=.96)
    fig.text(.34,.86,'Midpoint: 96.5 before zero point; ties round to 96, then −80 gives code 16.',fontsize=10)
    fig.text(.03,.025,'Image 4054, horse output. Reference/candidate values; both native paths still predict horse.',fontsize=9,color='#444444')
    for suffix in ['png','svg']:
        metadata={'Software':'Matplotlib'} if suffix=='png' else {'Date':None,'Creator':'Matplotlib'}
        with matplotlib.rc_context({'svg.hashsalt':'int8-exact-replay-v1'}):fig.savefig(out/f'rounding-boundary.{suffix}',dpi=180,metadata=metadata)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'generated/figures');render(p.parse_args().out)
