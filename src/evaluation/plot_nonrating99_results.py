"""生成99流域结果图；按实际运行名单配对，保留极端值。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'hydro-multitask-paper/figures/nonrating99'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})
COLORS = ['#92b5ce','#17628a','#e9ad82','#b84b23']
roots = {'physical':ROOT/'results/nonrating14_full_matrix',
         'observed':ROOT/'results/nonrating14_observed_matrix'}
meta = json.loads((roots['physical']/'experiment.json').read_text(encoding='utf-8'))
candidates = meta['candidate_basins']
controls = meta['matched_controls']
data = {k:pd.read_csv(v/'metrics_all.csv',dtype={'basin':str}) for k,v in roots.items()}

def save(fig, name, note):
    fig.text(.02,.015,note,fontsize=9,va='bottom',color='#444444')
    fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def basin_values(scale, scenario, ids, arch, task='flow'):
    d=data[scale]
    return d[(d.scenario==scenario)&(d.architecture==arch)&(d.task==task)&d.basin.isin(ids)].groupby('basin').nse.mean().reindex(ids)

# 图1：对称对数轴保留所有极端负NSE，右侧给出双头的准确数值。
fig,(ax,table)=plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[3.3,1.7]})
order=basin_values('observed','candidate_fixed',candidates,'dual_head').sort_values(ascending=False).index.tolist()
y=np.arange(len(order))
for i,(scale,arch,label) in enumerate([
    ('physical','single_flow','physical·单任务Q'),('physical','dual_head','physical·双头Q'),
    ('observed','single_flow','observed·单任务Q'),('observed','dual_head','observed·双头Q')]):
    ax.scatter(basin_values(scale,'candidate_fixed',order,arch),y+(i-1.5)*.15,
               color=COLORS[i],label=label,s=38)
ax.set_xscale('symlog',linthresh=1);ax.set_xlim(-15000,1)
ax.set_xticks([-10000,-1000,-100,-10,-1,0,.5,1]);ax.set_xticklabels(['−10000','−1000','−100','−10','−1','0','0.5','1'])
ax.set_yticks(y,order);ax.invert_yaxis();ax.axvline(0,color='#555',lw=1)
ax.grid(axis='x',alpha=.2);ax.set_xlabel('测试Q NSE（−1至1为线性区间，极端负值采用对称对数轴）')
ax.legend(loc='lower left',fontsize=9);ax.set_title('候选14流域：四种训练结果')
table.axis('off')
vals=np.column_stack([basin_values(s,'candidate_fixed',order,'dual_head') for s in ['physical','observed']])
t=table.table(cellText=[[b,f'{p:.3f}',f'{o:.3f}'] for b,(p,o) in zip(order,vals)],
              colLabels=['流域ID','双头 physical','双头 observed'],cellLoc='center',loc='center')
t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.9)
fig.tight_layout(rect=[0,.08,1,1]);save(fig,'fig01_candidates',
 '每个点为3个模型种子的平均NSE。训练与验证Q均屏蔽；H保留。observed使用真实训练期Q尺度。01100561属于原86。')

# 图2：流域作为统计单位，增益为同流域同种子双头减单任务。
fig,axes=plt.subplots(2,2,figsize=(12,9))
for col,(scenario,ids,title) in enumerate([('candidate_fixed',candidates,'候选14流域'),('matched_86_fixed',controls,'原86匹配对照14流域')]):
    arrays=[basin_values(s,scenario,ids,a).to_numpy() for s in ['physical','observed'] for a in ['single_flow','dual_head']]
    ax=axes[0,col];bp=ax.boxplot(arrays,patch_artist=True,showfliers=False)
    for patch,c in zip(bp['boxes'],COLORS):patch.set_facecolor(c);patch.set_alpha(.4)
    for j,(arr,c) in enumerate(zip(arrays,COLORS),1):
        ax.scatter(np.full(len(arr),j)+np.linspace(-.12,.12,len(arr)),arr,s=20,color=c)
    ax.set_xticks(range(1,5),['physical\n单任务','physical\n双头','observed\n单任务','observed\n双头'])
    ax.set_title(title+'：绝对NSE');ax.axhline(0,color='#555',lw=.8)
    if col==0:
        ax.set_yscale('symlog',linthresh=1)
        ax.set_ylim(min(np.min(v) for v in arrays)*1.3,1)
    ax.set_ylabel('Q NSE');ax.grid(axis='y',alpha=.2)
    ax=axes[1,col]
    for j,s in enumerate(['physical','observed'],1):
        delta=basin_values(s,scenario,ids,'dual_head')-basin_values(s,scenario,ids,'single_flow')
        ax.scatter(j+np.linspace(-.12,.12,len(delta)),delta,s=30,color=COLORS[2*j-1])
        ax.plot([j-.25,j+.25],[delta.median()]*2,color='#222',lw=2)
        ax.text(j,1.02,f'改善 {(delta>0).sum()}/14',transform=ax.get_xaxis_transform(),ha='center',va='bottom')
    if col==0:ax.set_yscale('symlog',linthresh=1)
    ax.set_xticks([1,2],['physical','observed']);ax.set_xlim(.5,2.5)
    ax.axhline(0,color='#555',lw=.8);ax.set_ylabel('ΔNSE＝双头−单任务');ax.grid(axis='y',alpha=.2)
fig.tight_layout(rect=[0,.08,1,1]);save(fig,'fig02_candidate_control',
 '每个散点为流域的3种子均值；黑横线为中位数。两组来自独立的99流域固定留出实验，均屏蔽训练与验证Q。候选极端值全部保留。')

# 图3：从各运行文件读取，避免metrics_all中缺少mask_seed导致重复配对。
rows=[]
for s,root in roots.items():
    for ratio in [30,50,70]:
        for ms in [1,2,3]:
            for xs in [42,123,456]:
                prefix=f'random_q_holdout{ratio}'
                r=json.loads((root/f'{prefix}_single_flow_ms{ms}_xs{xs}'/'run.json').read_text(encoding='utf-8'))
                held=r['mask']['per_task']['flow']['held_out_basins']
                for arch in ['single_flow','dual_head']:
                    d=pd.read_csv(root/f'{prefix}_{arch}_ms{ms}_xs{xs}'/'metrics.csv',dtype={'basin':str})
                    v=d[(d.task=='flow')&d.basin.isin(held)].nse
                    assert len(v)==len(held)
                    rows.append(dict(scale=s,ratio=ratio,model_seed=ms,mask_seed=xs,arch=arch,mean=v.mean(),median=v.median()))
random=pd.DataFrame(rows)
fig,axes=plt.subplots(2,2,figsize=(12,9))
for col,s in enumerate(['physical','observed']):
    for row,metric in enumerate(['mean','median']):
        ax=axes[row,col]
        for i,arch in enumerate(['single_flow','dual_head']):
            groups=[random[(random.scale==s)&(random.arch==arch)&(random.ratio==r)][metric] for r in [30,50,70]]
            x=np.array([30,50,70])+(i-.5)*1.5
            ax.plot(x,[v.mean() for v in groups],color=COLORS[2*col+i],marker='o',label='单任务Q' if i==0 else '双头Q')
            for xx,v in zip(x,groups):ax.scatter(xx+np.linspace(-.4,.4,9),v,s=15,alpha=.6,color=COLORS[2*col+i])
        if row==0 and col==0:ax.set_yscale('symlog',linthresh=1)
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks([30,50,70],['30%','50%','70%'])
        ax.set_title(s+('：各运行的流域平均NSE' if row==0 else '：各运行的流域中位NSE'))
        ax.set_xlabel('随机Q留出比例');ax.set_ylabel('测试Q NSE');ax.legend();ax.grid(alpha=.2)
fig.tight_layout(rect=[0,.08,1,1]);save(fig,'fig03_random_holdout',
 '散点为模型种子×掩膜种子的9次运行，折线为9次运行平均；只评价该次实际留出流域。当前协议：训练Q屏蔽，验证Q保留用于选模。')

# 图4：共同原85、共同模型种子1/2/3，分别比较同一架构。
fig,axes=plt.subplots(1,3,figsize=(15,6))
common=pd.read_csv(ROOT/'results/runs/4K_best_physical_complete_dual_head_ms1_L480_physical_extended/metrics.csv',dtype={'basin':str}).basin.unique().tolist()
common=[b for b in common if b not in candidates];assert len(common)==85
comparisons=[]
for ax,(arch,task,title) in zip(axes,[('single_flow','flow','单任务Q'),('dual_head','flow','双头Q'),('dual_head','waterlevel','双头H')]):
    chunks=[]
    for ms in [1,2,3]:
        d=pd.read_csv(ROOT/f'results/runs/4K_best_physical_complete_{arch}_ms{ms}_L480_physical_extended/metrics.csv',dtype={'basin':str})
        chunks.append(d[(d.task==task)&d.basin.isin(common)])
    old=pd.concat(chunks).groupby('basin').nse.mean().reindex(common)
    new=basin_values('physical','candidate_fixed',common,arch,task)
    ax.scatter(old,new,s=24,alpha=.7,color='#17628a');lo=min(old.min(),new.min())-.03;hi=max(old.max(),new.max())+.03
    ax.plot([lo,hi],[lo,hi],color='#666',ls='--');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
    ax.set_xlabel('原86训练：NSE');ax.set_ylabel('99训练：原85 NSE');ax.set_title(title)
    delta=new-old
    ax.text(.04,.96,f'均值 {old.mean():.3f} → {new.mean():.3f}\nΔ={delta.mean():+.4f}\n改善 {(delta>0).sum()}/85',transform=ax.transAxes,va='top')
    ax.grid(alpha=.2)
    comparisons.extend(dict(arch=arch,task=task,basin=b,old=old[b],new=new[b],delta=delta[b]) for b in common)
fig.tight_layout(rect=[0,.15,1,.95]);save(fig,'fig04_original85',
 '共同原85流域（排除01100561），每站平均模型种子1/2/3；L480、扩展属性、physical。\n原86完整监督 vs 99候选固定留出：后者屏蔽候选14的训练/验证Q；预处理统计量也重新计算。此图为描述性对比。')
print(f'已生成4张PNG和4张SVG：{OUT}')
