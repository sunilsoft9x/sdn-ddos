# Generates Table I (planned) and Table II (exploratory) LaTeX bodies from the WP7 CSVs.
import pandas as pd
iqm=pd.read_csv('summary_iqm.csv'); pt=pd.read_csv('paired_tests.csv'); wa=pd.read_csv('within_arm_tests.csv')
AG=['A2','A1','A3','static']; CMP={'A1':'A2 vs A1','A3':'A2 vs A3','static':'A2 vs static'}
ROWS={'C1_surge':[('service_post','Service after the event',3),
  ('surge_block_share','Share of steps with a rule on a surge host',3),
  ('surge_delivery','Surge-host delivery',3),('benign_delivery_other','Other benign delivery (post/pre)',3),
  ('collateral_post','Collateral',3),('collateral_other_post','Collateral, other benign hosts',3)],
 'C2_swap':[('service_post','Service after the event',3),('worst_post','Worst window, steps 52--99',3),
  ('adapt_latency','Recovery latency (steps)',2),('stale_steps','Steps with a rule on a former attacker',1),
  ('stale_clear_latency','Steps until no such rule',2),('former_attacker_delivery','Former-attacker delivery',3),
  ('attack_delivery_ratio','Attack delivery (post/pre)',3),('benign_delivery_other','Other benign delivery (post/pre)',3),
  ('collateral_post','Collateral',3),('collateral_other_post','Collateral, other benign hosts',3)]}
# Draft 2: C1 worst window and recovery latency (1.000 and 0.00 for every arm) and the six
# cost rows are stated in the text instead (Section V-C), to fit the 2.5-page budget.
out=[]
for cond,title in [('C1_surge','C1: benign surge'),('C2_swap','C2: attacker swap')]:
    out.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}}\\\\')
    for par,lab,d in ROWS[cond]:
        g=iqm[(iqm.cond==cond)&(iqm.parameter==par)&(iqm.family=='planned')].set_index('agent')
        cells=[]
        for a in AG:
            r=g.loc[a]; s=f"{r.iqm:.{d}f} [{r.ci_low:.{d}f}, {r.ci_high:.{d}f}]"
            if a!='A2':
                t=pt[(pt.cond==cond)&(pt.parameter==par)&(pt.comparison==CMP[a])].iloc[0]
                if pd.notna(t.p_holm) and t.p_holm<0.05: s+='$^{*}$'
            cells.append(s)
        out.append(lab+' & '+' & '.join(cells)+' \\\\')
    if cond=='C1_surge': out.append('\\midrule')
open('table1_body.tex','w').write('\n'.join(out)+'\n')
# Table II: exploratory
def wrow(cond,meas,a):
    r=wa[(wa.cond==cond)&(wa.measure==meas)&(wa.agent==a)].iloc[0]; return r
t2=[]
t2.append('\\multicolumn{5}{l}{\\textit{C1: before (steps 2--49) $\\rightarrow$ after (steps 50--99) the surge}}\\\\')
for meas,lab in [('Per-host blocking of the surge hosts (post - pre)','Surge hosts, share of steps with a rule'),
                 ('Per-host blocking of the other benign hosts (post - pre)','Other benign hosts, share of steps with a rule'),
                 ('Other benign hosts with a rule, per step (post - pre)','Other benign hosts with a rule, per step')]:
    cells=[]
    for a in AG:
        r=wrow('C1_surge',meas,a); s=f"{r.pre_iqm:.3f} $\\rightarrow$ {r.post_iqm:.3f}"
        if pd.notna(r.p_holm) and r.p_holm<0.05: s+='$^{*}$'
        cells.append(s)
    t2.append(lab+' & '+' & '.join(cells)+' \\\\')
t2.append('\\midrule')
t2.append('\\multicolumn{5}{l}{\\textit{C2: after the swap (steps 50--99)}}\\\\')
for par,lab,d in [('held_steps','Steps with a held-over rule',2),('reflag_only_steps','Steps with only re-added rules',2)]:
    g=iqm[(iqm.cond=='C2_swap')&(iqm.parameter==par)].set_index('agent'); cells=[]
    for a in AG:
        s=f"{g.loc[a].iqm:.{d}f}"
        if a!='A2':
            t=pt[(pt.cond=='C2_swap')&(pt.parameter==par)&(pt.comparison==CMP[a])].iloc[0]
            if t.p_holm<0.05: s+='$^{\\ddagger}$'
        cells.append(s)
    t2.append(lab+' & '+' & '.join(cells)+' \\\\')
meas='Per-host rate after the swap: re-added rules on former attackers - rules on other benign hosts'
cells=[]
for a in AG:
    r=wrow('C2_swap',meas,a); s=f"{r.post_iqm:.3f} vs {r.pre_iqm:.3f}"
    if r.p_holm<0.05: s+='$^{*}$'
    cells.append(s)
t2.append('Per-host rule rate, former attackers vs hosts benign throughout & '+' & '.join(cells)+' \\\\')
open('table2_body.tex','w').write('\n'.join(t2)+'\n')
print(open('table1_body.tex').read()); print(open('table2_body.tex').read())
