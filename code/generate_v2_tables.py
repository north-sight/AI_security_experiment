import os
import json
import numpy as np
import matplotlib.pyplot as plt
from analyze_v2 import (wilson_score_interval, paired_bootstrap_diff,
                        analyze_transitions_v2, analyze_three_target_overlap_v2,
                        plot_fig4_4_digit_asr_v2)

def generate_all_v2_tables(results_dir, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. 主表 表 4-1
    methods = ['FGSM', 'PGD', 'I-FGSM', 'MI-FGSM', 'DI-FGSM', 'TI-FGSM', 'MDI-FGSM', 'TI-MDI-FGSM',
               'E-I-FGSM', 'E-MI-FGSM', 'E-MDI-FGSM', 'E-TIMDI-FGSM', 'B-I-FGSM', 'B-TIMDI-FGSM']
               
    table_4_1 = []
    for m in methods:
        p = os.path.join(results_dir, f"{m}.json")
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            asrs = d['asrs']
            table_4_1.append({
                'Method': m,
                'NetA': asrs['NetA'],
                'NetB': asrs['NetB'],
                'TargetCNN': asrs['TargetCNN'],
                'TargetMLP': asrs['TargetMLP'],
                'TargetResNet': asrs['TargetResNet'],
                'TargetMean': asrs['TargetMean'],
                'MeanL2': d['mean_l2']
            })
            
    with open(os.path.join(figures_dir, 'table_4_1_main_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(table_4_1, f, indent=2, ensure_ascii=False)
        
    print("\n--- Table 4-1 Main Results V2 ---")
    for r in table_4_1:
        print(f"  {r['Method']:14s} | NetA:{r['NetA']:5.2f}% | NetB:{r['NetB']:5.2f}% | "
              f"CNN:{r['TargetCNN']:5.2f}%, MLP:{r['TargetMLP']:5.2f}%, ResNet:{r['TargetResNet']:5.2f}% "
              f"| TargetMean: {r['TargetMean']:5.2f}% | L2: {r['MeanL2']:.3f}")
              
    # 2. 表 4-2: 配对 Bootstrap 提升区间
    npz_i = os.path.join(results_dir, 'I-FGSM.npz')
    npz_mi = os.path.join(results_dir, 'MI-FGSM.npz')
    npz_mdi = os.path.join(results_dir, 'MDI-FGSM.npz')
    npz_timdi = os.path.join(results_dir, 'TI-MDI-FGSM.npz')
    
    d_i = np.load(npz_i, allow_pickle=True)
    d_mi = np.load(npz_mi, allow_pickle=True)
    d_mdi = np.load(npz_mdi, allow_pickle=True)
    d_timdi = np.load(npz_timdi, allow_pickle=True)
    
    mask = d_i['common_mask']
    lbl = d_i['labels'][mask]
    
    def get_mean_succ(d):
        p_c = d['preds'].item()['TargetCNN'][mask]
        p_m = d['preds'].item()['TargetMLP'][mask]
        p_r = d['preds'].item()['TargetResNet'][mask]
        return ((p_c != lbl).astype(float) + (p_m != lbl).astype(float) + (p_r != lbl).astype(float)) / 3.0
        
    succ_i = get_mean_succ(d_i)
    succ_mi = get_mean_succ(d_mi)
    succ_mdi = get_mean_succ(d_mdi)
    succ_timdi = get_mean_succ(d_timdi)
    
    table_4_2 = {}
    comps = [
        ('MI-FGSM - I-FGSM', succ_mi, succ_i),
        ('MDI-FGSM - I-FGSM', succ_mdi, succ_i),
        ('TI-MDI-FGSM - I-FGSM', succ_timdi, succ_i),
        ('TI-MDI-FGSM - MDI-FGSM', succ_timdi, succ_mdi)
    ]
    for c_name, s1, s2 in comps:
        diff, low, high = paired_bootstrap_diff(s1, s2)
        table_4_2[c_name] = {'diff': diff, 'ci95': [low, high]}
        
    with open(os.path.join(figures_dir, 'table_4_2_bootstrap_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(table_4_2, f, indent=2, ensure_ascii=False)
        
    print("\n--- Table 4-2 Paired Bootstrap V2 ---")
    for k, v in table_4_2.items():
        print(f"  {k:25s}: {v['diff']:+.2f}% (95% CI: [{v['ci95'][0]:.2f}%, {v['ci95'][1]:.2f}%])")
        
    # 3. 表 4-4: 源模型身份对照
    # 比较仅用 A、仅用 B、A+B
    # 获取 B-I-FGSM, B-TIMDI-FGSM, E-I-FGSM, E-TIMDI-FGSM
    table_4_4 = [
        {
            'Config': 'I-FGSM (基础迭代)',
            'Only_A': next(r['TargetMean'] for r in table_4_1 if r['Method'] == 'I-FGSM'),
            'Only_B': next(r['TargetMean'] for r in table_4_1 if r['Method'] == 'B-I-FGSM'),
            'Dual_AB': next(r['TargetMean'] for r in table_4_1 if r['Method'] == 'E-I-FGSM')
        },
        {
            'Config': 'TI-MDI-FGSM (全量增强)',
            'Only_A': next(r['TargetMean'] for r in table_4_1 if r['Method'] == 'TI-MDI-FGSM'),
            'Only_B': next(r['TargetMean'] for r in table_4_1 if r['Method'] == 'B-TIMDI-FGSM'),
            'Dual_AB': next(r['TargetMean'] for r in table_4_1 if r['Method'] == 'E-TIMDI-FGSM')
        }
    ]
    with open(os.path.join(figures_dir, 'table_4_4_source_identity_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(table_4_4, f, indent=2, ensure_ascii=False)
        
    # 4. 表 4-6: 开销表
    grad_calls = {
        'FGSM': 1, 'PGD': 10, 'I-FGSM': 10, 'MI-FGSM': 10, 'DI-FGSM': 10, 'TI-FGSM': 10,
        'MDI-FGSM': 10, 'TI-MDI-FGSM': 10, 'E-I-FGSM': 20, 'E-TIMDI-FGSM': 20
    }
    table_4_6 = []
    for m in ['FGSM', 'I-FGSM', 'PGD', 'MI-FGSM', 'TI-FGSM', 'TI-MDI-FGSM', 'E-TIMDI-FGSM']:
        p = os.path.join(results_dir, f"{m}.json")
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            table_4_6.append({
                'Method': m,
                'GradCalls': grad_calls[m],
                'MsPerImg': round(d['ms_per_img'], 3),
                'MeanL2': round(d['mean_l2'], 3),
                'MaxLinf': round(d['max_l_inf'], 5)
            })
    with open(os.path.join(figures_dir, 'table_4_6_overhead_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(table_4_6, f, indent=2, ensure_ascii=False)
        
    # 5. 表 4-8: 逐样本状态转移流转分析
    t4_8 = {}
    for comp_name, (f_b, f_i) in [('MI-FGSM / I-FGSM', (npz_i, npz_mi)),
                                  ('TI-MDI-FGSM / I-FGSM', (npz_i, npz_timdi)),
                                  ('TI-MDI-FGSM / MDI-FGSM', (npz_mdi, npz_timdi))]:
        t4_8[comp_name] = {}
        for tgt in ['TargetCNN', 'TargetMLP', 'TargetResNet']:
            ng, l, net_g = analyze_transitions_v2(f_b, f_i, tgt)
            t4_8[comp_name][tgt] = {'newly_gained': ng, 'lost': l, 'net_gain': net_g}
    with open(os.path.join(figures_dir, 'table_4_8_transitions_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(t4_8, f, indent=2, ensure_ascii=False)
        
    # 6. 表 4-9: 三黑盒目标重叠分类统计
    t4_9 = {}
    for m in ['I-FGSM', 'MI-FGSM', 'MDI-FGSM', 'TI-MDI-FGSM', 'E-TIMDI-FGSM']:
        p = os.path.join(results_dir, f"{m}.npz")
        all_3, any_2, only_1, all_f = analyze_three_target_overlap_v2(p)
        t4_9[m] = {
            'all_three_succeed': all_3,
            'any_two_succeed': any_2,
            'only_one_succeed': only_1,
            'all_three_fail': all_f
        }
    with open(os.path.join(figures_dir, 'table_4_9_overlap_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(t4_9, f, indent=2, ensure_ascii=False)
        
    # 7. 绘制图 4-4: 按数字易感性
    plot_fig4_4_digit_asr_v2(results_dir, figures_dir)
    print("\nAll V2 tables and figures compiled successfully!")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_v2_tables(os.path.join(base_dir, 'results_v2'), os.path.join(base_dir, 'figures_v2'))
