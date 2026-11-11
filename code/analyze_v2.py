import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def wilson_score_interval(successes, total, conf=0.95):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z = 1.959964
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return round((centre - spread) * 100.0, 2), round((centre + spread) * 100.0, 2)

def paired_bootstrap_diff(succ1, succ2, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(succ1)
    diff = succ1.astype(float) - succ2.astype(float)
    indices = rng.randint(0, n, size=(n_boot, n))
    boot_diffs = diff[indices].mean(axis=1) * 100.0
    low = np.percentile(boot_diffs, 2.5)
    high = np.percentile(boot_diffs, 97.5)
    return round(float(boot_diffs.mean()), 2), round(float(low), 2), round(float(high), 2)

def analyze_transitions_v2(npz_base_path, npz_imp_path, target_name='TargetCNN'):
    d_base = np.load(npz_base_path, allow_pickle=True)
    d_imp = np.load(npz_imp_path, allow_pickle=True)
    
    mask = d_base['common_mask']
    labels = d_base['labels'][mask]
    pred_base = d_base['preds'].item()[target_name][mask]
    pred_imp = d_imp['preds'].item()[target_name][mask]
    
    succ_base = (pred_base != labels)
    succ_imp = (pred_imp != labels)
    
    new_g = int(np.sum((~succ_base) & succ_imp))
    lost = int(np.sum(succ_base & (~succ_imp)))
    net_g = new_g - lost
    return new_g, lost, net_g

def analyze_three_target_overlap_v2(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    mask = d['common_mask']
    labels = d['labels'][mask]
    
    p_cnn = d['preds'].item()['TargetCNN'][mask]
    p_mlp = d['preds'].item()['TargetMLP'][mask]
    p_res = d['preds'].item()['TargetResNet'][mask]
    
    s_c = (p_cnn != labels).astype(int)
    s_m = (p_mlp != labels).astype(int)
    s_r = (p_res != labels).astype(int)
    
    total_succ = s_c + s_m + s_r
    all_three = int(np.sum(total_succ == 3))
    any_two = int(np.sum(total_succ == 2))
    only_one = int(np.sum(total_succ == 1))
    all_fail = int(np.sum(total_succ == 0))
    
    return all_three, any_two, only_one, all_fail

def plot_fig4_4_digit_asr_v2(results_dir, out_dir):
    methods = ['I-FGSM', 'MI-FGSM', 'MDI-FGSM', 'TI-MDI-FGSM', 'E-TIMDI-FGSM']
    digits = list(range(10))
    digit_asrs = {m: [] for m in methods}
    
    for m in methods:
        p = os.path.join(results_dir, f"{m}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        mask = d['common_mask']
        labels = d['labels'][mask]
        
        p_c = d['preds'].item()['TargetCNN'][mask]
        p_m = d['preds'].item()['TargetMLP'][mask]
        p_r = d['preds'].item()['TargetResNet'][mask]
        
        mean_w = ((p_c != labels).astype(float) + (p_m != labels).astype(float) + (p_r != labels).astype(float)) / 3.0
        for digit in digits:
            d_idx = np.where(labels == digit)[0]
            val = mean_w[d_idx].mean() * 100.0 if len(d_idx) > 0 else 0.0
            digit_asrs[m].append(round(float(val), 2))
            
    plt.figure(figsize=(9.5, 5))
    x = np.arange(10)
    width = 0.16
    colors = ['#4C72B0', '#55A868', '#DD8452', '#8172B3', '#C44E52']
    
    for idx, m in enumerate(methods):
        if m in digit_asrs and len(digit_asrs[m]) == 10:
            plt.bar(x + (idx - 2) * width, digit_asrs[m], width, label=m, color=colors[idx])
            
    plt.xlabel('Digit Class (0 ~ 9)', fontsize=12)
    plt.ylabel('Target Mean ASR (%) across 3 Black-box Models', fontsize=12)
    plt.title('Transferability Sensitivity across Digit Classes (eps=0.2, T=10)', fontsize=13)
    plt.xticks(x, [f"'{d}'" for d in digits], fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_4_digit_asr_v2.png'), dpi=300)
    plt.close()
    print("Saved fig4_4_digit_asr_v2.png")
    
    with open(os.path.join(out_dir, 'fig4_4_digit_asr_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(digit_asrs, f, indent=2, ensure_ascii=False)
