import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def wilson_score_interval(successes, total, conf=0.95):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z = 1.959964 # 95%
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (centre - spread) * 100.0, (centre + spread) * 100.0

def paired_bootstrap_diff(succ1, succ2, n_boot=10000, seed=42):
    """
    配对 Bootstrap 检验两种方法成功率差值的 95% 置信区间
    succ1, succ2: 形状为 (N,) 的布尔数组，表示每个样本是否成功
    """
    rng = np.random.RandomState(seed)
    n = len(succ1)
    diff = succ1.astype(float) - succ2.astype(float)
    boot_means = []
    # 向量化批抽样以加速计算
    indices = rng.randint(0, n, size=(n_boot, n))
    boot_diffs = diff[indices].mean(axis=1) * 100.0
    low = np.percentile(boot_diffs, 2.5)
    high = np.percentile(boot_diffs, 97.5)
    return float(boot_diffs.mean()), float(low), float(high)

def analyze_transitions(npz_baseline, npz_improved, target_name='TargetCNN'):
    d_base = np.load(npz_baseline, allow_pickle=True)
    d_imp = np.load(npz_improved, allow_pickle=True)
    
    mask = d_base['common_mask']
    labels = d_base['labels'][mask]
    pred_base = d_base['preds'].item()[target_name][mask]
    pred_imp = d_imp['preds'].item()[target_name][mask]
    
    succ_base = (pred_base != labels)
    succ_imp = (pred_imp != labels)
    
    newly_gained = np.sum((~succ_base) & succ_imp)
    lost = np.sum(succ_base & (~succ_imp))
    net_gain = newly_gained - lost
    
    return newly_gained, lost, net_gain

def analyze_dual_target_overlap(npz_file):
    d = np.load(npz_file, allow_pickle=True)
    mask = d['common_mask']
    labels = d['labels'][mask]
    pred_cnn = d['preds'].item()['TargetCNN'][mask]
    pred_mlp = d['preds'].item()['TargetMLP'][mask]
    
    succ_cnn = (pred_cnn != labels)
    succ_mlp = (pred_mlp != labels)
    
    both_succ = np.sum(succ_cnn & succ_mlp)
    only_cnn = np.sum(succ_cnn & (~succ_mlp))
    only_mlp = np.sum((~succ_cnn) & succ_mlp)
    both_fail = np.sum((~succ_cnn) & (~succ_mlp))
    
    return both_succ, only_cnn, only_mlp, both_fail

def plot_ablation_summary(results_dir='./results', out_dir='./figures'):
    os.makedirs(out_dir, exist_ok=True)
    methods = ['I-FGSM', 'MI-FGSM', 'DI-FGSM', 'MDI-FGSM',
               'E-I-FGSM', 'E-MI-FGSM', 'E-DI-FGSM', 'E-MDI-FGSM']
    
    cnn_asrs = []
    mlp_asrs = []
    mean_asrs = []
    
    for m in methods:
        path = os.path.join(results_dir, f"{m}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cnn_asrs.append(data['asrs']['TargetCNN'])
                mlp_asrs.append(data['asrs']['TargetMLP'])
                mean_asrs.append(data['asrs']['TargetMean'])
        else:
            cnn_asrs.append(0)
            mlp_asrs.append(0)
            mean_asrs.append(0)
            
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, cnn_asrs, width, label='Target CNN', color='#4C72B0')
    ax.bar(x, mlp_asrs, width, label='Target MLP', color='#DD8452')
    ax.bar(x + width, mean_asrs, width, label='Target Mean', color='#55A868')
    
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('Factorial Ablation on Black-box Transferability (eps=0.2, T=10)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation_summary.png'), dpi=300)
    plt.close()
    print("Saved ablation_summary.png")
