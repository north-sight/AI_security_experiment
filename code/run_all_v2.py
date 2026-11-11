import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models_v2 import NetA, NetB_Modern, TargetCNN_Deep, TargetMLP_Deep, TargetResNet_Mini
from dataset_v2 import get_mnist_data_v2
from attacks_v2 import fgsm_attack, pgd_attack, advanced_iterative_attack
from evaluate_v2 import load_models_v2, get_clean_predictions_v2, run_attack_config_v2
from analyze import paired_bootstrap_diff, wilson_score_interval
from generate_figures_v2 import (plot_fig3_1_val_accuracy_v2, plot_fig4_3_ablation_v2,
                                generate_adversarial_examples_v2)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def run_budget_and_step_scans_v2(models_dict, test_subset, common_mask, device, out_dir):
    net_a = models_dict['NetA']
    net_b = models_dict['NetB']
    dual = [net_a, net_b]
    loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    # 1. 预算扫描: eps in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    eps_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    scan_methods = {
        'FGSM': ('FGSM', net_a, False, False, False),
        'I-FGSM': ('I-FGSM', net_a, False, False, False),
        'MI-FGSM': ('MI-FGSM', net_a, True, False, False),
        'TI-FGSM': ('TI-FGSM', net_a, False, False, True),
        'TI-MDI-FGSM': ('TI-MDI-FGSM', net_a, True, True, True),
        'E-TIMDI-FGSM': ('E-TIMDI-FGSM', dual, True, True, True)
    }
    
    budget_results = {m: [] for m in scan_methods}
    for eps in eps_list:
        print(f"Scanning Budget eps={eps}...")
        for m_name, (mth, srcs, mom, div, ti) in scan_methods.items():
            cnn_w, mlp_w, res_w = 0, 0, 0
            for b_idx, (x, y) in enumerate(loader):
                torch.manual_seed(42 + 1000003 * b_idx)
                x, y = x.to(device), y.to(device)
                if mth == 'FGSM':
                    x_adv = fgsm_attack(srcs, x, y, eps=eps)
                else:
                    x_adv = advanced_iterative_attack(srcs, x, y, eps=eps, steps=10,
                                                      momentum=mom, diversity=div, translation_inv=ti)
                with torch.no_grad():
                    p_c = models_dict['TargetCNN'](x_adv).argmax(dim=1)
                    p_m = models_dict['TargetMLP'](x_adv).argmax(dim=1)
                    p_r = models_dict['TargetResNet'](x_adv).argmax(dim=1)
                    
                b_mask = torch.tensor(common_mask[b_idx*100 : (b_idx+1)*100], device=device)
                cnn_w += ((p_c != y) & b_mask).sum().item()
                mlp_w += ((p_m != y) & b_mask).sum().item()
                res_w += ((p_r != y) & b_mask).sum().item()
                
            total_s = common_mask.sum()
            mean_asr = ((cnn_w + mlp_w + res_w) / (3.0 * total_s)) * 100.0
            budget_results[m_name].append(round(mean_asr, 2))
            
    # 绘制图 4-1
    plt.figure(figsize=(7.5, 5))
    markers = ['o-', 's-', '^-', 'v-', 'd-', '*-']
    for idx, (m_name, vals) in enumerate(budget_results.items()):
        plt.plot(eps_list, vals, markers[idx], label=m_name, linewidth=2, markersize=6)
    plt.xlabel('Perturbation Budget $\\epsilon$', fontsize=12)
    plt.ylabel('Target Mean ASR (%) across 3 Black-box Models', fontsize=12)
    plt.title('Impact of Perturbation Budget on Black-box Transfer ASR', fontsize=13)
    plt.xticks(eps_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_1_budget_scan_v2.png'), dpi=300)
    plt.close()
    print("Saved fig4_1_budget_scan_v2.png")
    
    with open(os.path.join(out_dir, 'table_4_3_budget_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(budget_results, f, indent=2, ensure_ascii=False)
        
    # 2. 步数扫描: T in [1, 5, 10, 15, 20, 30, 40], eps=0.2
    steps_list = [1, 5, 10, 15, 20, 30, 40]
    step_methods = {
        'I-FGSM (White-box A)': ('I-FGSM', net_a, 'NetA', False, False, False),
        'I-FGSM (Target Mean)': ('I-FGSM', net_a, 'Mean', False, False, False),
        'MI-FGSM (Target Mean)': ('MI-FGSM', net_a, 'Mean', True, False, False),
        'TI-MDI-FGSM (Target Mean)': ('TI-MDI-FGSM', net_a, 'Mean', True, True, True),
        'E-TIMDI-FGSM (Target Mean)': ('E-TIMDI-FGSM', dual, 'Mean', True, True, True)
    }
    step_results = {m: [] for m in step_methods}
    for stp in steps_list:
        print(f"Scanning Steps T={stp}...")
        for m_name, (mth, srcs, target_type, mom, div, ti) in step_methods.items():
            cnt = 0
            for b_idx, (x, y) in enumerate(loader):
                torch.manual_seed(42 + 1000003 * b_idx)
                x, y = x.to(device), y.to(device)
                x_adv = advanced_iterative_attack(srcs, x, y, eps=0.2, steps=stp,
                                                  momentum=mom, diversity=div, translation_inv=ti)
                with torch.no_grad():
                    b_mask = torch.tensor(common_mask[b_idx*100 : (b_idx+1)*100], device=device)
                    if target_type == 'NetA':
                        p = models_dict['NetA'](x_adv).argmax(dim=1)
                        cnt += ((p != y) & b_mask).sum().item()
                    else:
                        p_c = models_dict['TargetCNN'](x_adv).argmax(dim=1)
                        p_m = models_dict['TargetMLP'](x_adv).argmax(dim=1)
                        p_r = models_dict['TargetResNet'](x_adv).argmax(dim=1)
                        cnt += (((p_c != y) & b_mask).sum().item() +
                                ((p_m != y) & b_mask).sum().item() +
                                ((p_r != y) & b_mask).sum().item()) / 3.0
            total_s = common_mask.sum()
            asr = (cnt / total_s) * 100.0
            step_results[m_name].append(round(asr, 2))
            
    # 绘制图 4-2
    plt.figure(figsize=(7.5, 5))
    plt.plot(steps_list, step_results['I-FGSM (White-box A)'], 'o--', color='gray', label='I-FGSM (White-box Source A)', linewidth=2)
    plt.plot(steps_list, step_results['I-FGSM (Target Mean)'], 's-', color='#4C72B0', label='I-FGSM (Target Mean)', linewidth=2)
    plt.plot(steps_list, step_results['MI-FGSM (Target Mean)'], '^-', color='#55A868', label='MI-FGSM (Target Mean)', linewidth=2)
    plt.plot(steps_list, step_results['TI-MDI-FGSM (Target Mean)'], 'd-', color='#DD8452', label='TI-MDI-FGSM (Target Mean)', linewidth=2)
    plt.plot(steps_list, step_results['E-TIMDI-FGSM (Target Mean)'], '*-', color='#C44E52', label='E-TIMDI-FGSM (Target Mean)', linewidth=2)
    plt.xlabel('Number of Iteration Steps $T$', fontsize=12)
    plt.ylabel('Attack Success Rate (%)', fontsize=12)
    plt.title('White-box vs Black-box Transfer Overfitting Dynamics', fontsize=13)
    plt.xticks(steps_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_2_step_scan_v2.png'), dpi=300)
    plt.close()
    print("Saved fig4_2_step_scan_v2.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running AI Security V2 Evaluation Pipeline ===")
    print(f"Device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    ckpt_dir = os.path.join(base_dir, 'checkpoints_v2')
    results_dir = os.path.join(base_dir, 'results_v2')
    figures_dir = os.path.join(base_dir, 'figures_v2')
    course_weight = os.path.join(base_dir, '..', 'resource', 'lenet_mnist_model.pth')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. 准备数据
    train_set, val_set, test_full, test_subset, sampled_indices = get_mnist_data_v2(data_dir=data_dir, seed=1024)
    
    # 2. 加载 5 款模型
    models_dict = load_models_v2(ckpt_dir, course_weight, device)
    
    # 3. 干净测试与共同正确集合 S 筛选
    labels_all, preds, common_mask = get_clean_predictions_v2(models_dict, test_subset, device)
    
    # 4. 运行主配置评测 (eps=0.2, T=10, seed=42)
    net_a = models_dict['NetA']
    net_b = models_dict['NetB']
    dual_sources = [net_a, net_b]
    
    main_configs = [
        ('FGSM', 'FGSM', net_a),
        ('PGD', 'PGD', net_a),
        ('I-FGSM', 'I-FGSM', net_a),
        ('MI-FGSM', 'MI-FGSM', net_a),
        ('DI-FGSM', 'DI-FGSM', net_a),
        ('TI-FGSM', 'TI-FGSM', net_a),
        ('MDI-FGSM', 'MDI-FGSM', net_a),
        ('TI-MDI-FGSM', 'TI-MDI-FGSM', net_a),
        ('E-I-FGSM', 'I-FGSM', dual_sources),
        ('E-MI-FGSM', 'MI-FGSM', dual_sources),
        ('E-MDI-FGSM', 'MDI-FGSM', dual_sources),
        ('E-TIMDI-FGSM', 'TI-MDI-FGSM', dual_sources),
        ('B-I-FGSM', 'I-FGSM', net_b),
        ('B-TIMDI-FGSM', 'TI-MDI-FGSM', net_b)
    ]
    
    print("\n=== Running Main Attack Configurations (eps=0.2, T=10, seed=42) ===")
    results = {}
    for name, mth, srcs in main_configs:
        res = run_attack_config_v2(name, mth, srcs, models_dict, test_subset,
                                   common_mask, labels_all, eps=0.2, steps=10, seed=42,
                                   device=device, save_dir=results_dir)
        results[name] = res
        
    # 5. 生成图 3-1 训练收敛图
    training_summary_path = os.path.join(ckpt_dir, 'training_summary.json')
    if os.path.exists(training_summary_path):
        plot_fig3_1_val_accuracy_v2(training_summary_path, figures_dir)
        
    # 6. 生成图 4-3 消融图
    plot_fig4_3_ablation_v2(results_dir, figures_dir)
    
    # 7. 生成图 4-5 对抗样本与差值图
    generate_adversarial_examples_v2(models_dict, test_subset, device, figures_dir)
    
    # 8. 运行预算与步数扫描 (图 4-1 与 图 4-2)
    run_budget_and_step_scans_v2(models_dict, test_subset, common_mask, device, figures_dir)
    
    print("\n=== All V2 experiments and figures completed successfully! ===")

if __name__ == '__main__':
    main()
