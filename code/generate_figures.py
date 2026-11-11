import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import NetA, NetB, TargetCNN, TargetMLP
from dataset import get_mnist_data
from attacks import fgsm_attack, iterative_attack
from analyze import analyze_transitions, analyze_dual_target_overlap, paired_bootstrap_diff

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def run_budget_and_step_scans(models_dict, test_subset, common_mask, labels_all, device, out_dir):
    net_a = models_dict['NetA']
    net_b = models_dict['NetB']
    dual = [net_a, net_b]
    
    loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    # 1. 扰动预算扫描: eps in [0.1, 0.2, 0.3], steps=10
    eps_list = [0.1, 0.2, 0.3]
    scan_methods = {
        'FGSM': ('FGSM', net_a),
        'I-FGSM': ('I-FGSM', net_a),
        'MI-FGSM': ('MI-FGSM', net_a),
        'MDI-FGSM': ('MDI-FGSM', net_a),
        'E-MDI-FGSM': ('MDI-FGSM', dual)
    }
    
    budget_results = {m: [] for m in scan_methods}
    for eps in eps_list:
        print(f"Scanning Budget eps={eps}...")
        for m_name, (mth, srcs) in scan_methods.items():
            cnn_wrong = 0
            mlp_wrong = 0
            for b_idx, (x, y) in enumerate(loader):
                torch.manual_seed(42 + 1000003 * b_idx)
                x, y = x.to(device), y.to(device)
                if mth == 'FGSM':
                    x_adv = fgsm_attack(srcs, x, y, eps=eps)
                elif mth == 'I-FGSM':
                    x_adv = iterative_attack(srcs, x, y, eps=eps, steps=10, momentum=False, diversity=False)
                elif mth == 'MI-FGSM':
                    x_adv = iterative_attack(srcs, x, y, eps=eps, steps=10, momentum=True, diversity=False)
                elif mth == 'MDI-FGSM':
                    x_adv = iterative_attack(srcs, x, y, eps=eps, steps=10, momentum=True, diversity=True)
                
                with torch.no_grad():
                    p_cnn = models_dict['TargetCNN'](x_adv).argmax(dim=1)
                    p_mlp = models_dict['TargetMLP'](x_adv).argmax(dim=1)
                    
                b_mask = common_mask[b_idx*100 : (b_idx+1)*100]
                b_mask_t = torch.tensor(b_mask, device=device)
                
                cnn_wrong += ((p_cnn != y) & b_mask_t).sum().item()
                mlp_wrong += ((p_mlp != y) & b_mask_t).sum().item()
                
            total_s = common_mask.sum()
            mean_asr = ((cnn_wrong + mlp_wrong) / (2.0 * total_s)) * 100.0
            budget_results[m_name].append(mean_asr)
            
    # 绘制预算扫描图
    plt.figure(figsize=(7, 5))
    markers = ['o-', 's-', '^-', 'd-', '*-']
    for idx, (m_name, vals) in enumerate(budget_results.items()):
        plt.plot(eps_list, vals, markers[idx], label=m_name, linewidth=2, markersize=7)
    plt.xlabel('Perturbation Budget $\\epsilon$', fontsize=12)
    plt.ylabel('Target Mean ASR (%)', fontsize=12)
    plt.title('Impact of Perturbation Budget on Black-box Transfer ASR', fontsize=13)
    plt.xticks(eps_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_1_budget_scan.png'), dpi=300)
    plt.close()
    print("Saved fig4_1_budget_scan.png")
    
    # 2. 步数扫描: T in [1, 5, 10, 20, 40], eps=0.2
    steps_list = [1, 5, 10, 20, 40]
    step_methods = {
        'I-FGSM (White-box A)': ('I-FGSM', net_a, 'NetA'),
        'I-FGSM (Target Mean)': ('I-FGSM', net_a, 'Mean'),
        'MI-FGSM (Target Mean)': ('MI-FGSM', net_a, 'Mean'),
        'E-MDI-FGSM (Target Mean)': ('MDI-FGSM', dual, 'Mean')
    }
    step_results = {m: [] for m in step_methods}
    for stp in steps_list:
        print(f"Scanning Steps T={stp}...")
        for m_name, (mth, srcs, target_type) in step_methods.items():
            cnt = 0
            for b_idx, (x, y) in enumerate(loader):
                torch.manual_seed(42 + 1000003 * b_idx)
                x, y = x.to(device), y.to(device)
                if mth == 'I-FGSM':
                    x_adv = iterative_attack(srcs, x, y, eps=0.2, steps=stp, momentum=False, diversity=False)
                elif mth == 'MI-FGSM':
                    x_adv = iterative_attack(srcs, x, y, eps=0.2, steps=stp, momentum=True, diversity=False)
                elif mth == 'MDI-FGSM':
                    x_adv = iterative_attack(srcs, x, y, eps=0.2, steps=stp, momentum=True, diversity=True)
                    
                with torch.no_grad():
                    if target_type == 'NetA':
                        p = models_dict['NetA'](x_adv).argmax(dim=1)
                        b_mask = common_mask[b_idx*100 : (b_idx+1)*100]
                        cnt += ((p != y) & torch.tensor(b_mask, device=device)).sum().item()
                    else:
                        p_cnn = models_dict['TargetCNN'](x_adv).argmax(dim=1)
                        p_mlp = models_dict['TargetMLP'](x_adv).argmax(dim=1)
                        b_mask = common_mask[b_idx*100 : (b_idx+1)*100]
                        b_mask_t = torch.tensor(b_mask, device=device)
                        cnt += (((p_cnn != y) & b_mask_t).sum().item() + ((p_mlp != y) & b_mask_t).sum().item()) / 2.0
            total_s = common_mask.sum()
            asr = (cnt / total_s) * 100.0
            step_results[m_name].append(asr)
            
    # 绘制步数扫描图
    plt.figure(figsize=(7, 5))
    plt.plot(steps_list, step_results['I-FGSM (White-box A)'], 'o--', color='gray', label='I-FGSM (White-box A)', linewidth=2)
    plt.plot(steps_list, step_results['I-FGSM (Target Mean)'], 's-', color='tab:blue', label='I-FGSM (Target Mean)', linewidth=2)
    plt.plot(steps_list, step_results['MI-FGSM (Target Mean)'], '^-', color='tab:orange', label='MI-FGSM (Target Mean)', linewidth=2)
    plt.plot(steps_list, step_results['E-MDI-FGSM (Target Mean)'], '*-', color='tab:red', label='E-MDI-FGSM (Target Mean)', linewidth=2)
    plt.xlabel('Number of Iteration Steps $T$', fontsize=12)
    plt.ylabel('Attack Success Rate (%)', fontsize=12)
    plt.title('White-box vs Black-box Transferability under Increasing Steps', fontsize=13)
    plt.xticks(steps_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_2_step_scan.png'), dpi=300)
    plt.close()
    print("Saved fig4_2_step_scan.png")

def generate_adversarial_visualization(models_dict, test_subset, common_mask, labels_all, device, out_dir):
    """生成对抗样本对比图与红蓝差值图 (对应图 4-5)"""
    loader = DataLoader(test_subset, batch_size=50, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    # 挑选 4 个典型样本展示 (包含成功与失败样本)
    indices = [10, 17, 42, 45]
    x_sel = x_batch[indices]
    y_sel = y_batch[indices]
    
    net_a = models_dict['NetA']
    net_b = models_dict['NetB']
    x_adv = iterative_attack([net_a, net_b], x_sel, y_sel, eps=0.2, steps=10, momentum=True, diversity=True)
    
    with torch.no_grad():
        p_cnn = models_dict['TargetCNN'](x_adv).argmax(dim=1).cpu().numpy()
        p_mlp = models_dict['TargetMLP'](x_adv).argmax(dim=1).cpu().numpy()
        
    delta = (x_adv - x_sel).cpu().numpy()
    x_orig_np = x_sel.cpu().numpy()
    x_adv_np = x_adv.cpu().numpy()
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(8, 10))
    for i in range(len(indices)):
        orig_lbl = y_sel[i].item()
        pred_c = p_cnn[i]
        pred_m = p_mlp[i]
        
        # 1. 干净图像
        axes[i, 0].imshow(x_orig_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"Clean (Label: {orig_lbl})", fontsize=11)
        axes[i, 0].axis('off')
        
        # 2. 对抗图像
        axes[i, 1].imshow(x_adv_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Adv (CNN:{pred_c}, MLP:{pred_m})", fontsize=11)
        axes[i, 1].axis('off')
        
        # 3. 扰动图 (红增蓝减)
        im = axes[i, 2].imshow(delta[i, 0], cmap='bwr', vmin=-0.2, vmax=0.2)
        axes[i, 2].set_title(f"Perturbation $\\delta$ ($\\epsilon=0.2$)", fontsize=11)
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_5_adversarial_examples.png'), dpi=300)
    plt.close()
    print("Saved fig4_5_adversarial_examples.png")

def generate_statistical_tables(results_dir, out_dir):
    """计算成对提升 Bootstrap 区间 (表 4-2)、逐样本变化表 (表 4-8) 和双目标重叠表 (表 4-9)"""
    npz_ifgsm = os.path.join(results_dir, 'I-FGSM.npz')
    npz_mifgsm = os.path.join(results_dir, 'MI-FGSM.npz')
    npz_mdifgsm = os.path.join(results_dir, 'MDI-FGSM.npz')
    npz_emdifgsm = os.path.join(results_dir, 'E-MDI-FGSM.npz')
    
    d_i = np.load(npz_ifgsm, allow_pickle=True)
    d_mi = np.load(npz_mifgsm, allow_pickle=True)
    d_mdi = np.load(npz_mdifgsm, allow_pickle=True)
    
    mask = d_i['common_mask']
    labels = d_i['labels'][mask]
    
    # 目标均值布尔成功判定
    succ_i_cnn = (d_i['preds'].item()['TargetCNN'][mask] != labels)
    succ_i_mlp = (d_i['preds'].item()['TargetMLP'][mask] != labels)
    succ_mi_cnn = (d_mi['preds'].item()['TargetCNN'][mask] != labels)
    succ_mi_mlp = (d_mi['preds'].item()['TargetMLP'][mask] != labels)
    succ_mdi_cnn = (d_mdi['preds'].item()['TargetCNN'][mask] != labels)
    succ_mdi_mlp = (d_mdi['preds'].item()['TargetMLP'][mask] != labels)
    
    # 配对 Bootstrap
    mean_diff_mi, low_mi, high_mi = paired_bootstrap_diff(
        (succ_mi_cnn.astype(float) + succ_mi_mlp.astype(float))/2.0,
        (succ_i_cnn.astype(float) + succ_i_mlp.astype(float))/2.0
    )
    mean_diff_mdi, low_mdi, high_mdi = paired_bootstrap_diff(
        (succ_mdi_cnn.astype(float) + succ_mdi_mlp.astype(float))/2.0,
        (succ_i_cnn.astype(float) + succ_i_mlp.astype(float))/2.0
    )
    mean_diff_mdi_mi, low_mdi_mi, high_mdi_mi = paired_bootstrap_diff(
        (succ_mdi_cnn.astype(float) + succ_mdi_mlp.astype(float))/2.0,
        (succ_mi_cnn.astype(float) + succ_mi_mlp.astype(float))/2.0
    )
    
    table_4_2 = {
        'MI-FGSM - I-FGSM': {'diff': mean_diff_mi, 'ci95': [low_mi, high_mi]},
        'MDI-FGSM - I-FGSM': {'diff': mean_diff_mdi, 'ci95': [low_mdi, high_mdi]},
        'MDI-FGSM - MI-FGSM': {'diff': mean_diff_mdi_mi, 'ci95': [low_mdi_mi, high_mdi_mi]}
    }
    
    with open(os.path.join(out_dir, 'table_4_2_paired_boostrap.json'), 'w', encoding='utf-8') as f:
        json.dump(table_4_2, f, indent=2, ensure_ascii=False)
        
    print("\n--- Table 4-2 Paired Bootstrap 95% CI ---")
    for comp, v in table_4_2.items():
        print(f"  {comp:25s}: diff = {v['diff']:+.2f}%, 95% CI = [{v['ci95'][0]:.2f}%, {v['ci95'][1]:.2f}%]")
        
    # 表 4-8 逐样本成功变化
    t4_8 = {}
    for comp_name, (f_base, f_imp) in [('MI-FGSM / I-FGSM', (npz_ifgsm, npz_mifgsm)),
                                       ('MDI-FGSM / MI-FGSM', (npz_mifgsm, npz_mdifgsm)),
                                       ('MDI-FGSM / I-FGSM', (npz_ifgsm, npz_mdifgsm))]:
        t4_8[comp_name] = {}
        for tgt in ['TargetCNN', 'TargetMLP']:
            new_g, lost, net_g = analyze_transitions(f_base, f_imp, tgt)
            t4_8[comp_name][tgt] = {'newly_gained': int(new_g), 'lost': int(lost), 'net_gain': int(net_g)}
            
    with open(os.path.join(out_dir, 'table_4_8_transitions.json'), 'w', encoding='utf-8') as f:
        json.dump(t4_8, f, indent=2, ensure_ascii=False)
        
    print("\n--- Table 4-8 Sample-level Transitions ---")
    for comp, d in t4_8.items():
        for tgt, v in d.items():
            print(f"  {comp:22s} | {tgt:10s}: +{v['newly_gained']} / -{v['lost']} => Net: {v['net_gain']:+d}")
            
    # 表 4-9 双目标重叠分类
    t4_9 = {}
    for m_name, f_path in [('I-FGSM', npz_ifgsm), ('MI-FGSM', npz_mifgsm),
                           ('MDI-FGSM', npz_mdifgsm), ('E-MDI-FGSM', npz_emdifgsm)]:
        both, cnn_only, mlp_only, fail = analyze_dual_target_overlap(f_path)
        t4_9[m_name] = {
            'both_succeed': int(both),
            'only_cnn': int(cnn_only),
            'only_mlp': int(mlp_only),
            'both_fail': int(fail)
        }
        
    with open(os.path.join(out_dir, 'table_4_9_overlap.json'), 'w', encoding='utf-8') as f:
        json.dump(t4_9, f, indent=2, ensure_ascii=False)
        
    print("\n--- Table 4-9 Dual-target Overlap ---")
    for m, v in t4_9.items():
        print(f"  {m:12s}: Both={v['both_succeed']}, OnlyCNN={v['only_cnn']}, OnlyMLP={v['only_mlp']}, Fail={v['both_fail']}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    results_dir = os.path.join(base_dir, 'results')
    figures_dir = os.path.join(base_dir, 'figures')
    course_weight = os.path.join(base_dir, '..', 'resource', 'lenet_mnist_model.pth')
    
    # 准备数据和模型
    _, _, _, test_subset, _ = get_mnist_data(data_dir=data_dir, seed=2026)
    from evaluate import load_models, get_clean_predictions
    models_dict = load_models(ckpt_dir, course_weight, device)
    labels_all, _, common_mask = get_clean_predictions(models_dict, test_subset, device)
    
    # 1. 运行预算与步数扫描，并绘制图 4-1 与图 4-2
    run_budget_and_step_scans(models_dict, test_subset, common_mask, labels_all, device, figures_dir)
    
    # 2. 生成对抗样本示例图 4-5
    generate_adversarial_visualization(models_dict, test_subset, common_mask, labels_all, device, figures_dir)
    
    # 3. 计算统计分析表 (表 4-2, 表 4-8, 表 4-9)
    generate_statistical_tables(results_dir, figures_dir)
    
    print("\nAll figures and statistical tables generated successfully!")

if __name__ == '__main__':
    main()
