import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models_v2 import NetA, NetB_Modern, TargetCNN_Deep, TargetMLP_Deep, TargetResNet_Mini
from dataset_v2 import get_mnist_data_v2
from attacks_v2 import fgsm_attack, advanced_iterative_attack
from analyze import wilson_score_interval, paired_bootstrap_diff

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_fig3_1_val_accuracy_v2(training_summary_path, out_dir):
    """图 3-1: 四个新增模型在验证集上的准确率变化 (10 轮)"""
    with open(training_summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    epochs = list(range(1, 11))
    val_b = data['NetB_Modern']['val_history']
    val_cnn = data['TargetCNN_Deep']['val_history']
    val_mlp = data['TargetMLP_Deep']['val_history']
    val_res = data['TargetResNet_Mini']['val_history']

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_cnn, 'o-', color='#4C72B0', label=f"Target CNN-Deep (Best: {data['TargetCNN_Deep']['best_val_acc']}%)", linewidth=2)
    plt.plot(epochs, val_b, 's-', color='#55A868', label=f"Source B-Modern (Best: {data['NetB_Modern']['best_val_acc']}%)", linewidth=2)
    plt.plot(epochs, val_res, 'd-', color='#8172B3', label=f"Target ResNet-Mini (Best: {data['TargetResNet_Mini']['best_val_acc']}%)", linewidth=2)
    plt.plot(epochs, val_mlp, '^-', color='#DD8452', label=f"Target MLP-Deep (Best: {data['TargetMLP_Deep']['best_val_acc']}%)", linewidth=2)

    plt.axvline(x=6, color='gray', linestyle=':', label='LR Decay (x0.3 at Epoch 6)')
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy Dynamics of Heterogeneous Models', fontsize=13)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_1_val_accuracy_v2.png'), dpi=300)
    plt.close()
    print("Saved fig3_1_val_accuracy_v2.png")

def plot_fig4_3_ablation_v2(results_dir, out_dir):
    """图 4-3: 全因子消融对比图 (涵盖 3 个黑盒目标模型及平均值)"""
    methods = ['I-FGSM', 'MI-FGSM', 'DI-FGSM', 'TI-FGSM', 'MDI-FGSM', 'TI-MDI-FGSM',
               'E-I-FGSM', 'E-MI-FGSM', 'E-MDI-FGSM', 'E-TIMDI-FGSM']
               
    cnn_asrs, mlp_asrs, res_asrs, mean_asrs = [], [], [], []
    valid_methods = []
    
    for m in methods:
        p = os.path.join(results_dir, f"{m}.json")
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
                valid_methods.append(m)
                cnn_asrs.append(d['asrs']['TargetCNN'])
                mlp_asrs.append(d['asrs']['TargetMLP'])
                res_asrs.append(d['asrs']['TargetResNet'])
                mean_asrs.append(d['asrs']['TargetMean'])
                
    x = np.arange(len(valid_methods))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5*width, cnn_asrs, width, label='Target CNN-Deep', color='#4C72B0')
    ax.bar(x - 0.5*width, res_asrs, width, label='Target ResNet-Mini', color='#8172B3')
    ax.bar(x + 0.5*width, mlp_asrs, width, label='Target MLP-Deep', color='#DD8452')
    ax.bar(x + 1.5*width, mean_asrs, width, label='Target Mean ASR', color='#C44E52')
    
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('Cross-Architecture Black-box Transferability Factorial Ablation (eps=0.2, T=10)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=25, ha='right', fontsize=10.5)
    ax.legend(fontsize=10.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_3_ablation_v2.png'), dpi=300)
    plt.close()
    print("Saved fig4_3_ablation_v2.png")

def generate_adversarial_examples_v2(models_dict, test_subset, device, out_dir):
    """图 4-5: 代表性样本的干净图像、对抗图像与红蓝差值图"""
    loader = DataLoader(test_subset, batch_size=50, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    indices = [3, 7, 12, 18] # 4 个具有代表性的手写数字
    x_sel = x_batch[indices]
    y_sel = y_batch[indices]
    
    net_a = models_dict['NetA']
    net_b = models_dict['NetB']
    
    x_adv = advanced_iterative_attack([net_a, net_b], x_sel, y_sel, eps=0.2, steps=10,
                                       momentum=True, diversity=True, translation_inv=True)
                                       
    with torch.no_grad():
        p_cnn = models_dict['TargetCNN'](x_adv).argmax(dim=1).cpu().numpy()
        p_mlp = models_dict['TargetMLP'](x_adv).argmax(dim=1).cpu().numpy()
        p_res = models_dict['TargetResNet'](x_adv).argmax(dim=1).cpu().numpy()
        
    delta = (x_adv - x_sel).cpu().numpy()
    x_orig_np = x_sel.cpu().numpy()
    x_adv_np = x_adv.cpu().numpy()
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(8, 10))
    for i in range(len(indices)):
        orig_lbl = y_sel[i].item()
        c_p = p_cnn[i]
        m_p = p_mlp[i]
        r_p = p_res[i]
        
        axes[i, 0].imshow(x_orig_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"Clean (Label: {orig_lbl})", fontsize=11)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(x_adv_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Adv (CNN:{c_p}, MLP:{m_p}, Res:{r_p})", fontsize=10)
        axes[i, 1].axis('off')
        
        im = axes[i, 2].imshow(delta[i, 0], cmap='bwr', vmin=-0.2, vmax=0.2)
        axes[i, 2].set_title(f"Perturbation delta (eps=0.2)", fontsize=11)
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_5_adversarial_examples_v2.png'), dpi=300)
    plt.close()
    print("Saved fig4_5_adversarial_examples_v2.png")
