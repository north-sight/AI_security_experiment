import os
import sys
import torch
from dataset import get_mnist_data
from models import NetA, NetB, TargetCNN, TargetMLP
from train import train_model, evaluate as eval_acc
from evaluate import load_models, get_clean_predictions, run_attack_config
from analyze import (wilson_score_interval, paired_bootstrap_diff,
                     analyze_transitions, analyze_dual_target_overlap, plot_ablation_summary)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running AI Security Black-Box Transfer Experiment ===")
    print(f"Device: {device}")
    
    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    results_dir = os.path.join(base_dir, 'results')
    figures_dir = os.path.join(base_dir, 'figures')
    course_weight = os.path.join(base_dir, '..', 'resource', 'lenet_mnist_model.pth')
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. 准备数据
    train_set, val_set, test_full, test_subset, sampled_indices = get_mnist_data(data_dir=data_dir, seed=2026)
    
    # 2. 准备模型 (检查点不存在则训练)
    net_b_path = os.path.join(ckpt_dir, 'net_b.pth')
    target_cnn_path = os.path.join(ckpt_dir, 'target_cnn.pth')
    target_mlp_path = os.path.join(ckpt_dir, 'target_mlp.pth')
    
    if not os.path.exists(net_b_path):
        print("Training NetB...")
        train_model(NetB, 1101, net_b_path, train_set, val_set, test_full, device)
    if not os.path.exists(target_cnn_path):
        print("Training TargetCNN...")
        train_model(TargetCNN, 2202, target_cnn_path, train_set, val_set, test_full, device)
    if not os.path.exists(target_mlp_path):
        print("Training TargetMLP...")
        train_model(TargetMLP, 3303, target_mlp_path, train_set, val_set, test_full, device)
        
    models_dict = load_models(ckpt_dir, course_weight, device)
    
    # 3. 干净测试与共同正确集合 S 筛选
    labels_all, preds, common_mask = get_clean_predictions(models_dict, test_subset, device)
    
    # 4. 主配置评测 (eps=0.2, T=10, seed=42)
    net_a = models_dict['NetA']
    net_b = models_dict['NetB']
    dual_sources = [net_a, net_b]
    
    main_configs = [
        ('FGSM', 'FGSM', net_a),
        ('PGD', 'PGD', net_a),
        ('I-FGSM', 'I-FGSM', net_a),
        ('MI-FGSM', 'MI-FGSM', net_a),
        ('DI-FGSM', 'DI-FGSM', net_a),
        ('MDI-FGSM', 'MDI-FGSM', net_a),
        ('E-I-FGSM', 'I-FGSM', dual_sources),
        ('E-MI-FGSM', 'MI-FGSM', dual_sources),
        ('E-DI-FGSM', 'DI-FGSM', dual_sources),
        ('E-MDI-FGSM', 'MDI-FGSM', dual_sources),
    ]
    
    print("\n=== Running Main Attack Configurations (eps=0.2, T=10, seed=42) ===")
    results = {}
    for name, mth, srcs in main_configs:
        res = run_attack_config(name, mth, srcs, models_dict, test_subset,
                                common_mask, labels_all, eps=0.2, steps=10, seed=42,
                                device=device, save_dir=results_dir)
        results[name] = res
        
    # 5. 事后源模型 B 独立攻击对照 (解释双源集成现象)
    print("\n=== Running Source B Control Attacks (eps=0.2, T=10, seed=42) ===")
    b_configs = [
        ('B-I-FGSM', 'I-FGSM', net_b),
        ('B-MI-FGSM', 'MI-FGSM', net_b),
        ('B-MDI-FGSM', 'MDI-FGSM', net_b)
    ]
    for name, mth, srcs in b_configs:
        run_attack_config(name, mth, srcs, models_dict, test_subset,
                          common_mask, labels_all, eps=0.2, steps=10, seed=42,
                          device=device, save_dir=results_dir)
                          
    # 6. 生成分析图表
    plot_ablation_summary(results_dir=results_dir, out_dir=figures_dir)
    print("\nExperiment completed successfully!")

if __name__ == '__main__':
    main()
