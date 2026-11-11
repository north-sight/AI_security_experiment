import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from models_v2 import NetA, NetB_Modern, TargetCNN_Deep, TargetMLP_Deep, TargetResNet_Mini
from dataset_v2 import get_mnist_data_v2
from attacks_v2 import pgd_attack, advanced_iterative_attack
from evaluate_v2 import load_models_v2, get_clean_predictions_v2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, 'checkpoints_v2')
    figures_dir = os.path.join(base_dir, 'figures_v2')
    results_dir = os.path.join(base_dir, 'results_v2')
    course_weight = os.path.join(base_dir, '..', 'resource', 'lenet_mnist_model.pth')

    _, _, _, test_subset, _ = get_mnist_data_v2(seed=1024)
    models_dict = load_models_v2(ckpt_dir, course_weight, device)
    labels_all, _, common_mask = get_clean_predictions_v2(models_dict, test_subset, device)

    loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    net_a = models_dict['NetA']
    dual = [net_a, models_dict['NetB']]

    eval_methods = {
        'PGD': ('PGD', net_a, False, False, False),
        'DI-FGSM': ('DI-FGSM', net_a, False, True, False),
        'MDI-FGSM': ('MDI-FGSM', net_a, True, True, False),
        'TI-MDI-FGSM': ('TI-MDI-FGSM', net_a, True, True, True),
        'E-TIMDI-FGSM': ('E-TIMDI-FGSM', dual, True, True, True)
    }

    seed_results = {m: {'CNN': [], 'MLP': [], 'ResNet': [], 'Mean': []} for m in eval_methods}

    # 读取种子 42
    for m in eval_methods:
        with open(os.path.join(results_dir, f"{m}.json"), 'r', encoding='utf-8') as f:
            d = json.load(f)
        asrs = d['asrs']
        seed_results[m]['CNN'].append(asrs['TargetCNN'])
        seed_results[m]['MLP'].append(asrs['TargetMLP'])
        seed_results[m]['ResNet'].append(asrs['TargetResNet'])
        seed_results[m]['Mean'].append(asrs['TargetMean'])

    # 运行种子 43 和 44
    for seed in [43, 44]:
        print(f"Running Seed {seed}...")
        for m_name, (mth, srcs, mom, div, ti) in eval_methods.items():
            c_w, m_w, r_w = 0, 0, 0
            for b_idx, (x, y) in enumerate(loader):
                batch_seed = seed + 1000003 * b_idx
                torch.manual_seed(batch_seed)
                np.random.seed(batch_seed)
                x, y = x.to(device), y.to(device)
                if mth == 'PGD':
                    x_adv = pgd_attack(srcs, x, y, eps=0.2, steps=10)
                else:
                    x_adv = advanced_iterative_attack(srcs, x, y, eps=0.2, steps=10,
                                                      momentum=mom, diversity=div, translation_inv=ti)
                with torch.no_grad():
                    pc = models_dict['TargetCNN'](x_adv).argmax(dim=1)
                    pm = models_dict['TargetMLP'](x_adv).argmax(dim=1)
                    pr = models_dict['TargetResNet'](x_adv).argmax(dim=1)
                b_mask = torch.tensor(common_mask[b_idx*100 : (b_idx+1)*100], device=device)
                c_w += ((pc != y) & b_mask).sum().item()
                m_w += ((pm != y) & b_mask).sum().item()
                r_w += ((pr != y) & b_mask).sum().item()
            tot = common_mask.sum()
            ac = round((c_w / tot) * 100.0, 2)
            am = round((m_w / tot) * 100.0, 2)
            ar = round((r_w / tot) * 100.0, 2)
            amean = round((ac + am + ar) / 3.0, 2)
            seed_results[m_name]['CNN'].append(ac)
            seed_results[m_name]['MLP'].append(am)
            seed_results[m_name]['ResNet'].append(ar)
            seed_results[m_name]['Mean'].append(amean)

    # 汇总均值与样本标准差
    table_4_5 = {}
    print("\n--- Table 4-5 Seed Repetition V2 (Mean +/- Std) ---")
    for m in eval_methods:
        vals = seed_results[m]['Mean']
        table_4_5[m] = {
            'runs': vals,
            'summary': f"{np.mean(vals):.2f} +/- {np.std(vals, ddof=1):.2f}",
            'CNN_summary': f"{np.mean(seed_results[m]['CNN']):.2f} +/- {np.std(seed_results[m]['CNN'], ddof=1):.2f}",
            'MLP_summary': f"{np.mean(seed_results[m]['MLP']):.2f} +/- {np.std(seed_results[m]['MLP'], ddof=1):.2f}",
            'ResNet_summary': f"{np.mean(seed_results[m]['ResNet']):.2f} +/- {np.std(seed_results[m]['ResNet'], ddof=1):.2f}"
        }
        print(f"  {m:14s}: Mean ASR = {table_4_5[m]['summary']}")

    with open(os.path.join(figures_dir, 'table_4_5_seeds_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(table_4_5, f, indent=2, ensure_ascii=False)
    print("Saved table_4_5_seeds_v2.json")

if __name__ == '__main__':
    main()
