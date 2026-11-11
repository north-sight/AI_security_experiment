import os
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from models_v2 import NetA, NetB_Modern, TargetCNN_Deep, TargetMLP_Deep, TargetResNet_Mini
from dataset_v2 import get_mnist_data_v2
from attacks_v2 import fgsm_attack, pgd_attack, advanced_iterative_attack

def load_models_v2(ckpt_dir, course_weight, device):
    net_a = NetA().to(device)
    net_a.load_state_dict(torch.load(course_weight, map_location=device))
    net_a.eval()
    
    net_b = NetB_Modern().to(device)
    net_b.load_state_dict(torch.load(os.path.join(ckpt_dir, 'net_b_modern.pth'), map_location=device))
    net_b.eval()
    
    target_cnn = TargetCNN_Deep().to(device)
    target_cnn.load_state_dict(torch.load(os.path.join(ckpt_dir, 'target_cnn_deep.pth'), map_location=device))
    target_cnn.eval()
    
    target_mlp = TargetMLP_Deep().to(device)
    target_mlp.load_state_dict(torch.load(os.path.join(ckpt_dir, 'target_mlp_deep.pth'), map_location=device))
    target_mlp.eval()
    
    target_res = TargetResNet_Mini().to(device)
    target_res.load_state_dict(torch.load(os.path.join(ckpt_dir, 'target_resnet_mini.pth'), map_location=device))
    target_res.eval()
    
    for m in [net_a, net_b, target_cnn, target_mlp, target_res]:
        for p in m.parameters():
            p.requires_grad = False
            
    return {
        'NetA': net_a,
        'NetB': net_b,
        'TargetCNN': target_cnn,
        'TargetMLP': target_mlp,
        'TargetResNet': target_res
    }

def get_clean_predictions_v2(models_dict, test_subset, device):
    loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    preds = {name: [] for name in models_dict}
    labels_all = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            labels_all.append(y.cpu().numpy())
            for name, m in models_dict.items():
                p = m(x).argmax(dim=1).cpu().numpy()
                preds[name].append(p)
                
    labels_all = np.concatenate(labels_all)
    for name in preds:
        preds[name] = np.concatenate(preds[name])
        
    correct_masks = [preds[name] == labels_all for name in models_dict]
    common_correct_mask = np.logical_and.reduce(correct_masks)
    
    print("\n--- Clean Evaluation on 2,000 Stratified Samples (Seed 1024) ---")
    for name in models_dict:
        corr_count = (preds[name] == labels_all).sum()
        acc = corr_count / len(labels_all)
        print(f"  {name:15s} Clean Correct: {corr_count}/2000 ({acc*100:.2f}%)")
    print(f"  Common Correct Set S size: {common_correct_mask.sum()}/2000 ({common_correct_mask.mean()*100:.2f}%)")
    
    return labels_all, preds, common_correct_mask

def run_attack_config_v2(config_name, method, source_models, models_dict, test_subset,
                         common_correct_mask, labels_all, eps=0.2, steps=10, seed=42, device='cpu',
                         save_dir='./results_v2'):
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    adv_preds = {name: [] for name in models_dict}
    l_inf_list = []
    l_2_list = []
    
    start_time = time.time()
    
    for b_idx, (x, y) in enumerate(loader):
        batch_seed = seed + 1000003 * b_idx
        torch.manual_seed(batch_seed)
        np.random.seed(batch_seed)
        
        x, y = x.to(device), y.to(device)
        
        if method == 'FGSM':
            x_adv = fgsm_attack(source_models, x, y, eps=eps)
        elif method == 'PGD':
            x_adv = pgd_attack(source_models, x, y, eps=eps, steps=steps)
        elif method == 'I-FGSM':
            x_adv = advanced_iterative_attack(source_models, x, y, eps=eps, steps=steps,
                                              momentum=False, diversity=False, translation_inv=False)
        elif method == 'MI-FGSM':
            x_adv = advanced_iterative_attack(source_models, x, y, eps=eps, steps=steps,
                                              momentum=True, diversity=False, translation_inv=False)
        elif method == 'DI-FGSM':
            x_adv = advanced_iterative_attack(source_models, x, y, eps=eps, steps=steps,
                                              momentum=False, diversity=True, translation_inv=False)
        elif method == 'TI-FGSM':
            x_adv = advanced_iterative_attack(source_models, x, y, eps=eps, steps=steps,
                                              momentum=False, diversity=False, translation_inv=True)
        elif method == 'MDI-FGSM':
            x_adv = advanced_iterative_attack(source_models, x, y, eps=eps, steps=steps,
                                              momentum=True, diversity=True, translation_inv=False)
        elif method == 'TI-MDI-FGSM':
            x_adv = advanced_iterative_attack(source_models, x, y, eps=eps, steps=steps,
                                              momentum=True, diversity=True, translation_inv=True)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        diff = (x_adv - x).abs()
        max_diff = diff.view(diff.size(0), -1).max(dim=1)[0].cpu().numpy()
        l2_diff = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1).cpu().numpy()
        l_inf_list.extend(max_diff)
        l_2_list.extend(l2_diff)
        
        with torch.no_grad():
            for name, m in models_dict.items():
                p = m(x_adv).argmax(dim=1).cpu().numpy()
                adv_preds[name].append(p)
                
    elapsed = time.time() - start_time
    ms_per_img = (elapsed / len(test_subset)) * 1000.0
    
    for name in adv_preds:
        adv_preds[name] = np.concatenate(adv_preds[name])
        
    l_inf_arr = np.array(l_inf_list)
    l_2_arr = np.array(l_2_list)
    
    asrs = {}
    for name in models_dict:
        wrong_mask = (adv_preds[name] != labels_all) & common_correct_mask
        asr = (wrong_mask.sum() / common_correct_mask.sum()) * 100.0
        asrs[name] = float(asr)
        
    # 三个黑盒目标模型的平均 ASR
    target_mean = (asrs['TargetCNN'] + asrs['TargetMLP'] + asrs['TargetResNet']) / 3.0
    asrs['TargetMean'] = float(target_mean)
    
    result_record = {
        'config_name': config_name,
        'method': method,
        'eps': eps,
        'steps': steps,
        'seed': seed,
        'ms_per_img': ms_per_img,
        'max_l_inf': float(l_inf_arr.max()),
        'mean_l2': float(l_2_arr.mean()),
        'asrs': asrs
    }
    
    np.savez_compressed(
        os.path.join(save_dir, f"{config_name}.npz"),
        labels=labels_all,
        common_mask=common_correct_mask,
        preds=adv_preds,
        l_inf=l_inf_arr,
        l2=l_2_arr
    )
    with open(os.path.join(save_dir, f"{config_name}.json"), 'w', encoding='utf-8') as f:
        json.dump(result_record, f, indent=2, ensure_ascii=False)
        
    print(f"[{config_name:13s}] NetA:{asrs['NetA']:5.2f}% | NetB:{asrs['NetB']:5.2f}% | "
          f"CNN:{asrs['TargetCNN']:5.2f}%, MLP:{asrs['TargetMLP']:5.2f}%, ResNet:{asrs['TargetResNet']:5.2f}% "
          f"| TargetMean: {target_mean:5.2f}% | L2: {l_2_arr.mean():.3f}")
    return result_record
