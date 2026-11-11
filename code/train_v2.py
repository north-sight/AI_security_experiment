import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models_v2 import NetA, NetB_Modern, TargetCNN_Deep, TargetMLP_Deep, TargetResNet_Mini
from dataset_v2 import get_mnist_data_v2

def evaluate_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def train_single_model(model_cls, init_seed, save_path, train_set, val_set, test_full, device, epochs=10):
    torch.manual_seed(init_seed)
    model = model_cls().to(device)
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    val_history = []
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    
    print(f"\n==========================================")
    print(f"Training {model_cls.__name__} (Seed {init_seed})")
    print(f"==========================================")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            
        if epoch == 6:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.3
                print(f"  [Epoch {epoch}] Decaying LR to {pg['lr']:.5f}")
                
        val_acc = evaluate_acc(model, val_loader, device)
        val_history.append(round(val_acc * 100.0, 2))
        print(f"  Epoch {epoch:2d}/{epochs} | Train Loss: {total_loss/len(train_set):.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            best_epoch = epoch
            
    print(f"  >>> Best Val Acc: {best_val_acc*100:.2f}% at Epoch {best_epoch}")
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    
    test_acc = evaluate_acc(model, test_loader, device)
    print(f"  >>> Official 10k Test Accuracy for {model_cls.__name__}: {test_acc*100:.2f}%")
    
    return {
        'model_name': model_cls.__name__,
        'init_seed': init_seed,
        'val_history': val_history,
        'best_val_acc': round(best_val_acc * 100.0, 2),
        'best_epoch': best_epoch,
        'test_acc': round(test_acc * 100.0, 2)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Environment Device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, 'checkpoints_v2')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    train_set, val_set, test_full, _, _ = get_mnist_data_v2(seed=1024)
    
    # 验证课程预训练 NetA
    net_a = NetA().to(device)
    course_weight = os.path.join(base_dir, '..', 'resource', 'lenet_mnist_model.pth')
    net_a.load_state_dict(torch.load(course_weight, map_location=device))
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False)
    acc_a = evaluate_acc(net_a, test_loader, device)
    print(f"Verified Course NetA 10k Test Accuracy: {acc_a*100:.2f}%")
    
    results = {'NetA': {'test_acc': round(acc_a * 100.0, 2)}}
    
    # 训练 4 款全新架构模型
    configs = [
        (NetB_Modern, 501, os.path.join(ckpt_dir, 'net_b_modern.pth')),
        (TargetCNN_Deep, 602, os.path.join(ckpt_dir, 'target_cnn_deep.pth')),
        (TargetMLP_Deep, 703, os.path.join(ckpt_dir, 'target_mlp_deep.pth')),
        (TargetResNet_Mini, 804, os.path.join(ckpt_dir, 'target_resnet_mini.pth'))
    ]
    
    for cls, seed, path in configs:
        info = train_single_model(cls, seed, path, train_set, val_set, test_full, device)
        results[cls.__name__] = info
        
    with open(os.path.join(ckpt_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nAll models trained and checkpoints saved successfully in checkpoints_v2/!")

if __name__ == '__main__':
    main()
