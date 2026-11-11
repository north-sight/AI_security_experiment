import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import NetA, NetB, TargetCNN, TargetMLP
from dataset import get_mnist_data

def evaluate(model, loader, device):
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

def train_model(model_cls, init_seed, save_path, train_set, val_set, test_full, device):
    torch.manual_seed(init_seed)
    model = model_cls().to(device)
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    
    print(f"\n--- Training {model_cls.__name__} (Seed {init_seed}) ---")
    for epoch in range(1, 11):
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
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.3
                print(f"Epoch {epoch}: Decaying LR to {param_group['lr']:.5f}")
                
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}/10 | Train Loss: {total_loss/len(train_set):.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            best_epoch = epoch
            
    print(f"Best Val Acc: {best_val_acc*100:.2f}% at Epoch {best_epoch}")
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    
    test_acc = evaluate(model, test_loader, device)
    print(f"Official 10k Test Accuracy for {model_cls.__name__}: {test_acc*100:.2f}%")
    return test_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_set, val_set, test_full, _, _ = get_mnist_data()
    
    # 1. 验证课程提供的预训练源模型 A
    net_a = NetA().to(device)
    course_weight = os.path.join(os.path.dirname(__file__), '..', 'resource', 'lenet_mnist_model.pth')
    net_a.load_state_dict(torch.load(course_weight, map_location=device))
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False)
    acc_a = evaluate(net_a, test_loader, device)
    print(f"Course Source NetA 10k Test Accuracy: {acc_a*100:.2f}%")
    
    # 2. 训练源模型 B (NetB, 种子 1101)
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    train_model(NetB, 1101, os.path.join(ckpt_dir, 'net_b.pth'), train_set, val_set, test_full, device)
    
    # 3. 训练目标模型 TargetCNN (种子 2202)
    train_model(TargetCNN, 2202, os.path.join(ckpt_dir, 'target_cnn.pth'), train_set, val_set, test_full, device)
    
    # 4. 训练目标模型 TargetMLP (种子 3303)
    train_model(TargetMLP, 3303, os.path.join(ckpt_dir, 'target_mlp.pth'), train_set, val_set, test_full, device)

if __name__ == '__main__':
    main()
