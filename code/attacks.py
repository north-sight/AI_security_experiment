import torch
import torch.nn as nn
import torch.nn.functional as F

def input_diversity(x, p=0.7, low=24, high=28):
    """
    输入多样性变换 (DI)：
    以概率 p 将图像双线性插值缩放到 [low, high] 之间的随机整数尺寸，
    并在随机位置零填充 (黑色背景) 回 28x28。
    同一批次共享本次变换。
    """
    if torch.rand(1).item() > p:
        return x
    
    img_size = x.shape[-1] # 28
    rnd_size = int(torch.randint(low, high + 1, (1,)).item())
    rescaled = F.interpolate(x, size=(rnd_size, rnd_size), mode='bilinear', align_corners=False)
    
    h_rem = img_size - rnd_size
    w_rem = img_size - rnd_size
    pad_top = int(torch.randint(0, h_rem + 1, (1,)).item())
    pad_bottom = h_rem - pad_top
    pad_left = int(torch.randint(0, w_rem + 1, (1,)).item())
    pad_right = w_rem - pad_left
    
    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return padded

def compute_loss(models, x, y):
    """
    支持单模型或多模型等权损失计算
    Lens = (1/N) * sum_i L(f_i(x), y)
    """
    if isinstance(models, (list, tuple)):
        total_loss = 0.0
        for m in models:
            logits = m(x)
            total_loss = total_loss + F.cross_entropy(logits, y)
        return total_loss / len(models)
    else:
        logits = models(x)
        return F.cross_entropy(logits, y)

def fgsm_attack(models, x_orig, y, eps=0.2):
    """单步 FGSM 攻击基线"""
    if eps <= 0:
        return x_orig.clone().detach()
    
    x = x_orig.clone().detach().requires_grad_(True)
    loss = compute_loss(models, x, y)
    loss.backward()
    
    grad_sign = x.grad.detach().sign()
    x_adv = x_orig + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()

def pgd_attack(models, x_orig, y, eps=0.2, steps=10, alpha=None):
    """随机起点 PGD 攻击基线 (强白盒基线)"""
    if eps <= 0:
        return x_orig.clone().detach()
    if alpha is None:
        alpha = 2.0 * eps / max(steps, 1)
        
    # 随机均匀初始化
    noise = torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_orig + noise, 0.0, 1.0)
    
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = compute_loss(models, x_adv, y)
        loss.backward()
        
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + alpha * grad_sign
        # 投影到 L_infinity 球与 [0, 1] 像素空间
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    return x_adv.detach()

def iterative_attack(models, x_orig, y, eps=0.2, steps=10, alpha=None,
                     momentum=False, diversity=False, mu=1.0, p_div=0.7):
    """
    统一的迭代攻击函数，覆盖 8 种二值因素配置：
    - momentum (动量): False (I-FGSM) / True (MI-FGSM, mu=1.0)
    - diversity (输入多样性): False / True (DI-FGSM, p=0.7)
    - models (源模型): 单模型 A 或 双模型 [A, B]
    """
    if eps <= 0:
        return x_orig.clone().detach()
    if alpha is None:
        alpha = float(eps) / float(max(steps, 1))
        
    x_adv = x_orig.clone().detach()
    g = torch.zeros_like(x_orig)
    
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        
        # 输入多样性变换
        if diversity:
            x_in = input_diversity(x_adv, p=p_div)
        else:
            x_in = x_adv
            
        loss = compute_loss(models, x_in, y)
        loss.backward()
        
        grad = x_adv.grad.detach()
        # 按图像通道、高、宽计算均值归一化 (保留批量维度独立)
        grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True)
        grad_normalized = grad / (grad_norm + 1e-12)
        
        if momentum:
            g = mu * g + grad_normalized
            direction = g.sign()
        else:
            direction = grad.sign()
            
        x_adv = x_adv + alpha * direction
        # 两层投影约束：先限制在 L_infinity 扰动邻域，再限制在 [0, 1] 像素空间
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    return x_adv.detach()
