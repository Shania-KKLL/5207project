import os
import numpy as np
import torch
import timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.transforms import RandAugment

def get_cifar100_subset(data_root, percent, transform=None, seed=42):
    """
    返回 CIFAR-100 训练集的指定百分比子集（分层采样）。

    Args:
        data_root (str): 包含 cifar-100-python 的目录
        percent (int): 百分比，如 1, 10, 20, 50, 100
        transform (callable, optional): 应用于子集的图像变换
        seed (int): 随机种子，保证可重复性

    Returns:
        Subset: torch.utils.data.Subset 对象，可直接用于 DataLoader
    """
    # 加载完整训练集（不下载，因为已有）
    full_train = CIFAR100(root=data_root, train=True, transform=transform, download=True)
    targets = np.array(full_train.targets)
    classes = np.unique(targets)

    # 按类别打乱索引
    np.random.seed(seed)
    indices_per_class = {}
    for c in classes:
        cls_idx = np.where(targets == c)[0]
        np.random.shuffle(cls_idx)
        indices_per_class[c] = cls_idx

    # 按比例取每个类别的前 n 个
    p = percent / 100.0
    subset_idx = []
    for c in classes:
        n_total = len(indices_per_class[c])
        n_select = int(np.ceil(n_total * p)) if p < 1.0 else n_total
        subset_idx.extend(indices_per_class[c][:n_select].tolist())

    # 整体打乱
    np.random.shuffle(subset_idx)
    return Subset(full_train, subset_idx)


def load_model(model_name, device, num_classes=100):
    """
    加载预训练模型（ConvNeXt-Tiny 或 ViT-Small/16），并将分类头替换为 num_classes。
    """
    if model_name == 'convnext':
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
        ckpt_path = './models/convnext_tiny.safetensors'
    elif model_name == 'vit':
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
        ckpt_path = './models/vit_small.safetensors'
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 加载 safetensors 权重
    state_dict = load_file(ckpt_path)

    # 移除所有与分类头相关的 key（避免形状不匹配）
    keys_to_remove = [k for k in state_dict.keys() if k.startswith('head.')]
    for k in keys_to_remove:
        state_dict.pop(k)
        print(f"Removed key: {k}")  # 可选，调试时查看

    # 加载剩余权重（strict=False 允许缺失分类头）
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


#绘图
def get_augmentation(strategy_name):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if strategy_name == 'none':
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    elif strategy_name == 'basic':
        return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    elif strategy_name == 'strong':
        return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), RandAugment(2, 15), transforms.ToTensor(), norm])

def plot_and_save(history_df, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Accuracy Plot
    ax[0].plot(history_df['epoch'], history_df['train_acc'], label='Train Acc')
    ax[0].plot(history_df['epoch'], history_df['val_acc'], label='Val Acc')
    ax[0].set_title(f'Acc: {title}')
    ax[0].legend()
    # Loss Plot
    ax[1].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    ax[1].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
    ax[1].set_title(f'Loss: {title}')
    ax[1].legend()
    plt.savefig(path)
    plt.close()

#训练主逻辑
def run_experiment(model_name, ratio, aug_name, seed, device):
    EPOCHS = 100
    BATCH_SIZE = 128
    WARMUP_EPOCHS = 10
    LR = 5e-5
    
    torch.manual_seed(seed)
    
    #数据流
    train_set = get_cifar100_subset('./data', ratio, get_augmentation(aug_name), seed)
    val_set = CIFAR100(root='./data', train=False, transform=get_augmentation('none'), download=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    #初始化
    model = load_model(model_name, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    #调度器：10轮预热后进入余弦退火
    main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[WARMUP_EPOCHS])
    
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_corr, tr_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_corr += (outputs.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)
        
        scheduler.step()
        
        #验证
        model.eval()
        val_loss, val_corr, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                val_corr += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        #记录每轮指标
        metrics = {
            'epoch': epoch,
            'train_loss': tr_loss / tr_total,
            'train_acc': 100. * tr_corr / tr_total,
            'val_loss': val_loss / val_total,
            'val_acc': 100. * val_corr / val_total
        }
        history.append(metrics)
        
    return pd.DataFrame(history)

#执行 36 组实验
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = ['convnext', 'vit']
    RATIOS = [10, 100]
    AUG_STRATEGIES = ['none', 'basic', 'strong']
    SEEDS = [42, 123, 2024]
    
    os.makedirs("./output/plots", exist_ok=True)
    summary_results = []

    for m in MODELS:
        for r in RATIOS:
            for aug in AUG_STRATEGIES:
                acc_seeds = []
                for seed in SEEDS:
                    exp_name = f"{m}_r{r}_{aug}_s{seed}"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running {exp_name}...")
                    
                    df_history = run_experiment(m, r, aug, seed, DEVICE)
                    
                    #学习曲线图
                    plot_and_save(df_history, exp_name, f"./output/plots/{exp_name}.png")
                    
                    #记录该组最高准确率
                    acc_seeds.append(df_history['val_acc'].max())
                
                # 汇总统计
                summary_results.append({
                    'Model': m, 'Ratio': r, 'Augmentation': aug,
                    'Mean_Acc': np.mean(acc_seeds), 'Std_Acc': np.std(acc_seeds)
                })

    # 输出最终汇总报告
    pd.DataFrame(summary_results).to_csv("experiment_summary.csv", index=False)
    print("All 36 experiments completed. Results and plots saved.")