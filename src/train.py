"""
训练脚本
支持RetinaSAM和EnhanceSegNet的训练
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 可选导入tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
    class tqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.desc = desc
            print(desc)
        
        def __iter__(self):
            return iter(self.iterable)
        
        def set_postfix(self, postfix_dict=None, **kwargs):
            pass

# 可选导入tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    
    class SummaryWriter:
        def __init__(self, log_dir=None):
            pass
        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            pass
        def close(self):
            pass

# 导入自定义模块
from models.retinasam import RetinaSAM
from models.enhancesegnet import EnhanceSegNet, Discriminator
from utils.losses import RetinaSAMLoss, EnhanceSegLoss
from utils.metrics import dice_coefficient, evaluate_multi_class


def parse_args():
    parser = argparse.ArgumentParser(description='训练眼底视网膜分割模型')
    parser.add_argument('--model', type=str, default='retinasam',
                        choices=['retinasam', 'enhancesegnet'],
                        help='选择模型类型')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='类别数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='训练设备')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    return parser.parse_args()


def train_epoch_retinasam(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch
):
    """
    训练RetinaSAM一个epoch
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch in pbar:
        # 数据移到设备
        images = batch['high_quality'].to(device)
        masks = batch['mask'].to(device) if 'mask' in batch else None
        
        # 前向传播
        pred_masks = model(images)
        
        # 计算损失
        if masks is not None:
            loss = criterion(pred_masks, masks)
        else:
            # 如果没有标签，生成伪标签用于演示
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算指标
        total_loss += loss.item()
        
        if masks is not None:
            with torch.no_grad():
                pred_np = torch.softmax(pred_masks, dim=1).cpu().numpy()
                target_np = masks.cpu().numpy()
                
                batch_dice = 0.0
                for i in range(len(pred_np)):
                    pred = pred_np[i].argmax(axis=0)
                    target = target_np[i]
                    batch_dice += dice_coefficient(pred, target)
                
                total_dice += batch_dice / len(pred_np)
        
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'dice': f'{total_dice/max(num_batches,1):.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / max(num_batches, 1)
    
    return avg_loss, avg_dice


def train_epoch_enhancesegnet(
    model,
    discriminator,
    dataloader,
    criterion,
    optimizer_g,
    optimizer_d,
    device,
    epoch
):
    """
    训练EnhanceSegNet一个epoch
    """
    model.train()
    discriminator.train()
    
    total_loss_g = 0.0
    total_loss_d = 0.0
    total_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch in pbar:
        # 数据移到设备
        high_quality = batch['high_quality'].to(device)
        low_quality = batch['low_quality'].to(device)
        masks = batch['mask'].to(device) if 'mask' in batch else None
        
        # ====================
        # 训练判别器
        # ====================
        optimizer_d.zero_grad()
        
        # 真实图像
        d_real = discriminator(high_quality)
        loss_d_real = torch.mean((d_real - 1) ** 2)
        
        # 生成图像
        enhanced, _, _ = model(low_quality)
        d_fake = discriminator(enhanced.detach())
        loss_d_fake = torch.mean(d_fake ** 2)
        
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        optimizer_d.step()
        
        # ====================
        # 训练生成器
        # ====================
        optimizer_g.zero_grad()
        
        enhanced, pred_masks, quality = model(low_quality)
        
        # 判别器损失
        d_fake = discriminator(enhanced)
        
        # 计算损失
        if masks is not None:
            loss_g = criterion(enhanced, high_quality, pred_masks, masks, d_real, d_fake)
        else:
            loss_g = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss_g.backward()
        optimizer_g.step()
        
        # 计算指标
        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        
        if masks is not None:
            with torch.no_grad():
                pred_np = torch.softmax(pred_masks, dim=1).cpu().numpy()
                target_np = masks.cpu().numpy()
                
                batch_dice = 0.0
                for i in range(len(pred_np)):
                    pred = pred_np[i].argmax(axis=0)
                    target = target_np[i]
                    batch_dice += dice_coefficient(pred, target)
                
                total_dice += batch_dice / len(pred_np)
        
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss_g': f'{total_loss_g/num_batches:.4f}',
            'loss_d': f'{total_loss_d/num_batches:.4f}',
            'dice': f'{total_dice/max(num_batches,1):.4f}'
        })
    
    avg_loss_g = total_loss_g / num_batches
    avg_loss_d = total_loss_d / num_batches
    avg_dice = total_dice / max(num_batches, 1)
    
    return avg_loss_g, avg_loss_d, avg_dice


def main():
    args = parse_args()
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    if args.model == 'retinasam':
        model = RetinaSAM(num_classes=args.num_classes).to(device)
        criterion = RetinaSAMLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    elif args.model == 'enhancesegnet':
        model = EnhanceSegNet(num_classes=args.num_classes).to(device)
        discriminator = Discriminator().to(device)
        criterion = EnhanceSegLoss()
        optimizer_g = optim.Adam(model.parameters(), lr=args.lr)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr * 0.1)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # 生成模拟数据（演示用）
    print('注意: 使用模拟数据进行演示')
    print('实际使用时请替换为真实的眼底图像数据集')
    
    # 训练循环
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        if args.model == 'retinasam':
            # 创建模拟数据加载器
            # 实际使用时替换为真实的dataloader
            from data.dataloader import FundusDataset
            from torch.utils.data import DataLoader
            
            # 模拟数据生成器
            def generate_dummy_data():
                for _ in range(10):
                    yield {
                        'high_quality': torch.randn(args.batch_size, 3, 512, 512),
                        'low_quality': torch.randn(args.batch_size, 3, 512, 512),
                        'mask': torch.randint(0, args.num_classes, (args.batch_size, 256, 256))  # 模型输出是256x256
                    }
            
            # 模拟数据加载器
            class DummyLoader:
                def __iter__(self):
                    return generate_dummy_data()
                def __len__(self):
                    return 10
            
            dummy_loader = DummyLoader()
            
            train_loss, train_dice = train_epoch_retinasam(
                model, dummy_loader, criterion, optimizer, device, epoch
            )
            
            # 记录日志
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            
            # 保存检查点
            if train_dice > best_dice:
                best_dice = train_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice
                }, os.path.join(args.checkpoint_dir, f'best_{args.model}.pth'))
        
        elif args.model == 'enhancesegnet':
            # 模拟数据加载器
            from data.dataloader import FundusDataset
            
            dummy_dataset = type('', (), {})()
            dummy_dataset.__len__ = lambda self: 100
            dummy_loader = type('', (), {})()
            dummy_loader.__iter__ = lambda self: iter([{
                'high_quality': torch.randn(args.batch_size, 3, 512, 512),
                'low_quality': torch.randn(args.batch_size, 3, 512, 512),
                'mask': torch.randint(0, args.num_classes, (args.batch_size, 512, 512))
            }])
            dummy_loader.__len__ = lambda self: 10
            
            train_loss_g, train_loss_d, train_dice = train_epoch_enhancesegnet(
                model, discriminator, dummy_loader, criterion,
                optimizer_g, optimizer_d, device, epoch
            )
            
            # 记录日志
            writer.add_scalar('Loss_G/train', train_loss_g, epoch)
            writer.add_scalar('Loss_D/train', train_loss_d, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            
            # 保存检查点
            if train_dice > best_dice:
                best_dice = train_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'dice': best_dice
                }, os.path.join(args.checkpoint_dir, f'best_{args.model}.pth'))
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.checkpoint_dir, f'{args.model}_epoch_{epoch+1}.pth'))
    
    writer.close()
    print('训练完成!')
    print(f'最佳Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
