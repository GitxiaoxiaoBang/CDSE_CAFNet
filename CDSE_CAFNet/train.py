import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset         
from CDSE_CAFNet import CDSE_CAFNet
import logging

# ---------------- 参数解析 ----------------
parser = argparse.ArgumentParser("CDSE_CAFNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--dinov2_path", type=str, required=True,
                    help="path to the pretrained dinov2")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=50,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=5e-4)
args = parser.parse_args()

# ---------------- 损失函数 ----------------
def structure_loss(pred, mask):
    weit  = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# ---------------- 随机种子 ----------------
def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---------------- 主函数 ----------------
def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, 'train_clean.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'),
                  logging.StreamHandler()]
    )

    seed_torch(1024)

    # 数据集
    train_dataset = FullDataset(
        image_root=args.train_image_path,
        gt_root=args.train_mask_path,
        size=1024,
        mode='train'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDSE_CAFNet(args.hiera_path, args.dinov2_path).to(device)

    # 优化器 & 调度器
    optim = opt.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "initial_lr": args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    # 训练循环
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            images    = batch['image'].to(device)
            seg_target= batch['label'].to(device)

            optim.zero_grad()
            seg_pred  = model(images)            # 单输出
            loss      = structure_loss(seg_pred, seg_target)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if i % 50 == 0:
                logging.info(f"Epoch: {epoch + 1}/{args.epoch}, "
                             f"Batch: {i + 1}/{len(train_loader)}, "
                             f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch: {epoch + 1}/{args.epoch}, Average Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(args.save_path, f'CDSE_CAFNet-{epoch + 1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"[Saving] {ckpt_path}")
        scheduler.step()

if __name__ == "__main__":
    main(args)