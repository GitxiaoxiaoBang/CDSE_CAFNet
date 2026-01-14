
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from CDSE_CAFNet import CDSE_CAFNet
from dataset import TestDataset

# -------------------- 参数解析 --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",      type=str, required=True,
                    help="path to the checkpoint of sam2-unext")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path",    type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path",       type=str, required=True,
                    help="root folder to save seg results")
args = parser.parse_args()

# -------------------- 目录准备 --------------------
seg_dir = os.path.join(args.save_path, 'seg')
os.makedirs(seg_dir, exist_ok=True)

# -------------------- 初始化 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_set = TestDataset(args.test_image_path, args.test_gt_path, 1024)

model = CDSE_CAFNet().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
model.eval()

# -------------------- 推理 --------------------
for _ in range(len(test_set)):
    with torch.no_grad():
        image, gt, name = test_set.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)

        seg_out = model(image)

        # resize 回原图尺寸
        seg_out = F.interpolate(seg_out, size=gt.shape, mode='bilinear', align_corners=False)

        # sigmoid → [0,255]
        seg_np = (seg_out.sigmoid().cpu().numpy().squeeze() * 255).astype(np.uint8)

        # 保存
        base_name = os.path.splitext(name)[0] + '.png'
        seg_path = os.path.join(seg_dir, base_name)

        imageio.imwrite(seg_path, seg_np, format='PNG')

        print(f"Saved {base_name} -> {seg_path}")