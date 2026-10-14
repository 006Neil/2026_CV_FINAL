import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.ops as ops  # 🚨 導入官方低顯存可形變捲積庫
from yuvProc import getOneFrame
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FUSION_MODEL_PATH = "joint_fusion_net_test22.pth"
REFINE_MODEL_PATH = "flow_refine_net_test22.pth"
PRECOMPUTED_DIR = "./precomputed_flow_pt_gpu"

class FlowRefineNet(nn.Module):
    def __init__(self):
        super(FlowRefineNet, self).__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(12, 2, kernel_size=3, padding=1)
        )
    def forward(self, t_up_small, t_target_small, flow_small):
        x = torch.cat([t_up_small, t_target_small, flow_small], dim=1)
        return flow_small + self.refiner(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False), nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(x).view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(x + self.conv(x))

# =========================================================================
# 【200k MACs 限制內終極外掛】時域自適應非剛體形變分支 (Deformable Branch)
# =========================================================================
class DeformableDilationBranch(nn.Module):
    """ 
    利用官方 ops.DeformConv2d，在 12 通道極窄空間下，逼出全向自適應對齊能力。
    計算量等同標準卷積，完全安全卡在 200k MACs 天花板之內！
    """
    def __init__(self, in_channels, out_channels, dilation):
        super(DeformableDilationBranch, self).__init__()
        self.dilation = dilation
        # 用一個超輕量的小卷積去幻想 2D 空間偏移量 (Offsets)
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=dilation, dilation=dilation)
        self.dcn = ops.DeformConv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.res = ResidualBlock(out_channels)
        
    def forward(self, x):
        offsets = self.offset_conv(x)
        out = self.dcn(x, offsets)
        return self.res(out)

class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        # 完美守衛 48 通道主幹
        self.conv1 = nn.Conv2d(8, 48, kernel_size=3, padding=1)
        self.se1 = SEBlock(48)
        
        # Dilation 1 保持剛性邊緣對齊
        self.conv2_d1 = nn.Sequential(nn.Conv2d(48, 12, kernel_size=3, padding=1, dilation=1), ResidualBlock(12))
        # 🚨【解鎖核武】Dilation 5 & 9 裝配 DCNv2 可形變矩陣，主動捕捉手腳揮舞非剛體形變！
        self.conv2_d5 = DeformableDilationBranch(48, 12, dilation=5)
        self.conv2_d9 = DeformableDilationBranch(48, 12, dilation=9)
        
        self.se2 = SEBlock(36)
        self.conv3 = nn.Conv2d(36, 24, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(24, 4, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n, flow_consistency, laplacian_guide, epi_p, epi_n):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n)
        
        x = torch.cat([diff_p, diff_n, diff_pn, t_up, flow_consistency, laplacian_guide, epi_p, epi_n], dim=1)
        feat1 = self.se1(self.gelu(self.conv1(x)))
        
        f1 = self.conv2_d1(feat1)
        f5 = self.conv2_d5(feat1) # 可形變激盪
        f9 = self.conv2_d9(feat1) # 可形變激盪
        
        feat2 = self.se2(torch.cat([f1, f5, f9], dim=1))
        feat3 = self.gelu(self.conv3(feat2))
        
        out = self.conv4(feat3)
        raw_logits = out[:, :3, :, :]
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        
        weights = F.softmax(raw_logits * self.temp, dim=1)
        fused_base = (weights[:, 0:1] * t_up) + (weights[:, 1:2] * warped_p) + (weights[:, 2:3] * warped_n)
        return torch.clamp(fused_base + delta_res, 0.0, 1.0), weights

# =========================================================================
# 輔助函數與主訓練大迴圈 (100% 繼承 test26 的黃金 IO 與安全 4K Warp 流)
# =========================================================================
W8_H = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 1, 8) / 64.0
W8_V = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 8, 1) / 64.0
LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def gpu_8tap_upscale_y(img_tensor):
    b, c, h, w = img_tensor.shape
    inter = torch.zeros((1, 1, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(F.pad(img_tensor, (3, 4, 0, 0), mode='reflect'), W8_H)
    out = torch.zeros((1, 1, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(F.pad(inter, (0, 0, 3, 4), mode='reflect'), W8_V)
    return out

def warp_tensor(t_img, flow_4k):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_4k[:, 0, :, :]
    grid_y = mesh_y.float() + flow_4k[:, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(t_img, torch.stack((norm_x, norm_y), dim=-1), mode='bicubic', padding_mode='reflection', align_corners=True)

def get_random_crops(t_up, t_p, t_n, t_gt, t_consistent, t_epi_p, t_epi_n, num_crops=64, patch_size=256):
    _, _, H, W = t_up.shape
    up_crops, p_crops, n_crops, gt_crops, cons_crops, epi_p_crops, epi_n_crops = [], [], [], [], [], [], []
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up_crops.append(t_up[:, :, h:h+patch_size, w:w+patch_size])
        p_crops.append(t_p[:, :, h:h+patch_size, w:w+patch_size])
        n_crops.append(t_n[:, :, h:h+patch_size, w:w+patch_size])
        gt_crops.append(t_gt[:, :, h:h+patch_size, w:w+patch_size])
        cons_crops.append(t_consistent[:, :, h:h+patch_size, w:w+patch_size])
        epi_p_crops.append(t_epi_p[:, :, h:h+patch_size, w:w+patch_size])
        epi_n_crops.append(t_epi_n[:, :, h:h+patch_size, w:w+patch_size])
    return (torch.cat(up_crops), torch.cat(p_crops), torch.cat(n_crops), 
            torch.cat(gt_crops), torch.cat(cons_crops), torch.cat(epi_p_crops), torch.cat(epi_n_crops))

def get_flow_smoothness_loss(flow, img):
    flow_grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    flow_grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    img_grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    img_grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
    weight_x = torch.exp(-img_grad_x * 10.0)
    weight_y = torch.exp(-img_grad_y * 10.0)
    return torch.mean(flow_grad_x * weight_x) + torch.mean(flow_grad_y * weight_y)

def get_laplacian_map(tensor_img): return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

def main():
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}
    ]
    
    width_base, height_base = 1920, 1080
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)

    fusion_model = MicroFusionNet().to(device)
    refine_model = FlowRefineNet().to(device)
    
    eps = 1e-6
    optimizer = optim.AdamW(list(fusion_model.parameters()) + list(refine_model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    epochs = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    print(f"🎬 啟動【48配12 + SOTA RAFT底座 + DCNv2可形變分支】test26_DCN 決戰特訓...")
    
    for epoch in range(epochs):
        fusion_model.train()
        refine_model.train()
        random.shuffle(train_tasks)
        
        for task in train_tasks:
            if not os.path.exists(task['base']): continue
            frameCount = (os.path.getsize(task['base']) // byteN_base) - 1
            task_loss = 0.0
            task_flow_dir = os.path.join(PRECOMPUTED_DIR, f"{task['name']}_qp{task['qp']}")
            
            pbar = tqdm(range(frameCount), desc=f"Epoch {epoch+1}/{epochs} | {task['name']}", unit="frame")
            for t in pbar:
                prev_file_path = task['base'].replace('base', 'enhance').replace('odd_', 'even_').replace('.layer0.yuv', '.layer1.yuv')
                
                f_base = getOneFrame(task['base'], width_base, height_base, bytesPerPel, t)
                f_prev = getOneFrame(prev_file_path, 3840, 2160, bytesPerPel, t)
                f_next = getOneFrame(prev_file_path, 3840, 2160, bytesPerPel, t+1)
                f_gt   = getOneFrame(task['gt'], 3840, 2160, bytesPerPel, t)
                
                with torch.no_grad():
                    t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                    t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_gt_y = torch.tensor(f_gt['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_up_y = gpu_8tap_upscale_y(t_base_y)
                    
                    # 讀取由新版 RAFT 抽取出的高精度幾何快取
                    flow_data = torch.load(os.path.join(task_flow_dir, f"frame_{t}.pt"), map_location=device, weights_only=True)
                    flow_p_t = flow_data['fp'].float()
                    flow_n_t = flow_data['fn'].float()
                    flow_p_back_t = flow_data['fpb'].float()
                    flow_n_back_t = flow_data['fnb'].float()
                    epi_p_cv_t = flow_data['ep'].float()
                    epi_n_cv_t = flow_data['en'].float()

                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    t_up_small = F.interpolate(t_up_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                    t_p_small = F.interpolate(t_p_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                    t_n_small = F.interpolate(t_n_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                    
                    flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_p_t)
                    flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_n_t)
                    flow_p_back_refined = refine_model(t_p_small, t_up_small, flow_p_back_t)
                    flow_n_back_refined = refine_model(t_n_small, t_up_small, flow_n_back_t)
                    
                    err_p_gpu = torch.norm(flow_p_t_refined + flow_p_back_refined, dim=1, keepdim=True)
                    err_n_gpu = torch.norm(flow_n_t_refined + flow_n_back_refined, dim=1, keepdim=True)
                    flow_consistency_t = torch.tanh(torch.max(err_p_gpu, err_n_gpu) / 3.0)
                    
                    flow_consistency_4k = F.interpolate(flow_consistency_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    epi_p_4k = F.interpolate(epi_p_cv_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    epi_n_4k = F.interpolate(epi_n_cv_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    
                    flow_p_4k = F.interpolate(flow_p_t_refined, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    flow_n_4k = F.interpolate(flow_n_t_refined, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    warped_p = warp_tensor(t_p_y, flow_p_4k)
                    warped_n = warp_tensor(t_n_y, flow_n_4k)
                    
                    b_up, b_p, b_n, b_gt, b_cons, b_epi_p, b_epi_n = get_random_crops(
                        t_up_y, warped_p, warped_n, t_gt_y, flow_consistency_4k, epi_p_4k, epi_n_4k, num_crops=64, patch_size=256
                    )
                    b_lap = get_laplacian_map(b_up)

                    fused_pred, weights = fusion_model(b_up, b_p, b_n, b_cons, b_lap, b_epi_p, b_epi_n)
                    
                    loss_pixel = torch.mean(torch.sqrt((fused_pred - b_gt)**2 + eps))
                    loss_smooth = get_flow_smoothness_loss(flow_p_t_refined, t_up_small) + get_flow_smoothness_loss(flow_n_t_refined, t_up_small)
                    loss = loss_pixel + 0.02 * loss_smooth

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                task_loss += loss.item()
                pbar.set_postfix({'Pixel': f"{loss_pixel.item():.4f}"})
                
            print(f"   └─ 平均 Loss: {task_loss/frameCount:.5f}")
            torch.cuda.empty_cache()

        scheduler.step()
        torch.save(fusion_model.state_dict(), FUSION_MODEL_PATH)
        torch.save(refine_model.state_dict(), REFINE_MODEL_PATH)
        print(f"💾 Epoch {epoch+1} 48配12可形變模型權重已儲存。\n")

if __name__ == '__main__':
    main()