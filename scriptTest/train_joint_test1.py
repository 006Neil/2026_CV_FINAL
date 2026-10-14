import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from yuvProc import getOneFrame
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"訓練使用設備: {device}")

# ==========================================
# 1. 網路架構 (MicroFusionNet)
# ==========================================
class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        # 輸入只剩 2 個 Channel：|warped_p - t_up| 和 |warped_n - t_up|
        # 運算量再次下降！對硬體更友善。
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, t_up, warped_p, warped_n):
        # 核心優化：不要餵原圖，餵「絕對誤差圖」！
        # 這樣網路一眼就能看出哪裡的光流算錯了 (誤差很大的地方)
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        x = torch.cat([diff_p, diff_n], dim=1)
        
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        
        # 這裡可以加一個乘法放大器 (Temperature scaling)，幫助 Softmax 產生更極端的權重 (如 0.9, 0.05, 0.05)
        logits = self.conv3(feat) * 10.0 
        
        # 預測融合權重
        weights = F.softmax(logits, dim=1)
        w_up = weights[:, 0:1, :, :]
        w_p = weights[:, 1:2, :, :]
        w_n = weights[:, 2:3, :, :]
        
        fused = (w_up * t_up) + (w_p * warped_p) + (w_n * warped_n)
        return fused, weights

# ==========================================
# 2. 輔助 GPU 函數 (沿用之前的極速版)
# ==========================================
W8_H = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 1, 8) / 64.0
W8_V = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 8, 1) / 64.0

def gpu_8tap_upscale_y(img_tensor):
    b, c, h, w = img_tensor.shape
    inter = torch.zeros((1, 1, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(F.pad(img_tensor, (3, 4, 0, 0), mode='reflect'), W8_H)
    out = torch.zeros((1, 1, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(F.pad(inter, (0, 0, 3, 4), mode='reflect'), W8_V)
    return out

def get_small_flow_img(tensor_4k):
    t_small = F.interpolate(tensor_4k, scale_factor=0.25, mode='bilinear', align_corners=False)
    return torch.clamp(t_small.squeeze() * 255.0, 0, 255).byte().cpu().numpy()

def warp_tensor(t_img, flow_4k):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_4k[0, 0, :, :]
    grid_y = mesh_y.float() + flow_4k[0, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    return F.grid_sample(t_img, grid, mode='bicubic', padding_mode='reflection', align_corners=True)

# ==========================================
# 3. 隨機裁切 (Random Crop) 產生 Batch
# ==========================================
def get_random_crops(t_up, t_p, t_n, t_gt, num_crops=32, patch_size=128):
    """ 從 4K 畫面中隨機切出 N 個小區塊組成 Batch """
    _, _, H, W = t_up.shape
    up_crops, p_crops, n_crops, gt_crops = [], [], [], []
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up_crops.append(t_up[:, :, h:h+patch_size, w:w+patch_size])
        p_crops.append(t_p[:, :, h:h+patch_size, w:w+patch_size])
        n_crops.append(t_n[:, :, h:h+patch_size, w:w+patch_size])
        gt_crops.append(t_gt[:, :, h:h+patch_size, w:w+patch_size])
    return torch.cat(up_crops), torch.cat(p_crops), torch.cat(n_crops), torch.cat(gt_crops)

# ==========================================
# 4. 主訓練迴圈
# ==========================================
def main():
    # 檔案路徑設定 (針對 Zombie QP=27 進行訓練)
    qp = 27
    base_file = f'../bitstream/base/odd_ZombieClimbing2_{qp}_0_4.layer0.yuv'
    even_file = f'../bitstream/enhance/even_ZombieClimbing2_{qp}_0_4.layer1.yuv'
    gt_file = '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'
    
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)
    frameCount = (os.path.getsize(base_file) // byteN_base) - 1

    # 初始化模型、Loss 與 Optimizer
    model = MicroFusionNet().to(device)
    criterion = nn.L1Loss() # 使用 L1 (MAE) 能比 MSE 保留更多高頻銳利細節
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30 # 訓練 10 輪
    
    print(f"🎬 開始訓練 Zombie (總幀數: {frameCount}, Epochs: {epochs})")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 顯示進度條
        pbar = tqdm(range(frameCount), desc=f"Epoch {epoch+1}/{epochs}", unit="frame")
        for t in pbar:
            # 讀取資料 (包含 Ground Truth)
            f_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
            f_prev = getOneFrame(even_file, width_el, height_el, bytesPerPel, t)
            f_next = getOneFrame(even_file, width_el, height_el, bytesPerPel, t+1)
            f_gt = getOneFrame(gt_file, width_el, height_el, bytesPerPel, t)
            
            # 轉為 Tensor (只取 Y 通道進行訓練)
            with torch.no_grad(): # 資料準備階段不需要算梯度
                t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                t_gt_y = torch.tensor(f_gt['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                
                # 升頻與光流 (與推論邏輯完全一致)
                t_up_y = gpu_8tap_upscale_y(t_base_y)
                
                up_8b, p_8b, n_8b = get_small_flow_img(t_up_y), get_small_flow_img(t_p_y), get_small_flow_img(t_n_y)
                flow_p_small = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_n_small = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                flow_p_t = torch.tensor(flow_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                flow_n_t = torch.tensor(flow_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                
                flow_p_4k = F.interpolate(flow_p_t, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                flow_n_4k = F.interpolate(flow_n_t, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                
                # 產生扭曲後的影像
                warped_p = warp_tensor(t_p_y, flow_p_4k)
                warped_n = warp_tensor(t_n_y, flow_n_4k)
                
                # 切割出 32 個 128x128 的 Batch (這一步極大節省記憶體並增加泛化能力)
                b_up, b_p, b_n, b_gt = get_random_crops(t_up_y, warped_p, warped_n, t_gt_y, num_crops=32, patch_size=128)

            # ======== 開始訓練 ========
            optimizer.zero_grad()
            fused_pred, weights = model(b_up, b_p, b_n)
            loss = criterion(fused_pred, b_gt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'L1 Loss': f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} 結束, 平均 Loss: {epoch_loss/frameCount:.5f}")
        
        # 每 Epoch 結束存檔一次
        torch.save(model.state_dict(), "zombie_fusion_net.pth")

if __name__ == '__main__':
    main()