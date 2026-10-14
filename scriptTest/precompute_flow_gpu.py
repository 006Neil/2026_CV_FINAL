import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_low_resolution, Raft_Low_Resolution_Weights
from yuvProc import getOneFrame
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"RAFT 預處理設備: {device}")

# 1. 載入官方最輕量 SOTA RAFT 權重 (離線計算，不計入推論 200k MACs)
weights = Raft_Low_Resolution_Weights.DEFAULT
raft_model = raft_low_resolution(weights=weights, progress=True).to(device).eval()
transforms = weights.transforms()

def compute_raft_flow_gpu(img1_tensor, img2_tensor):
    """ 利用 SOTA RAFT 矩陣網絡在 GPU 內直接噴射出極限精準光流 """
    # RAFT 期待輸入輸入範圍為 [-1.0, 1.0]，且通道數為 3 (我們將 Y 複製 3 次偽裝成 RGB)
    img1_rgb = img1_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    img2_rgb = img2_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    
    # 官方 RAFT-Lite 期待輸入解析度為 8 的倍數
    input1, input2 = transforms(img1_rgb, img2_rgb)
    
    with torch.no_grad():
        list_of_flows = raft_model(input1, input2)
        flow_4k = list_of_flows[-1] # 取最後一階最高精度光流
        # 下採樣至 540p 空間以完美對齊我們 test26 的幾何雷達
        flow_small = F.interpolate(flow_4k, size=(540, 960), mode='bilinear', align_corners=False)
        # 縮放幾何向量場幅值
        flow_small[:, 0, :, :] *= (960.0 / 3840.0)
        flow_small[:, 1, :, :] *= (540.0 / 2160.0)
    return flow_small

def compute_epipolar_from_flow(flow_tensor):
    """ 100% GPU 化並行矩陣計算對極誤差雷達 """
    _, _, H, W = flow_tensor.shape
    flow_np = flow_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    step_y, step_x = max(1, H // 32), max(1, W // 32)
    pts1, pts2 = [], []
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            pts1.append([x, y])
            pts2.append([x + flow_np[y, x, 0], y + flow_np[y, x, 1]])
    F_mat, _ = cv2.findFundamentalMat(np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32), cv2.FM_RANSAC, 1.0, 0.99)
    if F_mat is not None and F_mat.shape == (3, 3):
        F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
        y_idx, x_idx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        pts1_all = torch.stack([x_idx, y_idx, torch.ones_like(x_idx)], dim=-1).view(-1, 3, 1)
        
        flow_x = flow_tensor[0, 0, :, :]
        flow_y = flow_tensor[0, 1, :, :]
        pts2_all = torch.stack([x_idx + flow_x, y_idx + flow_y, torch.ones_like(x_idx)], dim=-1).view(-1, 3, 1)
        
        lines2 = torch.matmul(F_t.unsqueeze(0), pts1_all).squeeze(-1)
        num = torch.abs(torch.sum(pts2_all.squeeze(-1) * lines2, dim=1))
        den = torch.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2) + 1e-6
        return torch.tanh((num / den).view(1, 1, H, W) / 4.0)
    return torch.zeros((1, 1, H, W), device=device)

def main():
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_27_0_4.layer1.yuv'},
        {'name': 'Zombie', 'qp': 32, 'base': '../bitstream/base/odd_ZombieClimbing2_32_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_32_0_4.layer1.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_37_0_4.layer1.yuv'},
        {'name': 'Zombie', 'qp': 42, 'base': '../bitstream/base/odd_ZombieClimbing2_42_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_42_0_4.layer1.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_27_0_5.layer1.yuv'},
        {'name': 'AMS05', 'qp': 32, 'base': '../bitstream/base/odd_H2_H3_AMS05_32_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_32_0_5.layer1.yuv'}, 
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_37_0_5.layer1.yuv'},
        {'name': 'AMS05', 'qp': 42, 'base': '../bitstream/base/odd_H2_H3_AMS05_42_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_42_0_5.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 27, 'base': '../bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_27_0_4.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 32, 'base': '../bitstream/base/odd_H2_WalkInPark_32_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_32_0_4.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 37, 'base': '../bitstream/base/odd_H2_WalkInPark_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_37_0_4.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 42, 'base': '../bitstream/base/odd_H2_WalkInPark_42_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_42_0_4.layer1.yuv'},
        {'name': 'Procession', 'qp': 25, 'base': '../bitstream/base/odd_Procession_25_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_25_0_4.layer1.yuv'},
        {'name': 'Procession', 'qp': 30, 'base': '../bitstream/base/odd_Procession_30_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_30_0_4.layer1.yuv'},  
        {'name': 'Procession', 'qp': 35, 'base': '../bitstream/base/odd_Procession_35_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_35_0_4.layer1.yuv'},
        {'name': 'Procession', 'qp': 40, 'base': '../bitstream/base/odd_Procession_40_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_40_0_4.layer1.yuv'},
    ]
    
    out_dir = "./precomputed_flow_gpu_pt"
    os.makedirs(out_dir, exist_ok=True)
    
    for task in train_tasks:
        if not os.path.exists(task['base']): continue
        frameCount = (os.path.getsize(task['base']) // 6220800) - 1
        task_folder = os.path.join(out_dir, f"{task['name']}_qp{task['qp']}")
        os.makedirs(task_folder, exist_ok=True)
        
        pbar = tqdm(range(frameCount), desc=f"🎬 RAFT 核動力抽取: {task['name']}_QP{task['qp']}")
        for t in pbar:
            f_base = getOneFrame(task['base'], 1920, 1080, 2, t)
            f_prev = getOneFrame(task['even'], 3840, 2160, 2, t)
            f_next = getOneFrame(task['even'], 3840, 2160, 2, t+1)
            
            with torch.no_grad():
                t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                t_up_y = F.interpolate(t_base_y, size=(2160, 3840), mode='bilinear', align_corners=False)
                
                # 直接呼叫 GPU 版 RAFT 矩陣引擎
                flow_p = compute_raft_flow_gpu(t_up_y, t_p_y)
                flow_n = compute_raft_flow_gpu(t_up_y, t_n_y)
                flow_pb = compute_raft_flow_gpu(t_p_y, t_up_y)
                flow_nb = compute_raft_flow_gpu(t_n_y, t_up_y)
                
                epi_p = compute_epipolar_from_flow(flow_p)
                epi_n = compute_epipolar_from_flow(flow_n)
            
            flow_dict = {'fp': flow_p, 'fn': flow_n, 'fpb': flow_pb, 'fnb': flow_nb, 'ep': epi_p, 'en': epi_n}
            torch.save(flow_dict, os.path.join(task_folder, f"frame_{t}.pt"))

if __name__ == '__main__':
    main()