"""
比较 CF100 (per_layer_scaling=False) 和 CF100_per_layer (per_layer_scaling=True) 的 scaling factors
"""
import torch
import os
import numpy as np

# Paths
cf100_path = "out/CF100/"
cf100_per_layer_path = "out/CF100_per_layer/"

print("=" * 80)
print("比较 CF100 (全局) 和 CF100_per_layer (每层单独) 的 scaling_factor")
print("=" * 80)

# Check what files exist
cf100_files = sorted([f for f in os.listdir(cf100_path) if f.startswith('scaling_factor')])
cf100_per_layer_files = sorted([f for f in os.listdir(cf100_per_layer_path) if f.startswith('scaling_factor')])

print(f"\nCF100 (global) 文件: {cf100_files}")
print(f"CF100_per_layer 文件: {cf100_per_layer_files}")

# ============================================================================
# Part 1: 分析 CF100 (per_layer_scaling=False) - 全局 Scaling Factors
# ============================================================================
print("\n" + "=" * 80)
print("【CF100】per_layer_scaling=False: 所有层共享一个 Scaling Factor")
print("=" * 80)

global_scaling_data = {}

for sf_file in cf100_files:
    task_id = int(sf_file.replace('scaling_factor', '').replace('.pt', ''))
    sf = torch.load(os.path.join(cf100_path, sf_file), map_location='cpu')
    print(f"\n--- Task {task_id} 训练完成后保存的 scaling factors ---")
    print(f"    张量形状: {sf.shape}")
    
    # For global scaling, it's a (20, 20) matrix
    # Row i contains scaling factors for all LoRAs when processing task i
    for i in range(task_id + 1):
        row = sf[i, :i+1].numpy()
        global_scaling_data[(task_id, i)] = row
        print(f"    训练 Task {i} 时各 LoRA 的 scaling: {row}")

# ============================================================================
# Part 2: 分析 CF100_per_layer (per_layer_scaling=True) - 每层单独的 Scaling Factors
# ============================================================================
print("\n" + "=" * 80)
print("【CF100_per_layer】per_layer_scaling=True: 每层有自己的 Scaling Factor")
print("=" * 80)

per_layer_scaling_data = {}

for sf_file in cf100_per_layer_files:
    task_id = int(sf_file.replace('scaling_factor', '').replace('.pt', ''))
    sf = torch.load(os.path.join(cf100_per_layer_path, sf_file), map_location='cpu')
    print(f"\n--- Task {task_id} 训练完成后保存的 scaling factors ---")
    print(f"    张量形状: {sf.shape}")
    
    if len(sf.shape) == 3:
        num_layers = sf.shape[0]
        print(f"    ViT 层数: {num_layers}")
        
        for i in range(task_id + 1):
            print(f"\n    训练 Task {i} 时各层的 scaling factors:")
            layer_values = []
            for layer_idx in range(num_layers):
                val = sf[layer_idx, i, :i+1].numpy()
                layer_values.append(val)
                # 只打印当前 task 的 scaling (对角线元素)
                current_scale = val[-1] if len(val) > 0 else 0
                print(f"      Layer {layer_idx:2d}: 当前 task scaling = {current_scale:.6f}")
            per_layer_scaling_data[(task_id, i)] = layer_values
    else:
        print(f"    数据: {sf}")

# ============================================================================
# Part 3: 对比分析
# ============================================================================
print("\n" + "=" * 80)
print("【对比分析】全局 vs 每层单独训练的 Scaling Factor")
print("=" * 80)

# 找到两个目录中都有的最大 task id
max_global_task = max([int(f.replace('scaling_factor', '').replace('.pt', '')) for f in cf100_files])
max_per_layer_task = max([int(f.replace('scaling_factor', '').replace('.pt', '')) for f in cf100_per_layer_files])
common_max_task = min(max_global_task, max_per_layer_task)

print(f"\n全局方法最大 Task: {max_global_task}")
print(f"每层单独方法最大 Task: {max_per_layer_task}")
print(f"可比较的 Task 范围: 0 ~ {common_max_task}")

# 加载最新的 scaling factors 进行比较
if common_max_task >= 0:
    global_sf = torch.load(os.path.join(cf100_path, f'scaling_factor{common_max_task}.pt'), map_location='cpu')
    per_layer_sf = torch.load(os.path.join(cf100_per_layer_path, f'scaling_factor{common_max_task}.pt'), map_location='cpu')
    
    print(f"\n比较 Task {common_max_task} 的 scaling factors:")
    print("-" * 60)
    
    for task_i in range(common_max_task + 1):
        print(f"\n【Task {task_i} 的 LoRA scaling factor】")
        
        # 全局 scaling
        global_scale = global_sf[task_i, task_i].item()
        print(f"  全局 (所有层共享): {global_scale:.6f}")
        
        # 每层 scaling
        if len(per_layer_sf.shape) == 3:
            num_layers = per_layer_sf.shape[0]
            per_layer_scales = [per_layer_sf[layer_idx, task_i, task_i].item() for layer_idx in range(num_layers)]
            
            print(f"  每层单独训练:")
            for layer_idx, scale in enumerate(per_layer_scales):
                diff = scale - global_scale
                diff_str = f"+{diff:.6f}" if diff >= 0 else f"{diff:.6f}"
                print(f"    Layer {layer_idx:2d}: {scale:.6f}  (与全局差异: {diff_str})")
            
            # 统计信息
            mean_scale = np.mean(per_layer_scales)
            std_scale = np.std(per_layer_scales)
            min_scale = np.min(per_layer_scales)
            max_scale = np.max(per_layer_scales)
            
            print(f"\n    统计: mean={mean_scale:.6f}, std={std_scale:.6f}, min={min_scale:.6f}, max={max_scale:.6f}")
            print(f"    全局 vs 每层均值差异: {abs(global_scale - mean_scale):.6f}")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
