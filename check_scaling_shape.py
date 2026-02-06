"""检查保存的 scaling_factor 文件的形状"""
import torch
import os

cf100_per_layer_path = "out/CF100_per_layer/"

if os.path.exists(os.path.join(cf100_per_layer_path, 'scaling_factor9.pt')):
    sf = torch.load(os.path.join(cf100_per_layer_path, 'scaling_factor9.pt'), map_location='cpu')
    print(f"scaling_factor9.pt 存在")
    print(f"  类型: {type(sf)}")
    print(f"  形状: {sf.shape if hasattr(sf, 'shape') else 'N/A'}")
    print(f"  dtype: {sf.dtype if hasattr(sf, 'dtype') else 'N/A'}")
    print(f"  内容:\n{sf}")
else:
    print("scaling_factor9.pt 不存在")

# 也检查一下 CF100 目录的
cf100_path = "out/CF100/"
if os.path.exists(os.path.join(cf100_path, 'scaling_factor9.pt')):
    sf = torch.load(os.path.join(cf100_path, 'scaling_factor9.pt'), map_location='cpu')
    print(f"\nCF100/scaling_factor9.pt 存在")
    print(f"  类型: {type(sf)}")
    print(f"  形状: {sf.shape if hasattr(sf, 'shape') else 'N/A'}")
    print(f"  dtype: {sf.dtype if hasattr(sf, 'dtype') else 'N/A'}")
    print(f"  内容:\n{sf}")
