#!/usr/bin/env python3
"""
可视化比较 Per-Layer Scaling vs 统一 Scaling 的效果
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_log_file(log_path):
    """解析日志文件提取关键指标"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取 Top1 Accuracy curve
    top1_match = re.search(r'CNN top1 curve: \[(.*?)\]', content)
    if top1_match:
        top1_str = top1_match.group(1)
        # 处理 np.float64() 格式
        top1_values = [float(x.split('(')[1].split(')')[0]) if 'float64' in x else float(x) 
                      for x in top1_str.split(',')]
    else:
        top1_values = []
    
    # 提取 Top5 Accuracy curve
    top5_match = re.search(r'CNN top5 curve: \[(.*?)\]', content)
    if top5_match:
        top5_str = top5_match.group(1)
        top5_values = [float(x.split('(')[1].split(')')[0]) if 'float64' in x else float(x) 
                      for x in top5_str.split(',')]
    else:
        top5_values = []
    
    # 提取最终平均准确率
    avg_acc_match = re.search(r'Average Accuracy \(CNN\): ([\d.]+)', content)
    avg_acc = float(avg_acc_match.group(1)) if avg_acc_match else None
    
    # 提取 Forgetting
    forgetting_match = re.search(r'Forgetting \(CNN\): ([\d.]+)', content)
    forgetting = float(forgetting_match.group(1)) if forgetting_match else None
    
    # 提取 Accuracy Matrix
    matrix_match = re.search(r'Accuracy Matrix \(CNN\):\n(\[[\s\S]*?\])\n', content)
    accuracy_matrix = None
    if matrix_match:
        matrix_str = matrix_match.group(1)
        # 解析矩阵
        lines = matrix_str.strip().split('\n')
        matrix_data = []
        for line in lines:
            # 提取数字
            nums = re.findall(r'[\d.]+', line)
            matrix_data.append([float(x) for x in nums])
        accuracy_matrix = np.array(matrix_data)
    
    # 提取每个任务的最终准确率
    task_accuracies = []
    task_pattern = r"CNN: \{.*?'total': np\.float64\(([\d.]+)\)"
    for match in re.finditer(task_pattern, content):
        task_accuracies.append(float(match.group(1)))
    
    return {
        'top1_curve': top1_values,
        'top5_curve': top5_values,
        'avg_accuracy': avg_acc,
        'forgetting': forgetting,
        'accuracy_matrix': accuracy_matrix,
        'task_accuracies': task_accuracies
    }

def plot_comparison(per_layer_data, unified_data, output_dir='./visualizations'):
    """创建对比图表"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # 1. Top1 Accuracy 曲线对比
    fig, ax = plt.subplots(figsize=(10, 6))
    tasks = np.arange(1, len(per_layer_data['top1_curve']) + 1)
    
    ax.plot(tasks, per_layer_data['top1_curve'], 'o-', linewidth=2, 
            markersize=8, label='Per-Layer Scaling', color='#2E86AB')
    ax.plot(tasks, unified_data['top1_curve'], 's-', linewidth=2, 
            markersize=8, label='Unified Scaling', color='#A23B72')
    
    ax.set_xlabel('Number of Tasks Learned', fontsize=12)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Top-1 Accuracy Comparison: Per-Layer vs Unified Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tasks)
    
    # 添加最终准确率标注
    final_per = per_layer_data['top1_curve'][-1]
    final_uni = unified_data['top1_curve'][-1]
    ax.annotate(f'{final_per:.2f}%', 
                xy=(tasks[-1], final_per), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, color='#2E86AB')
    ax.annotate(f'{final_uni:.2f}%', 
                xy=(tasks[-1], final_uni), 
                xytext=(5, -15), textcoords='offset points',
                fontsize=10, color='#A23B72')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top1 vs Top5 对比（子图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top1
    ax1.plot(tasks, per_layer_data['top1_curve'], 'o-', linewidth=2, 
            markersize=7, label='Per-Layer', color='#2E86AB')
    ax1.plot(tasks, unified_data['top1_curve'], 's-', linewidth=2, 
            markersize=7, label='Unified', color='#A23B72')
    ax1.set_xlabel('Number of Tasks', fontsize=11)
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax1.set_title('Top-1 Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(tasks)
    
    # Top5
    ax2.plot(tasks, per_layer_data['top5_curve'], 'o-', linewidth=2, 
            markersize=7, label='Per-Layer', color='#2E86AB')
    ax2.plot(tasks, unified_data['top5_curve'], 's-', linewidth=2, 
            markersize=7, label='Unified', color='#A23B72')
    ax2.set_xlabel('Number of Tasks', fontsize=11)
    ax2.set_ylabel('Top-5 Accuracy (%)', fontsize=11)
    ax2.set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(tasks)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top1_vs_top5_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 关键指标对比（柱状图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 平均准确率对比
    methods = ['Per-Layer\nScaling', 'Unified\nScaling']
    avg_accs = [per_layer_data['avg_accuracy'], unified_data['avg_accuracy']]
    colors = ['#2E86AB', '#A23B72']
    
    bars1 = ax1.bar(methods, avg_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax1.set_title('Final Average Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim([min(avg_accs) - 1, max(avg_accs) + 1])
    
    # 在柱子上添加数值
    for bar, acc in zip(bars1, avg_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Forgetting 对比
    forgettings = [per_layer_data['forgetting'], unified_data['forgetting']]
    bars2 = ax2.bar(methods, forgettings, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Forgetting (%)', fontsize=12)
    ax2.set_title('Catastrophic Forgetting', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, max(forgettings) + 1])
    
    # 在柱子上添加数值
    for bar, fgt in zip(bars2, forgettings):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{fgt:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/key_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Accuracy Matrix 热图对比
    if per_layer_data['accuracy_matrix'] is not None and unified_data['accuracy_matrix'] is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Per-Layer Scaling
        im1 = ax1.imshow(per_layer_data['accuracy_matrix'], cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        ax1.set_title('Per-Layer Scaling\nAccuracy Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Task Index', fontsize=11)
        ax1.set_ylabel('Task Index', fontsize=11)
        ax1.set_xticks(range(10))
        ax1.set_yticks(range(10))
        ax1.set_xticklabels(range(1, 11))
        ax1.set_yticklabels(range(1, 11))
        
        # 添加数值标注
        for i in range(10):
            for j in range(i+1):
                if per_layer_data['accuracy_matrix'][i, j] > 0:
                    text = ax1.text(j, i, f'{per_layer_data["accuracy_matrix"][i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # Unified Scaling
        im2 = ax2.imshow(unified_data['accuracy_matrix'], cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        ax2.set_title('Unified Scaling\nAccuracy Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Task Index', fontsize=11)
        ax2.set_ylabel('Task Index', fontsize=11)
        ax2.set_xticks(range(10))
        ax2.set_yticks(range(10))
        ax2.set_xticklabels(range(1, 11))
        ax2.set_yticklabels(range(1, 11))
        
        # 添加数值标注
        for i in range(10):
            for j in range(i+1):
                if unified_data['accuracy_matrix'][i, j] > 0:
                    text = ax2.text(j, i, f'{unified_data["accuracy_matrix"][i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # 添加共享的colorbar
        fig.colorbar(im2, ax=[ax1, ax2], label='Accuracy (%)', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_matrix_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 性能差异分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diff = np.array(per_layer_data['top1_curve']) - np.array(unified_data['top1_curve'])
    colors_diff = ['#2E86AB' if d > 0 else '#A23B72' for d in diff]
    
    bars = ax.bar(tasks, diff, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Number of Tasks', fontsize=12)
    ax.set_ylabel('Accuracy Difference (%)\n(Per-Layer - Unified)', fontsize=12)
    ax.set_title('Performance Difference: Per-Layer vs Unified Scaling', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(tasks)
    
    # 添加数值标注
    for i, (bar, d) in enumerate(zip(bars, diff)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{d:+.2f}',
                ha='center', va='bottom' if d > 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_difference.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 所有可视化图表已保存到: {output_dir}/")

def print_summary_statistics(per_layer_data, unified_data):
    """打印统计摘要"""
    print("\n" + "="*70)
    print("比较统计摘要".center(70))
    print("="*70)
    
    print("\n【最终性能指标】")
    print(f"  Per-Layer Scaling:")
    print(f"    - 平均准确率: {per_layer_data['avg_accuracy']:.2f}%")
    print(f"    - 遗忘程度:   {per_layer_data['forgetting']:.2f}%")
    print(f"    - 最终任务准确率: {per_layer_data['top1_curve'][-1]:.2f}%")
    
    print(f"\n  Unified Scaling:")
    print(f"    - 平均准确率: {unified_data['avg_accuracy']:.2f}%")
    print(f"    - 遗忘程度:   {unified_data['forgetting']:.2f}%")
    print(f"    - 最终任务准确率: {unified_data['top1_curve'][-1]:.2f}%")
    
    print(f"\n【性能差异】")
    acc_diff = per_layer_data['avg_accuracy'] - unified_data['avg_accuracy']
    fgt_diff = per_layer_data['forgetting'] - unified_data['forgetting']
    
    print(f"  平均准确率差异: {acc_diff:+.2f}%")
    print(f"  遗忘程度差异:   {fgt_diff:+.2f}%")
    
    if acc_diff > 0:
        print(f"  → Per-Layer Scaling 表现更好 (准确率高 {acc_diff:.2f}%)")
    elif acc_diff < 0:
        print(f"  → Unified Scaling 表现更好 (准确率高 {-acc_diff:.2f}%)")
    else:
        print(f"  → 两种方法性能相当")
    
    print("\n【各任务性能对比】")
    print("  任务  | Per-Layer | Unified | 差异")
    print("  " + "-"*45)
    for i, (pl, uni) in enumerate(zip(per_layer_data['top1_curve'], 
                                       unified_data['top1_curve']), 1):
        diff = pl - uni
        print(f"  {i:2d}    | {pl:6.2f}%  | {uni:6.2f}% | {diff:+6.2f}%")
    
    print("="*70 + "\n")

def main():
    # 日志文件路径
    per_layer_log = './C100_per_layer.log'
    unified_log = './C100.log'
    
    print("正在解析日志文件...")
    per_layer_data = parse_log_file(per_layer_log)
    unified_data = parse_log_file(unified_log)
    
    print("✓ 日志解析完成")
    
    # 打印统计摘要
    print_summary_statistics(per_layer_data, unified_data)
    
    # 生成可视化图表
    print("正在生成可视化图表...")
    plot_comparison(per_layer_data, unified_data)
    
    print("\n完成! 请查看 ./visualizations/ 目录下的图表。")

if __name__ == '__main__':
    main()
