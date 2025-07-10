import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 配置参数
file_mapping = {
    '8': 'output_8_racks.xlsx',
    '12': 'output_12_racks.xlsx',
    '16': 'output_16_racks.xlsx',
    '20': 'output_20_racks.xlsx',
    '24': 'output_24_racks.xlsx'
}
time_algorithms = ['MAIN', 'CASSINI', 'SJF', 'LAS', 'HRRN', 'FCFS']
util_algorithms = ['MAIN', 'CASSINI', 'SJF', 'LAS', 'HRRN', 'FCFS']


# 2. 数据处理函数
def process_data_rack(filepath):
    """处理文件并返回时间和利用率的统计数据"""
    try:
        df = pd.read_excel(filepath)
        # 处理时间数据 (前6列)
        time_data = df.iloc[:, :6]
        time_data.columns = time_algorithms
        time_data = time_data.apply(pd.to_numeric, errors='coerce')

        # 处理利用率数据 (后6列)
        util_data = df.iloc[:, 6:12]
        util_data.columns = util_algorithms
        util_data = util_data.apply(pd.to_numeric, errors='coerce')

        return {
            'time_mean': time_data.mean(),
            'util_mean': util_data.mean()
        }
    except Exception as e:
        print(f"处理文件 {filepath} 出错: {str(e)}")
        return None


# 3. 处理所有文件
results = {}
for label, filename in file_mapping.items():
    if os.path.exists(filename):
        result = process_data_rack(filename)
        if result is not None:
            results[label] = result


# 4. 绘图函数
def plot_metrics(metric_type, title, ylabel):
    """通用绘图函数"""
    plt.figure(figsize=(10, 5))
    x_labels = sorted(results.keys(), key=int)
    x = np.arange(len(x_labels))

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    algorithms = time_algorithms if metric_type == 'time' else util_algorithms
    for i, algo in enumerate(algorithms):
        means = [results[label][f'{metric_type}_mean'][algo] for label in x_labels]

        plt.plot(x, means, 'o-', color=colors[i], label=algo.replace('_util', ''), linewidth=2)

    plt.title(title, fontsize=12)
    plt.xlabel('Number of Racks', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(x, x_labels)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', framealpha=0.7, edgecolor='gray')
    plt.tight_layout()
    plt.savefig(f'{metric_type}_performance_rack.png', dpi=300)
    plt.close()


# 5. 生成并保存两个图表
plot_metrics('time', '', 'Average Iteration Time')
plot_metrics('util', '', 'Network Utilization')

print("图表已保存为：")
print("- time_performance_rack.png (执行时间)")
print("- util_performance_rack.png (网络利用率)")