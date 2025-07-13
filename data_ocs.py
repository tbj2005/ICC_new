import pandas as pd
import matplotlib.pyplot as plt

# 读取数据文件
data_files = {
    "Dataset 1": "output_1_ocs.xlsx",
    "Dataset 2": "output_2_ocs.xlsx",
    "Dataset 3": "output_3_ocs.xlsx",
    "Dataset 4": "output_4_ocs.xlsx",
    "Dataset 5": "output_5_ocs.xlsx"
}

# 算法名称（前7列和后7列）
avg_algorithms = ["main", "main-notpe", "cassini", "sjf", "las", "hrrn", "fcfs"]
max_algorithms = ["main", "main-notpe", "cassini", "sjf", "las", "hrrn", "fcfs"]

# 提取数据
avg_data = {algo: [] for algo in avg_algorithms}
max_data = {algo: [] for algo in max_algorithms}

for dataset_name, file_path in data_files.items():
    df = pd.read_excel(file_path, header=None)

    # 前7列是平均时间
    for i, algo in enumerate(avg_algorithms):
        avg_data[algo].append(df.iloc[:, i].mean())  # 取每列的平均值

    # 后7列是最大时间
    for i, algo in enumerate(max_algorithms):
        max_data[algo].append(df.iloc[:, i + 7].mean())  # 从第8列开始

# 创建图表
plt.style.use('seaborn')
datasets = list(data_files.keys())

# 要显示的算法（排除fcfs）
plot_algorithms = [algo for algo in avg_algorithms if algo != "sjf"]

# 图1: 平均时间折线图
plt.figure(figsize=(10, 6))
for algo in plot_algorithms:
    plt.plot(datasets, avg_data[algo], marker='o', label=algo)

plt.title("Average Time Comparison Across Datasets")
plt.xlabel("Dataset")
plt.ylabel("Time")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 标签放在左上角外侧
plt.grid(True)
plt.tight_layout()
plt.savefig("average_time_comparison_ocs.png", bbox_inches='tight')
plt.close()

# 图2: 最大时间折线图
plt.figure(figsize=(10, 6))
for algo in plot_algorithms:
    plt.plot(datasets, max_data[algo], marker='o', label=algo)

plt.title("Maximum Time Comparison Across Datasets")
plt.xlabel("Dataset")
plt.ylabel("Time")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 标签放在左上角外侧
plt.grid(True)
plt.tight_layout()
plt.savefig("max_time_comparison_ocs.png", bbox_inches='tight')
plt.close()

print("两张图表已保存为 average_time_comparison_ocs.png 和 max_time_comparison_ocs.png")
