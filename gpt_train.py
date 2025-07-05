import torch
from transformers import GPT2LMHeadModel, GPT2Config
import time
import numpy as np

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 测试配置
batch_sizes = [2, 4, 8, 16, 32]
seq_length = 128  # 减小序列长度以适应内存
n_iterations = 10  # 每次测试的迭代次数

# GPT-1配置 (近似为GPT-1的117M参数)
gpt1_config = GPT2Config(
    n_layer=12, n_head=12, n_embd=768,
    vocab_size=50257,
)

# GPT-2 small配置 (124M参数)
gpt2_config = GPT2Config()


def benchmark_model(model, batch_sizes, seq_length, n_iterations):
    results = {}
    model.to(device)

    for bs in batch_sizes:
        print(f"\nBenchmarking batch size {bs}...")
        iteration_times = []

        # 准备固定输入数据（避免数据生成影响计时）
        input_ids = torch.randint(0, model.config.vocab_size, (bs, seq_length)).to(device)
        labels = torch.randint(0, model.config.vocab_size, (bs, seq_length)).to(device)

        # 预热（不计时）
        with torch.no_grad():
            _ = model(input_ids)

        # 训练模式
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for i in range(n_iterations):
            # 计时开始
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # 前向传播
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 计时结束
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            iteration_time = end_time - start_time
            iteration_times.append(iteration_time)
            print(f"Iter {i + 1}/{n_iterations}: {iteration_time:.4f}s", end="\r")

        # 计算平均时间（忽略第一次迭代可能存在的冷启动开销）
        avg_time = np.mean(iteration_times[1:])  # 排除第一次迭代
        results[bs] = avg_time

        # 清理内存
        del input_ids, labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# 测试GPT-1
print("\n" + "=" * 50)
print("Benchmarking GPT-1 (117M parameters)...")
gpt1_model = GPT2LMHeadModel(gpt1_config)
gpt1_results = benchmark_model(gpt1_model, batch_sizes, seq_length, n_iterations)

# 测试GPT-2
print("\n" + "=" * 50)
print("Benchmarking GPT-2 (124M parameters)...")
gpt2_model = GPT2LMHeadModel(gpt2_config)
gpt2_results = benchmark_model(gpt2_model, batch_sizes, seq_length, n_iterations)

# 打印结果表格
print("\n" + "=" * 50)
print("Average Training Time per Iteration (10 iterations)")
print(f"{'Model':<10} | {'Batch Size':>10} | {'Time (s)':>10}")
print("-" * 40)
for bs in batch_sizes:
    print(f"{'GPT-1':<10} | {bs:>10} | {gpt1_results[bs]:>10.4f}")
for bs in batch_sizes:
    print(f"{'GPT-2':<10} | {bs:>10} | {gpt2_results[bs]:>10.4f}")
