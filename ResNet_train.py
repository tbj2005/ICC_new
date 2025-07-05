import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torchvision.models import resnet18, resnet34, resnet50, resnet101


def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 测试的batch_size列表
    batch_sizes = [32, 64, 128, 256, 512]

    # 要测试的模型列表
    models = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101
    }

    # 存储结果的字典
    results = {model_name: {} for model_name in models.keys()}

    # 超参数
    learning_rate = 0.001

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 对每个batch_size进行测试
    for batch_size in batch_sizes:
        print(f"\n{'=' * 40}")
        print(f"开始测试 batch_size = {batch_size}")
        print(f"{'=' * 40}")

        # 准备数据集
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 测试每个模型
        for model_name, model_fn in models.items():
            print(f"\n测试模型: {model_name} (batch_size={batch_size})")

            # 初始化模型 (适配CIFAR-10的10分类)
            model = model_fn(num_classes=10).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 预热一次，避免初始化时间影响
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                if i > 0:
                    break
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 正式计时测试
            total_time = 0
            iterations = 10
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                if i >= iterations:
                    break

                start_time = time.time()

                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_time = time.time() - start_time
                total_time += iter_time
                print(f"迭代 {i + 1}/{iterations}: {iter_time:.4f} 秒")

            avg_time = total_time / iterations
            results[model_name][batch_size] = avg_time
            print(f"{model_name} 平均单次迭代时间: {avg_time:.4f} 秒 (batch_size={batch_size})")

            # 打印内存使用情况
            if torch.cuda.is_available():
                print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    # 打印最终结果表格
    print("\n\n最终测试结果:")
    print(f"{'模型':<10}", end="")
    for bs in batch_sizes:
        print(f"{f'bs={bs}':<15}", end="")
    print()

    for model_name, timings in results.items():
        print(f"{model_name:<10}", end="")
        for bs in batch_sizes:
            print(f"{timings[bs]:<15.4f}", end="")
        print()


if __name__ == '__main__':
    main()
