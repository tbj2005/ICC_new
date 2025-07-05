import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time


def main():
    # 在这里直接设置您想要的batch_size
    batch_size = 128  # 可以修改为您想要的任何值

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"设置 batch_size = {batch_size}")

    # 超参数
    learning_rate = 0.001

    # CIFAR-10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # VGG配置
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }

    class VGG(nn.Module):
        def __init__(self, vgg_name):
            super(VGG, self).__init__()
            self.features = self._make_layers(cfg[vgg_name])
            self.classifier = nn.Linear(512, 10)

        def forward(self, x):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

        def _make_layers(self, cfg):
            layers = []
            in_channels = 3
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)

    def train_model(model_name):
        print(f"\n评估 {model_name}...")
        model = VGG(model_name).to(device)
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

        # 正式计时
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
        print(f"{model_name} 平均单次迭代时间: {avg_time:.4f} 秒 (batch_size={batch_size})")

        # 打印内存使用情况
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        return avg_time

    # 训练所有模型
    print(f"\n开始评估 (batch_size={batch_size})")
    vgg11_time = train_model('VGG11')
    vgg16_time = train_model('VGG16')
    vgg19_time = train_model('VGG19')

    # 打印总结
    print("\n评估结果总结:")
    print(f"VGG11 平均迭代时间: {vgg11_time:.4f} 秒")
    print(f"VGG16 平均迭代时间: {vgg16_time:.4f} 秒")
    print(f"VGG19 平均迭代时间: {vgg19_time:.4f} 秒")
    print(f"测试配置: batch_size={batch_size}, 迭代次数=10")


if __name__ == '__main__':
    main()
