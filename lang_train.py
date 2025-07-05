import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import BertModel, BertTokenizer
    from transformers import RobertaModel, RobertaTokenizer
    from transformers import CamembertModel, CamembertTokenizer
    from transformers import XLMModel, XLMTokenizer
except ImportError:
    print("请先安装transformers库: pip install transformers==4.28.1")
    exit()


class DummyTextDataset(Dataset):
    """创建一个虚拟文本数据集用于测试"""

    def __init__(self, num_samples=1000, max_length=128):
        self.num_samples = num_samples
        self.max_length = max_length
        self.texts = ["This is a sample text for testing. " * 10] * num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, 30000, (self.max_length,)),
                "attention_mask": torch.ones(self.max_length)}


def test_model(model_class, model_name, batch_size, device):
    try:
        # 使用虚拟数据避免tokenizer依赖
        dataset = DummyTextDataset()
        train_loader = DataLoader(dataset, batch_size=batch_size)

        # 创建简单模型
        model = model_class(config=model_class.config_class()).to(device)
        model.train()

        # 计时测试
        total_time = 0
        iterations = 5  # 减少迭代次数

        for i, batch in enumerate(train_loader):
            if i >= iterations:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            start_time = time.time()
            outputs = model(**batch)
            loss = outputs.last_hidden_state.mean()
            loss.backward()
            iter_time = time.time() - start_time

            total_time += iter_time
            print(f"Batch {batch_size} | Iter {i + 1}/{iterations}: {iter_time:.4f}s")

        return total_time / iterations

    except Exception as e:
        print(f"Error with {model_name} batch_size {batch_size}: {str(e)}")
        return np.nan


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_sizes = [32, 64, 128]  # 减少测试的batch_size范围
    models = {
        "BERT": BertModel,
        "RoBERTa": RobertaModel,
        # 移除了CamemBERT和XLM以减少依赖
    }

    results = {}
    for name, model_class in models.items():
        results[name] = {}
        for bs in batch_sizes:
            print(f"\nTesting {name} with batch_size {bs}")
            avg_time = test_model(model_class, name, bs, device)
            results[name][bs] = avg_time
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 打印结果
    print("\nResults:")
    print(f"{'Model':<10}", end="")
    for bs in batch_sizes:
        print(f"{f'bs={bs}':<15}", end="")
    print()

    for name, timings in results.items():
        print(f"{name:<10}", end="")
        for bs in batch_sizes:
            print(f"{timings.get(bs, 'N/A'):<15.4f}", end="")
        print()


if __name__ == "__main__":
    main()
