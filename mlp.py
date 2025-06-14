from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.linear import Linear
from mytorch.opt import Optimizer, SGD
from mytorch.dataset import DataLoader, MNISTDataset

import time


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x


def cross_entropy_loss(logits, targets):
    log_probs = logits.log_softmax(dim=1)
    batch_size = targets.shape[0]
    loss = -log_probs[range(batch_size), targets].mean()
    return loss


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def train(model: Module, dataloader: DataLoader, optimizer: Optimizer, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        # 将数据转换为Tensor
        x = Tensor([d.flatten() for d in data])  # shape: (batch_size, 784)
        y = Tensor(targets, dtype="int")         # shape: (batch_size,)

        # 前向传播
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录
        total_loss += loss.item()
        total_acc += accuracy(logits, y).item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    print(
        f"Epoch {epoch} Summary: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")


def test(model: Module, dataloader: DataLoader):
    model.eval()
    total_acc = 0
    with Tensor.no_grad():
        for data, targets in dataloader:
            x = Tensor([d.flatten() for d in data])
            y = Tensor(targets, dtype="int")

            logits = model(x)
            total_acc += accuracy(logits, y).item()

    avg_acc = total_acc / len(dataloader)
    print(f"Test Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    # 加载数据
    train_dataset = MNISTDataset(
        root="./data/mnist", train=True, from_csv=True)
    test_dataset = MNISTDataset(
        root="./data/mnist", train=False, from_csv=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型和优化器
    model = MLP()
    optimizer = SGD(model.parameters(), lr=0.01)

    # 训练模型
    for epoch in range(1, 6):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
