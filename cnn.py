from mytorch.nn.loss import cross_entropy_loss
from mytorch.grad import no_grad
from mytorch.utils.dataset import DataLoader, MNISTDataset
from mytorch.opt import Adam
from mytorch.nn import Module, Conv2d, MaxPool2d, Linear
from mytorch.tensor import Tensor


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2d(kernel_size=2)
        self.fc = Linear(8 * 14 * 14, 10)  # 输入图像为 1x28x28，池化后为 8x14x14

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.pool(x).flatten()
        x = self.fc(x)
        return x


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def train(model: Module, dataloader: DataLoader, optimizer: Adam, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        # 将数据转换为 Tensor，保持图像形状 (N, 1, 28, 28)
        x = Tensor(data)
        y = Tensor(targets, dtype="int")

        # 前向传播
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

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
    with no_grad():
        for data, targets in dataloader:
            x = Tensor(data)  # (N, 1, 28, 28)
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
    model = SimpleCNN()
    optimizer = Adam(model.parameters(), lr=0.0001)

    # 训练模型
    for epoch in range(1, 6):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
