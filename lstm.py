from mytorch.utils.tokenizer import Tokenizer
from mytorch.utils.dataset import IMDBDataset, DataLoader
from mytorch.nn import Module, Linear
from mytorch.loss import cross_entropy_loss
from mytorch.tensor import Tensor
from mytorch.backend import xp
from mytorch.opt import Adam
from mytorch.utils.dataset.transform import to_tensor
from mytorch.grad import no_grad

from tqdm import tqdm


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(
            xp.random.randn(num_embeddings, embedding_dim) * 0.01
        )

    def forward(self, x: Tensor):
        return self.weight[x.data]  # x.data 是 int 类型的索引


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2h = Linear(input_size, 4 * hidden_size)
        self.h2h = Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        # print("x.shape:", x.shape)
        # print("h_prev.shape:", h_prev.shape)
        # print("x2h.weight.shape:", self.x2h.weight.shape)
        # print("h2h.weight.shape:", self.h2h.weight.shape)
        gates: Tensor = self.x2h(x) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)
        i = i.sigmoid()
        f = f.sigmoid()
        g = g.tanh()
        o = o.sigmoid()
        c = f * c_prev + i * g
        h = o * c.tanh()
        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, h_c=None):
        seq_len, batch_size, _ = x.shape
        if h_c is None:
            h_t = Tensor.zeros((batch_size, self.hidden_size))
            c_t = Tensor.zeros((batch_size, self.hidden_size))
        else:
            h_t, c_t = h_c
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            h_t = h_t.detach()  # Detach to prevent backprop through the entire sequence
            c_t = c_t.detach()  # Detach to prevent backprop through the entire sequence
            outputs.append(h_t)
        output = Tensor.stack(outputs, dim=0)
        return output, (h_t, c_t)


class SentimentClassifier(Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, num_classes=2):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim)
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, input_ids: Tensor):
        # input_ids: (batch, seq_len)
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        # (seq_len, batch, embed_dim)
        embedded = embedded.transpose((1, 0, 2))
        output, (h_n, _) = self.lstm(embedded)
        return self.fc(h_n)  # (batch, num_classes)


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for input_ids, label in tqdm(dataloader, desc="Training"):
        # print(input_ids)
        # print(label)
        label = to_tensor(label, dtype=xp.int32)
        logits = model(to_tensor(input_ids, dtype=xp.int32))
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with no_grad():
        for input_ids, label in tqdm(dataloader, desc="Evaluating"):
            label = to_tensor(label, dtype=xp.int32)
            logits = model(to_tensor(input_ids, dtype=xp.int32))
            loss = criterion(logits, label)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # 数据准备
    tokenizer = Tokenizer(max_vocab_size=10000)
    tokenizer.build_vocab_from_csv(
        "data/imdb/imdb_master.csv", encoding='ISO-8859-1')

    train_dataset = IMDBDataset(
        root="data/imdb",
        tokenizer=tokenizer,
        max_len=200,
        train=True,
        from_csv=True,
        use_cache=True
    )
    test_dataset = IMDBDataset(
        root="data/imdb",
        tokenizer=tokenizer,
        max_len=200,
        train=False,
        from_csv=True,
        use_cache=True
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 模型定义
    model = SentimentClassifier(vocab_size=tokenizer.vocab_size())
    criterion = cross_entropy_loss
    optimizer = Adam(model.parameters(), lr=0.0001)

    # 训练
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

    # 模型保存（可选）
    model.save("sentiment_lstm_model.pt")
    print("Model saved to sentiment_lstm_model.pt")


if __name__ == "__main__":
    main()
