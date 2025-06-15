from mytorch.utils.tokenizer import Tokenizer
from mytorch.utils.dataset import IMDBDataset

# 构建 tokenizer 并加载词表
tokenizer = Tokenizer(max_vocab_size=10000)
tokenizer.build_vocab_from_csv("data/imdb/imdb_master.csv")

# 加载数据集
train_dataset = IMDBDataset(
    root="data/imdb",
    tokenizer=tokenizer,
    max_len=200,
    train=True,
    from_csv=True,
    use_cache=True
)

# 获取样本
input_ids, label = train_dataset[0]
print("Input shape:", input_ids.shape)
print("Label:", label)
tokenizer.print_vocab(limit=10000)