import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

tokenizer = get_tokenizer("basic_english")


# 构建词汇表
def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)


def build_vocab():
    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

vocab = build_vocab()

def len_vocab():
    return len(vocab)

# 文本和标签处理
def text_pipeline(text):
    return vocab(tokenizer(text))


label_pipeline = lambda label: 1 if label == 2 else 0


# 批处理函数
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.long)
        text_list.append(processed_text)
        lengths.append(len(processed_text))

    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    lengths = torch.tensor(lengths, dtype=torch.long)
    label_list = torch.tensor(label_list, dtype=torch.float32)

    return label_list, text_list, lengths


# 加载数据
def load_data(batch_size):
    train_iter, test_iter = list(IMDB(split='train')), list(IMDB(split='test'))

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_iter, batch_size=batch_size, collate_fn=collate_batch)

    return train_loader, test_loader
