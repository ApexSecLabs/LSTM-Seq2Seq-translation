import random
from collections import Counter
import jieba
import re
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from config import Config



def clean_text(text):
    text = text.lower().strip() #转为小写，删除首尾空白字符
    text = re.sub(r'\s+', ' ', text) #使用正则表达式将文本中任意连续的空白字符替换为单个空格，确保单词之间只有一个空格分隔。
    return text

def tokenize_eng(sentence):
    return word_tokenize(sentence) #将英文句子拆分为单词

def tokenize_chi(sentence):
    return list(jieba.cut(sentence)) #对输入的中文使用结巴分词库进行分词，按顺序产生句子中的词语

def load_and_preprocess(config):
    sentences = []
    with open(config.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = tokenize_eng(clean_text(parts[0]))
                chi = tokenize_chi(clean_text(parts[1]))
                if len(eng) <= config.max_length and len(chi) <= config.max_length:
                    sentences.append((eng,chi))
    print(f"加载句子对数量：{len(sentences)}")
    return sentences

class Vocab:
    def __init__(self, sentences, lang_type, min_freq=1):
        self.lang_type = lang_type
        self.token2idx = {'<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3}
        self.idx2token = {0:'<pad>', 1:'<bos>', 2:'<eos>', 3:'<unk>'}
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        counter = Counter()
        for eng_tokens,chi_tokens in sentences:
            tokens = eng_tokens if lang_type == 'eng' else chi_tokens
            counter.update(tokens)

        for token,freq in counter.items():
            if freq >= min_freq:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        print (f"{lang_type} 词表大小: {len(self.token2idx)}")

    def __len__(self):
        return len(self.token2idx)

    def encode(self, tokens, add_special=True):
        indices = []
        if add_special:
            indices.append(self.bos_idx)
        for token in tokens:
            indices.append(self.token2idx.get(token, self.unk_idx))
        if add_special:
            indices.append(self.eos_idx)
        return indices

    def decode(self, indices, skip_special=True):
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx,'<unk>')
            if skip_special and token in ['<pad>', '<bos>', '<eos>']:
                continue
            tokens.append(token)
        return tokens

class TranslationDataset(Dataset):
    def __init__(self, sentences, eng_vocab, chi_vocab):
        self.sentences = sentences
        self.eng_vocab = eng_vocab
        self.chi_vocab = chi_vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        eng_tokens,chi_tokens = self.sentences[idx]
        eng_indices = self.eng_vocab.encode(eng_tokens, add_special=True)
        chi_indices = self.chi_vocab.encode(chi_tokens, add_special=True)
        return torch.LongTensor(eng_indices), torch.LongTensor(chi_indices)

def collate_fn(batch):
    eng_batch, chi_batch = zip(*batch)
    eng_lengths = [len(seq) for seq in eng_batch]
    chi_lengths = [len(seq) for seq in chi_batch]

    sorted_indices = sorted(range(len(eng_lengths)), key=lambda i: eng_lengths[i], reverse=True)
    eng_batch = [eng_batch[i] for i in sorted_indices]
    chi_batch = [chi_batch[i] for i in sorted_indices]
    eng_lengths = [eng_lengths[i] for i in sorted_indices]
    chi_lengths = [chi_lengths[i] for i in sorted_indices]

    eng_padded = torch.nn.utils.rnn.pad_sequence(eng_batch, batch_first=True, padding_value=0)
    chi_padded = torch.nn.utils.rnn.pad_sequence(chi_batch, batch_first=True, padding_value=0)

    return eng_padded, chi_padded, torch.tensor(eng_lengths), torch.tensor(chi_lengths)

def get_dataloaders(config):
    sentences = load_and_preprocess(config)
    random.seed(42)
    random.shuffle(sentences)

    n = len(sentences)

    train_sents = sentences[:int(0.8*n)]
    val_sents = sentences[int(0.8*n):int(0.9*n)]
    test_sents = sentences[int(0.9*n):]

    eng_vocab = Vocab(train_sents, 'eng', config.min_freq)
    chi_vocab = Vocab(train_sents, 'chi', config.min_freq)

    train_dataset = TranslationDataset(train_sents, eng_vocab, chi_vocab)
    valid_dataset = TranslationDataset(val_sents, eng_vocab, chi_vocab)
    test_dataset = TranslationDataset(test_sents, eng_vocab, chi_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn, pin_memory=True)

    return train_loader, valid_loader, test_loader, eng_vocab, chi_vocab

if __name__ == "__main__":
    config = Config()
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 打印一些信息以验证
    print("数据加载器准备就绪")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")

    # 可选：打印一个批次的数据形状
    for eng, chi, eng_len, chi_len in train_loader:
        print("英文批次形状:", eng.shape)
        print("中文批次形状:", chi.shape)
        print("英文长度:", eng_len)
        print("中文长度:", chi_len)
        break