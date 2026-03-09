import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from torch import nn, optim
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data_utils import get_dataloaders
from config import Config
from models import Encoder, Decoder, Seq2Seq

# 设置 matplotlib 后端（避免在没有 GUI 的环境中出错）
matplotlib.use('Agg')


def greedy_decode(model, src_tensor, src_len, trg_vocab, device, max_len):
    """
    贪心解码单个句子
    src_tensor: (1, src_seq_len) 已包含 <bos> 和 <eos>
    src_len: (1) 源句实际长度
    返回: 目标词索引列表（不含 <bos> 和 <eos>）
    """
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    input_token = torch.LongTensor([trg_vocab.bos_idx]).to(device).unsqueeze(0)
    output_indices = []
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)
        top1 = output.argmax(1)
        if top1.item() == trg_vocab.eos_idx:
            break
        output_indices.append(top1.item())
        input_token = top1.unsqueeze(0)
    return output_indices


def compute_corpus_bleu(model, data_loader, src_vocab, trg_vocab, device, max_len, swap=False):
    """
    在给定的 DataLoader 上计算平均 BLEU（使用贪心解码）
    swap: 是否交换源语言和目标语言（True 表示源是中文，目标是英文）
    """
    model.eval()
    smooth = SmoothingFunction().method1
    total_bleu = 0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            if swap:
                src, trg, src_len, trg_len = batch[1], batch[0], batch[3], batch[2]
            else:
                src, trg, src_len, trg_len = batch
            src, trg = src.to(device), trg.to(device)
            src_len, trg_len = src_len.to(device), trg_len.to(device)
            batch_size = src.size(0)
            for i in range(batch_size):
                # 参考译文
                trg_indices = trg[i].cpu().tolist()
                ref_tokens = trg_vocab.decode(trg_indices, skip_special=True)
                if trg_vocab.lang_type == 'eng':
                    reference = [ref_tokens]
                else:
                    reference = [list(''.join(ref_tokens))]

                # 模型翻译
                src_input = src[i].unsqueeze(0)
                src_len_input = src_len[i].unsqueeze(0)
                output_indices = greedy_decode(model, src_input, src_len_input, trg_vocab, device, max_len)
                translation_tokens = trg_vocab.decode(output_indices, skip_special=True)
                if trg_vocab.lang_type == 'eng':
                    candidate = translation_tokens
                else:
                    candidate = list(''.join(translation_tokens))

                bleu = sentence_bleu(reference, candidate, smoothing_function=smooth)
                total_bleu += bleu
                count += 1

    avg_bleu = total_bleu / count if count > 0 else 0.0
    return avg_bleu


def train_epoch(model, dataloader, criterion, optimizer, teacher_forcing_ratio, device, swap=False):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="train"):
        if swap:
            src, trg, src_len, trg_len = batch[1], batch[0], batch[3], batch[2]
        else:
            src, trg, src_len, trg_len = batch
        src, trg = src.to(device), trg.to(device)
        src_len, trg_len = src_len.to(device), trg_len.to(device)
        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, swap=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
            if swap:
                src, trg, src_len, trg_len = batch[1], batch[0], batch[3], batch[2]
            else:
                src, trg, src_len, trg_len = batch
            src, trg = src.to(device), trg.to(device)
            src_len, trg_len = src_len.to(device), trg_len.to(device)
            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', type=str, default='zh2en', choices=['en2zh', 'zh2en'],
                        help="翻译方向：en2zh(英译汉), zh2en(汉译英)")
    parser.add_argument('--plot', action='store_true', default=True,
                        help="是否绘制训练曲线 (默认: True)")
    parser.add_argument('--plot_save_path', type=str, default='./training_curves.png',
                        help="训练曲线保存路径")
    args = parser.parse_args()
    swap_data = (args.direction == 'zh2en')
    config = Config()

    print(f"使用设备：{config.device}")
    print(f"翻译方向：{'汉译英' if args.direction == 'zh2en' else '英译汉'}")
    print(f"双向LSTM：{config.bidirectional}")
    print(f"层数：{config.num_layers}")
    print(f"Dropout：{config.dropout}")
    print(f"Weight Decay：{config.weight_decay}")
    print(f"早停耐心值：{config.early_stop_patience}")

    train_loader, valid_loader, test_loader, eng_vocab, zh_vocab = get_dataloaders(config)

    if args.direction == 'en2zh':
        src_vocab = eng_vocab
        trg_vocab = zh_vocab
    else:
        src_vocab = zh_vocab
        trg_vocab = eng_vocab

    # 初始化模型
    encoder = Encoder(len(src_vocab), config.emb_dim, config.hidden_dim,
                      config.num_layers, config.bidirectional, dropout=config.dropout)
    decoder = Decoder(len(trg_vocab), config.emb_dim, config.hidden_dim,
                      config.num_layers, dropout=config.dropout)
    model = Seq2Seq(encoder, decoder, len(trg_vocab), config.device).to(config.device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    # 记录训练曲线数据
    train_losses = []
    valid_losses = []
    valid_bleus = []

    best_valid_loss = float('inf')
    patience_counter = 0
    model_save_path = config.model_save_path.format(direction=args.direction)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 config.teacher_forcing_ratio, config.device, swap=swap_data)
        valid_loss = evaluate(model, valid_loader, criterion, config.device, swap=swap_data)
        valid_bleu = compute_corpus_bleu(model, valid_loader, src_vocab, trg_vocab,
                                         config.device, config.max_length, swap=swap_data)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_bleus.append(valid_bleu)

        print(f"Epoch {epoch+1:02} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f} | Valid BLEU: {valid_bleu:.4f}")
        scheduler.step(valid_loss)

        # 早停检查
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型至 {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"验证损失未改善，当前连续未改善次数: {patience_counter}/{config.early_stop_patience}")

        if patience_counter >= config.early_stop_patience:
            print(f"早停触发：连续 {config.early_stop_patience} 个 epoch 验证损失未改善。训练停止。")
            break

    # 训练结束，绘制曲线（使用实际训练的 epoch 数）
    if args.plot and len(train_losses) > 0:
        try:
            epochs_range = range(1, len(train_losses) + 1)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            # 损失曲线
            ax1.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
            ax1.plot(epochs_range, valid_losses, 'r-o', label='Valid Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)

            # BLEU 曲线
            ax2.plot(epochs_range, valid_bleus, 'g-o', label='Valid BLEU')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('BLEU')
            ax2.set_title('Validation BLEU')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(args.plot_save_path, dpi=150)
            print(f"训练曲线已保存至 {args.plot_save_path}")
        except Exception as e:
            print(f"绘图失败：{e}")
    else:
        print("已跳过绘图。")


if __name__ == '__main__':
    main()