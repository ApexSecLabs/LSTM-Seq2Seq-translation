import argparse
import torch
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_utils import get_dataloaders
from models import Encoder, Decoder, Seq2Seq
from config import Config


def beam_search_translate(model, src_tensor, src_len, trg_vocab, device, max_len, beam_size=3):
    """
    束搜索翻译单个句子
    src_tensor: (1, src_seq_len)  包含 <bos> 和 <eos>
    src_len: (1)                   源句实际长度（不含填充）
    返回: 目标词索引列表（已去除 <bos> 和 <eos>）
    """
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)

    bos_idx = trg_vocab.bos_idx
    eos_idx = trg_vocab.eos_idx

    # 每个 beam 维护 (序列, 对数概率, hidden, cell)
    beams = [([bos_idx], 0.0, hidden, cell)]
    completed = []  # 存放已完成的序列 (seq without bos/eos, log_prob)

    for _ in range(max_len):
        new_beams = []
        for seq, log_prob, h, c in beams:
            if seq[-1] == eos_idx:
                completed.append((seq[1:-1], log_prob))  # 已经结束，跳过
                continue

            input_token = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
            with torch.no_grad():
                output, h_new, c_new = model.decoder(input_token, h, c)
            log_probs = torch.log_softmax(output, dim=-1)  # (1, vocab_size)

            topk_log_probs, topk_indices = torch.topk(log_probs[0], beam_size)

            for i in range(beam_size):
                token = topk_indices[i].item()
                new_log_prob = log_prob + topk_log_probs[i].item()
                new_seq = seq + [token]
                if token == eos_idx:
                    completed.append((new_seq[1:-1], new_log_prob))
                else:
                    new_beams.append((new_seq, new_log_prob, h_new, c_new))

        if not new_beams:
            break
        # 按对数概率降序排序，保留前 beam_size 个
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # 将未完成的 beam 也加入 completed（去掉 <bos>，如果末尾有 <eos> 也要去掉）
    for seq, log_prob, _, _ in beams:
        if seq[-1] == eos_idx:
            final_seq = seq[1:-1]
        else:
            final_seq = seq[1:]  # 仅去掉 <bos>
        completed.append((final_seq, log_prob))

    # 选择概率最高的序列
    completed.sort(key=lambda x: x[1], reverse=True)
    return completed[0][0]


def translate(model, sentence, src_vocab, trg_vocab, device, max_len, beam_size=1):
    """
    交互式翻译句子
    beam_size=1 时使用贪心搜索，>1 时使用束搜索
    """
    model.eval()
    # 分词
    if src_vocab.lang_type == 'eng':
        tokens = sentence.strip().split()
    else:
        import jieba
        tokens = list(jieba.cut(sentence.strip()))

    # 转为索引并添加 <bos> 和 <eos>
    indices = [src_vocab.bos_idx] + [src_vocab.token2idx.get(tok, src_vocab.unk_idx) for tok in tokens] + [src_vocab.eos_idx]
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(tokens)]).to(device)

    if beam_size <= 1:
        # 贪心解码
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
    else:
        output_indices = beam_search_translate(model, src_tensor, src_len, trg_vocab, device, max_len, beam_size)

    translation = trg_vocab.decode(output_indices, skip_special=True)
    return ' '.join(translation) if trg_vocab.lang_type == 'eng' else ''.join(translation)


def compute_bleu(model, test_loader, src_vocab, trg_vocab, device, max_len, swap=False, beam_size=3):
    """在测试集上计算平均 BLEU 值"""
    model.eval()
    smooth = SmoothingFunction().method1
    total_bleu = 0
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if swap:
                src, trg, src_len, trg_len = batch[1], batch[0], batch[3], batch[2]
            else:
                src, trg, src_len, trg_len = batch
            src, trg = src.to(device), trg.to(device)
            src_len, trg_len = src_len.to(device), trg_len.to(device)
            batch_size = src.size(0)
            for i in range(batch_size):
                # 源语言句子（用于展示）
                src_sentence = src[i].cpu().tolist()
                src_tokens = src_vocab.decode(src_sentence, skip_special=True)
                src_str = ' '.join(src_tokens) if src_vocab.lang_type == 'eng' else ''.join(src_tokens)

                # 参考译文
                trg_indices = trg[i].cpu().tolist()
                ref_tokens = trg_vocab.decode(trg_indices, skip_special=True)
                if trg_vocab.lang_type == 'eng':
                    reference = [ref_tokens]
                else:
                    reference = [list(''.join(ref_tokens))]

                # 模型翻译
                src_input = src[i].unsqueeze(0)          # (1, seq_len)
                src_len_input = src_len[i].unsqueeze(0)  # (1)

                if beam_size <= 1:
                    # 贪心
                    hidden, cell = model.encoder(src_input, src_len_input)
                    input_token = torch.LongTensor([trg_vocab.bos_idx]).to(device).unsqueeze(0)
                    output_indices = []
                    for _ in range(max_len):
                        output, hidden, cell = model.decoder(input_token, hidden, cell)
                        top1 = output.argmax(1)
                        if top1.item() == trg_vocab.eos_idx:
                            break
                        output_indices.append(top1.item())
                        input_token = top1.unsqueeze(0)
                else:
                    output_indices = beam_search_translate(model, src_input, src_len_input, trg_vocab, device, max_len, beam_size)

                translation_tokens = trg_vocab.decode(output_indices, skip_special=True)
                if trg_vocab.lang_type == 'eng':
                    candidate = translation_tokens
                else:
                    candidate = list(''.join(translation_tokens))

                bleu = sentence_bleu(reference, candidate, smoothing_function=smooth)
                total_bleu += bleu
                count += 1

                if count <= 5:
                    print(f"源语言: {src_str}")
                    print(f"参考: {(' '.join(ref_tokens) if trg_vocab.lang_type == 'eng' else ''.join(ref_tokens))}")
                    print(f"翻译: {(' '.join(translation_tokens) if trg_vocab.lang_type == 'eng' else ''.join(translation_tokens))}")
                    print(f"BLEU: {bleu:.4f}\n")

    avg_bleu = total_bleu / count
    print(f"测试集平均BLEU: {avg_bleu:.4f}")
    return avg_bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', type=str, default='zh2en', choices=['en2zh', 'zh2en'], help="Translation direction")
    parser.add_argument('--sentence', type=str, default=None, help="Sentence to translate (e.g., 'hello world')")
    parser.add_argument('--beam_size', type=int, default=5, help="Beam size for beam search (default: 5)")
    args = parser.parse_args()

    config = Config()
    print(f"使用设备：{config.device}")
    print(f"翻译方向：{args.direction}")
    print(f"束宽：{args.beam_size}")

    train_loader, valid_loader, test_loader, eng_vocab, chi_vocab = get_dataloaders(config)

    if args.direction == 'en2zh':
        src_vocab = eng_vocab
        trg_vocab = chi_vocab
    else:
        src_vocab = chi_vocab
        trg_vocab = eng_vocab

    encoder = Encoder(len(src_vocab), config.emb_dim, config.hidden_dim, config.num_layers, config.bidirectional)
    decoder = Decoder(len(trg_vocab), config.emb_dim, config.hidden_dim, config.num_layers)
    model = Seq2Seq(encoder, decoder, len(trg_vocab), config.device).to(config.device)

    model_save_path = config.model_save_path.format(direction=args.direction)
    model.load_state_dict(torch.load(model_save_path, map_location=config.device, weights_only=True))

    if args.sentence:
        translation = translate(model, args.sentence, src_vocab, trg_vocab, config.device, config.max_length, beam_size=args.beam_size)
        print(f"输入：{args.sentence}")
        print(f"翻译：{translation}")
    else:
        print("\n测试集BLEU评估：")
        swap_test = (args.direction == 'zh2en')
        compute_bleu(model, test_loader, src_vocab, trg_vocab, config.device, config.max_length, swap=swap_test, beam_size=args.beam_size)


if __name__ == '__main__':
    main()