# config.py
import torch


class Config:
    # 数据处理部分的参数
    data_file = "./data/cmn_new.txt"  # 训练预料存放路径
    max_length = 50  # 模型处理的最大tokens长度。超过此长度的删除
    min_freq = 1  #
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bidirectional = True
    hidden_dim = 512
    emb_dim = 512
    num_layers = 3
    lr = 0.001
    teacher_forcing_ratio = 0.5
    model_save_path = "./model/best_model_{direction}.pth"
    epochs = 20
    beam_size = 5
    dropout = 0.5
    weight_decay = 1e-5
    early_stop_patience = 3