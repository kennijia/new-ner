import utils
import config
import logging
import numpy as np
from data_process import Processor
from data_loader import NERDataset
from model import BertNER
from train import train, evaluate
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

import warnings
import os

warnings.filterwarnings('ignore')

# -----------------
# Reproducibility
# -----------------

def _set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_env_seed = os.getenv("BERT_CRF_SEED")
if _env_seed is not None:
    try:
        config.seed = int(_env_seed)
    except ValueError:
        pass

if not hasattr(config, "seed"):
    config.seed = 42

_set_seed(int(config.seed))
logging.info(f"seed: {config.seed}")

# Optional config overrides from environment (useful for grid/ablation runs)
# NOTE: these overrides are intentionally simple and only affect a few knobs.
_env_use_dice = os.getenv("BERT_CRF_USE_DICE_LOSS")
if _env_use_dice is not None:
    config.use_dice_loss = _env_use_dice.strip().lower() in {"1", "true", "yes", "y"}

_env_dice_w = os.getenv("BERT_CRF_DICE_LOSS_WEIGHT")
if _env_dice_w is not None:
    try:
        config.dice_loss_weight = float(_env_dice_w)
    except ValueError:
        pass

_env_exclude_o = os.getenv("BERT_CRF_DICE_EXCLUDE_O")
if _env_exclude_o is not None:
    config.dice_exclude_o = _env_exclude_o.strip().lower() in {"1", "true", "yes", "y"}

_env_exp_dir = os.getenv("BERT_CRF_EXP_DIR")
if _env_exp_dir:
    # Ensure trailing slash style aligns with existing code expectations
    config.model_dir = os.path.join(_env_exp_dir, "")
    config.exp_dir = config.model_dir
    config.log_dir = os.path.join(config.model_dir, "train.log")

_env_backbone = os.getenv("BERT_CRF_BACKBONE")
if _env_backbone:
    # Override backbone path/name at runtime for quick model comparison.
    # Example:
    #   BERT_CRF_BACKBONE=/path/to/chinese-macbert-base python run.py
    config.bert_model = _env_backbone

_env_use_bilstm = os.getenv("BERT_CRF_USE_BILSTM")
if _env_use_bilstm is not None:
    config.use_bilstm = _env_use_bilstm.strip().lower() in {"1", "true", "yes", "y"}

_env_use_fgm = os.getenv("BERT_CRF_USE_FGM")
if _env_use_fgm is not None:
    config.use_fgm = _env_use_fgm.strip().lower() in {"1", "true", "yes", "y"}


def dev_split(dataset_dir):
    """split dev set"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return

    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))


def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev


def _build_hf_config():
    """Build the HuggingFace BertConfig used to initialize BertNER.

    Important: our custom model reads switches from the HF config object (not the project config.py module),
    so we must mirror relevant knobs here.
    """
    from transformers import BertConfig

    hf_cfg = BertConfig.from_pretrained(config.bert_model)
    hf_cfg.num_labels = len(config.label2id)

    # Make sure hidden states are returned (model.py expects them)
    hf_cfg.output_hidden_states = True

    # Mirror project switches into HF config so BertNER.__init__ can see them
    hf_cfg.use_bilstm = bool(getattr(config, 'use_bilstm', True))
    hf_cfg.use_dice_loss = bool(getattr(config, 'use_dice_loss', False))
    hf_cfg.dice_loss_weight = float(getattr(config, 'dice_loss_weight', 0.0))
    hf_cfg.dice_exclude_o = bool(getattr(config, 'dice_exclude_o', True))

    return hf_cfg


def run():
    """train the model"""
    # set the logger
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))

    # Log effective experiment knobs (helps debug grid runs)
    logging.info(
        "exp_dir=%s | seed=%s | backbone=%s | use_bilstm=%s | use_fgm=%s | use_dice_loss=%s | dice_loss_weight=%s | dice_exclude_o=%s",
        getattr(config, 'exp_dir', config.model_dir),
        getattr(config, 'seed', None),
        getattr(config, 'bert_model', None),
        getattr(config, 'use_bilstm', None),
        getattr(config, 'use_fgm', None),
        getattr(config, 'use_dice_loss', None),
        getattr(config, 'dice_loss_weight', None),
        getattr(config, 'dice_exclude_o', None),
    )

    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
    # 分离出验证集
    word_train, word_dev, label_train, label_dev = load_dev('train')
    # build dataset
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")

    # Prepare model
    device = config.device
    hf_cfg = _build_hf_config()
    model = BertNER.from_pretrained(config.bert_model, config=hf_cfg)
    model.to(device)

    # Prepare optimizer
    if config.full_fine_tuning:
        bert_optimizer = list(model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # Collect head parameters (optionally include BiLSTM)
        head_params = list(model.classifier.parameters()) + list(model.crf.parameters())
        if getattr(config, 'use_bilstm', True) and getattr(model, 'bilstm', None) is not None:
            head_params += list(model.bilstm.parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.bert_lr},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.bert_lr},
            {'params': head_params, 'lr': config.head_lr, 'weight_decay': config.weight_decay}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.model_dir + "/tensorboard")

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir, writer)

    # 训练结束后，加载最佳模型并运行测试
    test()
    writer.close()


if __name__ == '__main__':
    run()
