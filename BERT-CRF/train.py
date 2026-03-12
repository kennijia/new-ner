import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import config
from model import BertNER
from metrics import f1_score, bad_case
from transformers import BertTokenizer
from utils import FGM, EMA


def train_epoch_with_ema(train_loader, model, optimizer, scheduler, epoch, ema):
    # set model to training mode
    model.train()
    
    train_losses = 0
    fgm = FGM(model) if config.use_fgm else None

    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0) 
        batch_masks[:, 0] = True
        
        # 基础训练逻辑
        if config.use_rdrop:
            # Forward two passes for R-Drop
            loss1, logits1 = model((batch_data, batch_token_starts),
                                    token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            loss2, logits2 = model((batch_data, batch_token_starts),
                                    token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            
            # CE Loss average
            loss = (loss1 + loss2) / 2
            
            # KL divergence for R-Drop
            p = F.log_softmax(logits1, dim=-1)
            q = F.log_softmax(logits2, dim=-1)
            p_tec = F.softmax(logits1, dim=-1)
            q_tec = F.softmax(logits2, dim=-1)
            
            # Mask out padding for KL (CRF labels usually -1 for padding)
            kl_mask = batch_labels.gt(-1)
            kl_loss = (F.kl_div(p, q_tec, reduction='none') + F.kl_div(q, p_tec, reduction='none')) / 2
            kl_loss = (kl_loss.sum(dim=-1) * kl_mask).sum() / kl_mask.sum()
            
            loss = loss + config.rdrop_alpha * kl_loss
        else:
            loss, logits = model((batch_data, batch_token_starts),
                                  token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        
        loss = loss / config.gradient_accumulation_steps
        train_losses += loss.item() * config.gradient_accumulation_steps
        loss.backward()

        # FGM Adversarial Training
        if fgm is not None:
            # Update: RoBERTa/BERT usually use 'word_embeddings'
            fgm.attack(epsilon=config.fgm_epsilon, emb_name='word_embeddings')
            loss_adv, _ = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            loss_adv = loss_adv / config.gradient_accumulation_steps
            loss_adv.backward()
            fgm.restore()

        # 达到累积步数后更新参数
        if (idx + 1) % config.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()
            scheduler.step()
            # 更新 EMA 影子权重
            ema.update()
            model.zero_grad()
        
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    return train_loss


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir, writer=None):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    
    # 统一管理 EMA 实例
    ema = EMA(model, config.ema_decay)
    ema.register()

    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        # 传递 ema 实例到 train_epoch 内部进行 update
        train_loss = train_epoch_with_ema(train_loader, model, optimizer, scheduler, epoch, ema)
        
        # 评估前应用 EMA 影子权重
        ema.apply_shadow()
        val_metrics = evaluate(dev_loader, model, mode='dev')
        # 评估后恢复原始权重，以便继续训练
        ema.restore()

        val_f1 = val_metrics['f1']
        val_loss = val_metrics['loss']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_loss, val_f1))
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/dev', val_loss, epoch)
            writer.add_scalar('F1/dev', val_f1, epoch)
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            # 在保存之前先应用 EMA 的权重，这样保存的模型就是 EMA 的平均权重
            ema.apply_shadow()
            model.save_pretrained(model_dir)
            # 保存完之后再恢复，以便继续使用原始权重训练
            ema.restore()
            logging.info("--------Save best model (with EMA weights)!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    
    # 应用 EMA (如果有的话，通常在测试阶段应用)
    # 注意：为了逻辑严密，我们需要在 train.py 全局创建一个 ema 实例
    # 这里我们临时通过手动方式模拟，更严谨的做法是在 train 里持有它。
    
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            
            # 同时计算 loss 和 logits，减少一次前向传播
            outputs = model((batch_data, batch_token_starts),
                             token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            loss, logits = outputs[0], outputs[1]
            dev_losses += loss.item()
            
            # 直接使用解出的一对进行解码
            batch_output = model.crf.decode(logits, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics


if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)
