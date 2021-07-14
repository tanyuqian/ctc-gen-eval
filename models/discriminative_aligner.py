import os

import torch
from torch import nn

from pytorch_lightning import LightningModule

from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta import alignment_utils

from models.aligner import Aligner

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score

from data_utils.data_utils import TokenClassificationExample, \
    text_clean, get_words


INIT = 'roberta.large'
MAX_LENGTH = 512


class DiscriminativeAligner(Aligner, LightningModule):
    def __init__(self, aggr_type):
        Aligner.__init__(self, aggr_type)
        LightningModule.__init__(self)

        cache_dir = f'{os.getenv("HOME")}/.cache/'
        if not os.path.exists(f'{cache_dir}/{INIT}'):
            os.system(f'wget https://dl.fbaipublicfiles.com/fairseq/models/'
                      f'{INIT}.tar.gz -P {cache_dir}')
            os.system(f'tar -xzvf {cache_dir}/{INIT}.tar.gz -C {cache_dir}')

        self._roberta = RobertaModel.from_pretrained(f'{cache_dir}/{INIT}')
        self._classifier = nn.Linear(
            self._roberta.model.args.encoder_embed_dim, 2)

        self._hparams = None

    def forward(self, input_text, words, context):
        tokens = self._roberta.encode(
            input_text, context)[:MAX_LENGTH].unsqueeze(0)
        features = self._roberta.extract_features(tokens=tokens)[0]

        try:
            len_input_text = self._roberta.encode(input_text).shape[0]
            word_features = self.extract_features_aligned_to_words(
                input_text=input_text,
                words=words,
                features=features[:len_input_text])
        except:
            print(f'Error bpe-to-words, '
                  f'word_logits=None for this batch: {input_text}')
            return None

        word_logits = self._classifier(word_features)

        return word_logits

    def extract_features_aligned_to_words(self, input_text, words, features):
        bpe_toks = self._roberta.encode(input_text)

        alignment = alignment_utils.align_bpe_to_words(
            self._roberta, bpe_toks, words)
        aligned_feats = alignment_utils.align_features_to_words(
            self._roberta, features, alignment)

        assert len(words) + 2 == len(aligned_feats)

        return aligned_feats[1:-1]

    def training_step(self, batch, batch_idx):
        batch_loss = []

        for example in batch:
            loss = self.get_token_classification_loss(example)

            if loss is not None:
                batch_loss.append(loss)

        if len(batch_loss) == 0:
            return torch.tensor(0., requires_grad=True, device=self.device)

        mean_loss = 0.
        for i in range(len(batch_loss)):
            mean_loss = mean_loss + batch_loss[i] / len(batch_loss)

        return mean_loss / self._hparams['accumulate_grad_batches']

    def get_token_classification_loss(self, example):
        assert isinstance(example, TokenClassificationExample)

        word_logits = self(
            input_text=example.input_text,
            words=get_words(example.input_text),
            context=example.context)

        if word_logits is None:
            return None

        word_labels = torch.tensor(example.labels, device=self.device)

        nll_criterion = nn.CrossEntropyLoss()
        loss = nll_criterion(word_logits, word_labels)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_preds, batch_labels = [], []
        with torch.no_grad():
            for example in batch:
                if isinstance(example, TokenClassificationExample):
                    word_logits = self(
                        input_text=example.input_text,
                        words=get_words(example.input_text),
                        context=example.context)

                    if word_logits is not None:
                        batch_preds.extend(
                            torch.argmax(word_logits, dim=-1).tolist())
                        batch_labels.extend(example.labels)

        return batch_preds, batch_labels

    def validation_epoch_end(self, val_outputs):
        all_preds, all_labels = [], []
        for batch_preds, batch_labels in val_outputs:
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

        counter = torch.zeros((2, 2))
        for pred, label in zip(all_preds, all_labels):
            counter[pred][label] += 1

        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        val_acc = torch.sum(all_preds == all_labels) / all_labels.shape[0]
        val_f1_ok = f1_score(
            y_true=all_labels.tolist(), y_pred=all_preds.tolist(), pos_label=0)
        val_f1_bad = f1_score(
            y_true=all_labels.tolist(), y_pred=all_preds.tolist(), pos_label=1)

        self.print('Validation Accuracy:', val_acc)
        self.print('Validation F1-OK:', val_f1_ok)
        self.print('Validation F1-BAD:', val_f1_bad)
        self.print('Validation Matrix:', counter)

        lr = self.optimizers().state_dict()['param_groups'][0]['lr']
        self.print(f'lr = {lr}')

        self.log('val_f1', val_f1_ok + val_f1_bad)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def set_hparams(self, **kwargs):
        self._hparams = kwargs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self._hparams['weight_decay']},
            {"params": [p for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self._hparams['lr'], eps=self._hparams['adam_epsilon'])

        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self._hparams['warmup_steps'],
                num_training_steps=self._hparams['train_steps']),
            'interval': 'step'}

        return [optimizer], [lr_scheduler]

    def align(self, input_text, context):
        input_text, context = text_clean(input_text), text_clean(context)

        input_text = self._roberta.decode(
            self._roberta.encode(input_text)[:MAX_LENGTH // 2])

        word_logits = self(
            input_text=input_text,
            words=get_words(input_text),
            context=context)

        if word_logits is None:
            return None, None

        preds = torch.softmax(word_logits, dim=-1)[:, 0]

        return get_words(input_text), preds.tolist()