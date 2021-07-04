import os

import torch
from torch import nn

import pytorch_lightning as pl

from transformers import BartTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from fairseq.models.bart import BARTModel
from fairseq.sequence_generator import SequenceGenerator
from fairseq.search import Sampling


class BART(pl.LightningModule):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def __init__(self, init, text_logger=None):
        super(BART, self).__init__()

        assert init in ['bart.large', 'bart.large.cnn']

        cache_dir = f'{os.getenv("HOME")}/.cache/'
        if not os.path.exists(f'{cache_dir}/{init}'):
            os.system(f'wget https://dl.fbaipublicfiles.com/fairseq/models/'
                      f'{init}.tar.gz -P {cache_dir}')
            os.system(f'tar -xzvf {cache_dir}/{init}.tar.gz -C {cache_dir}')

        self._model = BARTModel.from_pretrained(f'{cache_dir}/{init}').model

        self._hparams = None

        self._text_logger = text_logger

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        return self._model(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=prev_output_tokens)

    def training_step(self, batch, batch_idx):
        logits, _ = self(
            src_tokens=batch['src_tokens'],
            src_lengths=batch['src_lengths'],
            prev_output_tokens=batch['prev_output_tokens'])

        nll_criterion = nn.CrossEntropyLoss(
            ignore_index=BART.tokenizer.pad_token_id)

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch['prev_output_tokens'][:, 1:].contiguous()

        loss = nll_criterion(
            shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1))

        if self._hparams['label_smooth'] > 0:
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            loss = (1. - self._hparams['label_smooth']) * loss + \
                   self._hparams['label_smooth'] * torch.mean(-log_probs)

        return loss / self._hparams['accumulate_grad_batches']

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            logits, _ = self(
                src_tokens=batch['src_tokens'],
                src_lengths=batch['src_lengths'],
                prev_output_tokens=batch['prev_output_tokens'])

            nll_criterion = nn.CrossEntropyLoss(
                ignore_index=BART.tokenizer.pad_token_id)

            shift_logits = logits[:, :-1].contiguous()
            shift_labels = batch['prev_output_tokens'][:, 1:].contiguous()

            loss = nll_criterion(
                shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1))

            return loss.item()

    def validation_epoch_end(self, val_step_outputs):
        val_loss = sum(val_step_outputs) / len(val_step_outputs)

        self.print('Validation Loss:', val_loss)
        self.log('val_loss', val_loss)

        lr = self.optimizers().state_dict()['param_groups'][0]['lr']
        self._text_logger.info(
            f'epoch {self.current_epoch}, lr = {lr}, val loss = {val_loss}')

    def set_hparams(self, **kwargs):
        self._hparams = kwargs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self._hparams['weight_decay']},
            {"params": [p for n, p in self._model.named_parameters()
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

    def generate(self, src_texts, tgt_prefix='',
                 min_len=1, max_len=tokenizer.model_max_length,
                 beam_size=1, len_penalty=1., no_repeat_ngram_size=0,
                 sampling=False, topk=-1, topp=-1.,):
        if sampling:
            search_strategy = Sampling(
                tgt_dict=self.tgt_dict, sampling_topk=topk, sampling_topp=topp)
        else:
            search_strategy = None

        generator = SequenceGenerator(
            models=[self._model],
            tgt_dict=self.tgt_dict,
            min_len=min_len,
            max_len_b=max_len,
            beam_size=beam_size,
            search_strategy=search_strategy,
            len_penalty=len_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size)

        src_tokenization = BART.tokenizer(
            src_texts, padding=True, truncation=True, return_tensors='pt')

        src_tokens = src_tokenization['input_ids']
        src_lengths = torch.sum(src_tokenization['attention_mask'], dim=-1)

        prefix_tokens = BART.tokenizer(tgt_prefix)['input_ids'][:-1]
        prefix_tokens = torch.tensor([prefix_tokens] * src_tokens.shape[0])

        outputs = generator.generate(
            models=[self._model],
            sample={'net_input': {
                'src_tokens': src_tokens.to(self.device),
                'src_lengths': src_lengths.to(self.device)}},
            prefix_tokens=prefix_tokens.to(self.device))

        gen_texts = []
        for output in outputs:
            gen_text = BART.tokenizer.decode(output[0]['tokens'])

            assert gen_text.startswith('<s>')
            gen_text = gen_text[len('<s>'):]

            assert gen_text.startswith(tgt_prefix)
            gen_text = gen_text[len(tgt_prefix):]

            if gen_text.endswith('</s>'):
                gen_text = gen_text[:-len('</s>')]

            gen_texts.append(gen_text)

        return gen_texts

    @property
    def tgt_dict(self):
        return self._model.decoder.dictionary