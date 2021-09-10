# ctc-gen-eval

## Preparation

```
cd data/
unzip topical_chat.zip
unzip yelp.zip
```

```
pip install -r requirements.txt
python -m spacy download en
```

and remove [this line96](https://github.com/pytorch/fairseq/blob/v0.10.0/fairseq/models/roberta/alignment_utils.py#L96) in Fairseq (a too strict assertion):
```python
assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
```

## Alignment Model Training

### Data Construction
The following commands construct the alignment datasets suitable for training alignment models for discriminative model (D) and regression (R) approaches in the paper:

```
python construct.py --dataset_name xsum --task_type summ --target_size 10000
python construct.py --dataset_name cnndm --task_type summ --target_size 10000
python construct.py --dataset_name cnndm_ref --task_type summ_ref --target_size 10000
python construct.py --dataset_name yelp --task_type transduction --target_size 10000
python construct.py --dataset_name persona_chat --task_type dialog --target_size 100000
python construct.py --dataset_name persona_chat_fact --task_type dialog_fact --target_size 10000
python construct.py --dataset_name topical_chat --task_type dialog --target_size 100000
python construct.py --dataset_name topical_chat_fact --task_type dialog_fact --target_size 10000
```

Constructed data will be saved in ```constructed_data/```.

### Training RoBERTa
The following command trains a `RoBERTa-large` token classifier for the discriminative model (D) approach:

```
python finetune.py \
    --dataset_name [xsum/persona_chat/...] \
    --n_epochs [1 by default] \
    --dialog_context [fact/history/fact_history/history_fact]
```

Checkpoints will be saved in ```ckpts/```.

### Convert Dataset for Regression (R) Approach
The following commands converts our alignment datasets for training our models for the regression (R) approach:

```
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/xsum/example.json --dataset_name xsum --aggr_type mean --train_pct 0.9
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/cnndm/example.json --dataset_name cnndm --aggr_type mean --train_pct 0.9
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/cnndm_ref/examples.json --dataset_name cnndm_ref --aggr_type mean --reverse_cand_ref --train_pct 0.9 
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/yelp/example.json --dataset_name cnndm --aggr_type mean --train_pct 0.9
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/persona_chat/example.json --dataset_name persona_chat --aggr_type sum --remove_stopwords --dialog_context fact_history --train_pct 0.9
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/persona_chat_fact/examples.json --dataset_name persona_chat_fact --aggr_type sum --remove_stopwords --dialog_context fact --train_pct 0.9 
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/topical_chat/example.json --dataset_name topical_chat --aggr_type sum --remove_stopwords --dialog_context fact_history --train_pct 0.9
python convert_constructed_data_to_bleurt_format.py --data_path constructed_data/topical_chat_fact/examples.json --dataset_name topical_chat_fact --aggr_type sum --remove_stopwords --dialog_context fact --train_pct 0.9 
```

### Training BERT-base-midtrained with BLEURT
The following command trains our regression (R) model with the [`BLEURT`](https://github.com/google-research/bleurt) codebase:

```
BERT_CKPT=path/to/bert/checkpoint
MY_BLEURT_CKPT=target/path/for/trained/model
TRAIN_PATH=path/to/training/data
DEV_PATH=path/to/dev/data
python -m bleurt.finetune \
  -init_checkpoint=${BERT_CKPT}/bert-base \
  -bert_config_file=${BERT_CKPT}/bert_config.json \
  -vocab_file=${BERT_CKPT}/vocab.txt \
  -model_dir=${MY_BLEURT_CKPT} \
  -train_set=${TRAIN_PATH} \
  -dev_set=${DEV_PATH} \
  -num_train_steps=15000 \
  -max_seq_length=512 \
  -batch_size=6 \
  -do_lower_case=true \
  -export_metric=correlation \
  -keep_checkpoint_max=15
```


### Download Constructed Data and Trained Models
Constructed data and trained models can be downloaded via ```download_model_data.py```

Usage:
```
python download_model_data.py \
    --download_type [all/data/model] \
    --cons_data [all/cnndm/xsum/yelp/persona_chat/topical_chat] \
    --data_path (default:)constructed_data/ \
    --model_name [all/cnndm/xsum/yelp/persona_chat/topical_chat] \
    --model_path (default:)ckpts/ \
    --context [fact/history/fact_history/history_fact] \
```

## Test Correlation

```
python test_correlation.py \
    --dataset_name [qags_cnndm/persona_chat/...] \
    --aspect [aspect] \
    --aligner_type [disc/bert] \
    --disc_init [path_to_disc_ckpt] \
    --bert_model_type [roberta-large/bert-base-uncased] \
    [--bert_rescale_with_baseline] \ 
    --dialog_context [fact/history/fact_history/history_fact] \
    --aggr_type [mean/sum]
```

Correlation scores will be printed in the running terminal, and evaluation scores for all examples will be saved in ```eval_results/```.


## Results

### SummEval

| Consistency              | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     | 0.4359  |  0.3744  |  0.2974 |
| Ours (E) (RoBERTa-large) |  0.3315     | 0.3202  |  0.2518  |   
| Ours (D) (CNN/DM)        |  0.5240 |   0.4293 |  0.3422 |
| Ours (D) (XSUM)          |     0.5314    |   0.4273       |    0.3414     |
| Ours (R) (CNN/DM)          |     0.4626    |   0.3871       |    0.3071     |
| Ours (D) (XSUM)          |     0.4868    |   0.3896       |    0.3094     |

| Relevance (11 Refs)      | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.3906 |  0.3547  | 0.2544  |
| Ours (E) (RoBERTa-large)  |  0.5198    | 0.4990  |  0.3671  |  
| Ours (D) (CNN/DM)        |  0.4423 |     0.3962     |  0.2862 |
| Ours (D) (XSUM)          |    0.4426     |     0.3991     |    0.2878     |
| Ours (R) (CNN/DM)        |  0.4115 |     0.3617     |  0.2644 |
| Ours (R) (XSUM)          |    0.4121     |     0.3680     |    0.2687     |

| Relevance (1 Ref)      | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.3635 |  0.3359  | 0.2401  |
| Ours (E) (RoBERTa-large)  |  0.4985    | 0.4882  |  0.3563  |  
| Ours (D) (CNN/DM)        |  0.3824 |     0.3499     |  0.2528 |
| Ours (D) (XSUM)          |    0.3802     |     0.3502     |    0.2526     |
| Ours (R) (CNN/DM)        |  0.3733 |     0.3439     |  0.2495 |
| Ours (R) (XSUM)          |    0.3714     |     0.3445     |    0.2493     |

### QAGS

| Consistency (Pearson)                | CNN/DM | XSUM |
|--------------------------|---------|----------|
| Ours (E) (BERT-base)     | 0.6083  |  0.1436  | 
| Ours (E) (RoBERTa-large) |  0.6091  |  0.0548 |    
| Ours (D) (CNN/DM)        |  0.6188 | 0.3085 |
| Ours (D) (XSUM)          |    0.6205     |    0.3222      |
| Ours (R) (CNN/DM)        |  0.6468 | 0.2157 |
| Ours (R) (XSUM)          |    0.6612     |    0.2718      |

### Yelp

| Preservation             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.5147 |  0.5168  |  0.3737 |
| Ours (E) (RoBERTa-large) |   0.5142    | 0.5152  | 0.3753   | 
| Ours (E) (RoBERTa-large-MNLI-9) |   0.5216    | 0.5236  | 0.3805   | 
| Ours (D)                 |    0.4974     |     0.4952     |    0.3579     |
| Ours (R)                 |    0.5060     |     0.5059     |    0.3645     |

### PersonaChat

| Engagingness | Pearson | Spearman | Kendall |
| ------------ | ------- | -------- | ------- |
| Ours (E) (BERT-base)     |  0.5003 |   0.5490 | 0.4193  |
| Ours (E) (RoBERTa-large)        |  0.4036 |  0.4437  |  0.3317 |
| Ours (D) (PersonaChat)    |   0.5265      |    0.5793      |    0.4412     |
| Ours (D) (TopicalChat)    |   0.5317      |    0.5818      |    0.4409     |
| Ours (R) (PersonaChat)    |   0.5320      |    0.5692      |    0.4346     |
| Ours (R) (TopicalChat)    |   0.4933      |    0.5333      |    0.4043     |

| Groundedness | Pearson | Spearman | Kendall |
| ------------ | ------- | -------- | ------- |
| Ours (E) (BERT-base)     |  0.5761 |  0.5683  |  0.4492 |
| Ours (E) (RoBERTa-large)        | 0.3758  |  0.3652  | 0.2862  |
| Ours (D) (PersonaChat)    |   0.5683      |    0.5674      |    0.4505     |
| Ours (D) (TopicalChat)    |   0.4056      |    0.4172      |    0.3270     |
| Ours (R) (PersonaChat)    |   0.6597      |    0.6689      |    0.5338     |
| Ours (R) (TopicalChat)    |   0.6819      |    0.7113      |    0.5636     |

### TopicalChat

| Engagingness             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.4937 |  0.5047  |  0.3710 |
| Ours (E) (RoBERTa-large)        |  0.4471 |  0.4479  |  0.3288 |
| Ours (D) (PersonaChat)    |   0.5124      |    0.5245      |    0.3878     |
| Ours (D) (TopicalChat)    |   0.5163      |    0.5253      |    0.3270     |
| Ours (R) (PersonaChat)    |   0.4542      |    0.4588      |    0.3357     |
| Ours (R) (TopicalChat)    |   0.4653      |    0.4643      |    0.3395     |

| Groundedness             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     | 0.4293  |  0.3949  | 0.3075  |
| Ours (E) (RoBERTa-large)        |  0.3122 |   0.3141 | 0.2421  |
| Ours (D) (PersonaChat)    |   0.3697      |    0.3691      |    0.2856     |
| Ours (D) (TopicalChat)    |   0.3099      |    0.3159      |    0.2421     |
| Ours (R) (PersonaChat)    |   0.4026      |    0.3788      |    0.3137     |
| Ours (R) (TopicalChat)    |   0.5235      |    0.4768      |    0.3838     |
