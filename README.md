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

## Discriminative Model Training

### Data Construction

```
python construct.py --dataset_name xsum --task_type summ --target_size 10000
python construct.py --dataset_name cnndm --task_type summ --target_size 10000
python construct.py --dataset_name yelp --task_type transduction --target_size 10000
python construct.py --dataset_name persona_chat --task_type dialog --target_size 100000
python construct.py --dataset_name topical_chat --task_type dialog --target_size 100000
```

Constructed data will be saved in ```constructed_data/```.

### Finetune RoBERTa

```
python finetune.py \
    --dataset_name [xsum/persona_chat/...] \
    --n_epochs [1 by default] \
    --dialog_context [fact/history/fact_history/history_fact]
```

Checkpoints will be saved in ```ckpts/```.

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

| Relevance                | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.3773 |  0.3437  | 0.2464  |
| Ours (E) (RoBERTa-large)  |  0.5222    | 0.4992  |  0.3674  |  
| Ours (D) (CNN/DM)        |  0.2828 |     0.2012     |  0.1433 |
| Ours (D) (XSUM)          |    0.2906     |     0.2142     |    0.1523     |

### QAGS

| Consistency (Pearson)                | CNN/DM | XSUM |
|--------------------------|---------|----------|
| Ours (E) (BERT-base)     | 0.6083  |  0.1436  | 
| Ours (E) (RoBERTa-large) |  0.6091  |  0.0548 |    
| Ours (D) (CNN/DM)        |  0.6188 | 0.3085 |
| Ours (D) (XSUM)          |    0.6205     |    0.3222      |

### Yelp

| Preservation             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.5147 |  0.5168  |  0.3737 |
| Ours (E) (RoBERTa-large) |   0.5142    | 0.5152  | 0.3753   | 
| Ours (D)                 |    0.4974     |     0.4952     |    0.3579     |

### PersonaChat

| Engagingness | Pearson | Spearman | Kendall |
| ------------ | ------- | -------- | ------- |
| Ours (E) (BERT-base)     |  0.5003 |   0.5490 | 0.4193  |
| Ours (E) (RoBERTa-large)        |  0.4036 |  0.4437  |  0.3317 |
| Ours (D)     |   0.5132      |    0.5688      |    0.4312     |

| Groundedness | Pearson | Spearman | Kendall |
| ------------ | ------- | -------- | ------- |
| Ours (E) (BERT-base)     |  0.5761 |  0.5683  |  0.4492 |
| Ours (E) (RoBERTa-large)        | 0.3713  |  0.3608  | 0.2826  |
| Ours (D)     |    0.4733     |    0.4547      |    0.3564     |

### TopicalChat

| Engagingness             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     |  0.4514 |  0.4527  |  0.3296 |
| Ours (E) (RoBERTa-large)        |  0.4512 |  0.4508  |  0.3306 |
| Ours (D)                 |    0.4972     |    0.4988      |    0.3675     |

| Groundedness             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (E) (BERT-base)     | 0.3956  |  0.3705  | 0.2900  |
| Ours (E) (RoBERTa-large)        |  0.3137 |   0.3142 | 0.2418  |
| Ours (D)                 |    0.3437     |    0.3467      |    0.2665     |
