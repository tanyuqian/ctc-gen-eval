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
| Ours (D) (CNN/DM)        |    0.5196     |    0.4208      |    0.3351     |
| Ours (D) (XSUM)          |     0.5314    |   0.4273       |    0.3414     |

| Relevance                | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (D) (CNN/DM)        |    0.2376     |     0.1292     |    0.0909     |
| Ours (D) (XSUM)          |    0.2906     |     0.2142     |    0.1523     |

### QAGS

| Consistency (Pearson)                | CNN/DM | XSUM | 
|--------------------------|---------|----------|
| Ours (D) (CNN/DM)        |    0.6292     |    0.3093      |    
| Ours (D) (XSUM)          |    0.6205     |    0.3222      |

### Yelp

| Preservation             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (D)                 |    0.4974     |     0.4952     |    0.3579     |

### TopicalChat

| Engagingness             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (D) (XSUM)          |    0.4972     |    0.4988      |    0.3675     |

| Groundedness             | Pearson | Spearman | Kendall |
|--------------------------|---------|----------|---------|
| Ours (D) (XSUM)          |    0.3437     |    0.3467      |    0.2665     |

