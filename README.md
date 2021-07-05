# ctc-gen-eval

## Preparation

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
python construct.py --dataset_name yelp --task_type transduction --target_size 100000
python construct.py --dataset_name persona_chat --task_type dialog --target_size 100000
python construct.py --dataset_name topical_chat --task_type dialog --target_size 100000
```

Constructed data will be saved in ```constructed_data/```.

### Finetune RoBERTa

```
python finetune.py \
    --dataset_name [xsum/persona_chat/...] \
    --n_epochs [1 by default] \
    --dialog_context [fact/history/fact_history/history_fact] (only effective for dialog datasets)
```

Checkpoints will be saved in ```ckpts/```.

## Test Correlation

```
python test_correlation.py \
    --dataset_name [xsum/persona_chat/...] \
    --aspect [aspect] \
    --aligner_type [disc/bert] \
    --disc_init [path_to_disc_ckpt]
    --dialog_context [fact/history/fact_history/history_fact] (only effective for dialog datasets) \
    --aggr_type [mean/sum]
```

Correlation scores will be printed in the running terminal, and evaluation scores for all examples will be saved in ```eval_results/```.
