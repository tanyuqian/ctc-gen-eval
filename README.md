# ctc-gen-eval-forte

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
By default, Constructed data will be saved in `./constructed_data/`, pre-trained models will be saved in `./ckpts/`

## Test Correlation with Forte

```
python test_correlation_ft.py \
    --dataset_name [qags_cnndm/persona_chat/...] \
    --aspect [aspect] \
    --aligner_type [disc/bert] \
    --disc_init [path_to_disc_ckpt] \
    --bert_model_type [roberta-large/bert-base-uncased] \
    [--bert_rescale_with_baseline] \ 
    --dialog_context [fact/history/fact_history/history_fact] \
    --aggr_type [mean/sum]
```

Correlation scores will be printed in the running terminal.
