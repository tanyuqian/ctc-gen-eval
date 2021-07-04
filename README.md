# ctc-gen-eval

## Preparation

```
pip install -r requirements.txt
python -m spacy download en
```

## Discriminative Model

### Data Construction

```
python construct.py --dataset_name xsum --task_type summ
python construct.py --dataset_name cnndm --task_type summ
python construct.py --dataset_name yelp --task_type transduction
python construct.py --dataset_name persona_chat --task_type dialog
python construct.py --dataset_name topical_chat --task_type dialog
```

### Finetune

```
python finetune.py --dataset_name persona_chat --dialog_context history_fact
```