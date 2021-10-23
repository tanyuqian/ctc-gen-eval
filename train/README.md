# CTC: A Unified Framework for Evaluating Natural Language Generation

## Requirements

Our python version is ```3.7```. Run these commands before you start:
```
cd data/
unzip topical_chat.zip
unzip yelp.zip
```

```
pip install -r requirements.txt
python -m spacy download en
```

## Alignment Model Training

The constructed data and trained models can be directed downloaded [here](https://drive.google.com/drive/folders/1Vgt0eOhxwubf-2ukEhkIBHcPbnLd0_uL?usp=sharing). If you want to run the process, see below.

### Data Construction
The script to construct data for our discriminative (D) and regression (R) model: [scripts/construct_data.sh](scripts/construct_data.sh). 

Constructed data will be saved in ```constructed_data/```.

### Training Discriminative Model (D)

The script to train the discriminative models: [scripts/train_discriminative.sh](scripts/train_discriminative.sh).

Checkpoints will be saved in ```ckpts/```.

### Training Regression Model (R)
The script to train the regression models: [scripts/train_regression.sh](scripts/train_regression.sh).

Checkpoints will be saved in ```xxx/```.


## Test Correlation
The script to test alignment models: [scripts/test_correlation.sh](scripts/test_correlation.sh) (instance-level) & [scripts/test_correlation_system.sh](scripts/test_correlation_system.sh) (system-level). 

Correlation scores will be printed in the running terminal, and evaluation scores for all examples will be saved in ```eval_results/```.

## Results
All result numbers can be found in the appendix of our paper.  
(exactly the same numbers would be got with [our trained models](https://drive.google.com/drive/folders/1IxqDRKjE1XJzPvAVpvFunyG3InSRofxN?usp=sharing).)