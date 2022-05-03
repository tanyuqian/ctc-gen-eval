# Factual Consistency Metric

This readme describes how to load and call the metric used to compute the results in the paper \
**[TRUE: Re-evaluating Factual Consistency Evaluation](https://arxiv.org/pdf/2204.04991.pdf)**. 

## Metric Description

- Our metric is based on a token classification model consisting of an `albert-xlarge-vitaminc-mnli` backbone and a classification head shared across tokens
- The model was trained on self-supervision data constructed from the [TopicalChat](https://github.com/alexa/Topical-Chat) and [CNN/DM](https://huggingface.co/datasets/cnn_dailymail) datasets
- The data construction process is described Appendix A of our paper [here](https://arxiv.org/pdf/2109.06379.pdf)

## Usage

The routine below assumes that you have installed our library as described [here](https://github.com/tanyuqian/ctc-gen-eval). 

### Python

```python
from ctc_score import FactualConsistencyScorer

# Example from the Q2 dataset
grounding = "Ross went from being a public-television personality in the 1980s and 1990s to being an Internet celebrity in the 21st century, popular with fans on YouTube and many other websites."
hypo = "he became popular in the 1980 ' s and 1980s ."
scorer = FactualConsistencyScorer(align='D-mix-albert')

score = scorer.score(grounding=grounding, hypo=hypo)
print(score)
```

### Command Line Interface (CLI)

```commandline
ctc_score 
    --task factual_consistency
    --align D-mix-albert
    --aspect consistency
    --grounding [a_file_with_all_grounding_texts (line-by-line)]
    --hypo [a_file_with_all_hypothesized_texts_to_evaluate (line-by-line)]
    --scores_save_path [the_path_to_save_example-wise_scores]
```

## ROC AUC Results

| Dataset | Q<sup>2</sup> (Metric) | ANLI | SC (ZS) | CTC (Ours) | F1 | BLEURT | QuestEval | FactCC | BARTScore | BERTScore |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FRANK** | 87.8 | **89.4** | 89.1 | 87.5 | 76.1 | 82.8 | 84.0 | 76.4 | 86.1 | 84.3 |
| **SummEval** | 78.8 | 80.5 | **81.7** | 76.0 | 61.4 | 66.7 | 70.1 | 75.9 | 73.5 | 77.2 |
| **MNBM** | 68.7 | **77.9** | 71.3 | 72.3 | 46.2 | 64.5 | 65.3 | 59.4 | 60.9 | 62.8 |
| **QAGS-C** | **83.5** | 82.1 | 80.9 | 73.4 | 63.8 | 71.6 | 64.2 | 76.4 | 80.9 | 69.1 |
| **QAGS-X** | 70.9 | **83.8** | 78.1 | 73.1 | 51.1 | 57.2 | 56.3 | 64.9 | 53.8 | 49.5 |
| **BEGIN** | 79.7 | 82.6 | 82.0 | 77.9 | 86.4 | 86.4 | 84.1 | 64.4 | 86.3 | **87.9** |
| **Q<sup>2</sup>** | 80.9 | 72.7 | 77.4 | **85.3** | 65.9 | 72.4 | 72.2 | 63.7 | 64.9 | 70.0 |
| **DialFact** | **86.1** | 77.7 | 84.1 | 83.5 | 72.3 | 73.1 | 77.3 | 55.3 | 65.6 | 64.2 |
| **PAWS** | **89.7** | 86.4 | 88.2 | 86.0 | 51.1 | 68.3 | 69.2 | 64.0 | 77.5 | 77.5 |
| **FEVER** | 88.4 | **93.2** | ~~93.2~~ | 84.8 | 51.8 | 59.5 | 72.6 | 61.9 | 64.1 | 63.3 |
| **VitaminC** | 81.4 | **88.3** | ~~97.9~~ | 84.9 | 61.4 | 61.8 | 66.5 | 56.3 | 63.2 | 62.5 |
| **Avg (w/o VitC, FEVER)** | 80.7 | **81.5** | 81.4 | 79.4 | 63.8 | 71.4 | 71.5 | 66.7 | 72.3 | 71.4 |
