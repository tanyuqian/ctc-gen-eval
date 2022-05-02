# Factual Consistency Metric

This readme describes how to load and call the metric used to compute the results in the paper \
**[TRUE: Re-evaluating Factual Consistency Evaluation](https://arxiv.org/pdf/2204.04991.pdf)**. 

## Metric Description

Our metric is based on a token classification model consisting of an `albert-xlarge-vitaminc-mnli` backbone and a classification head shared across tokens. The model was trained on self-supervision data constructed from the [TopicalChat](https://github.com/alexa/Topical-Chat) and [CNN/DM](https://huggingface.co/datasets/cnn_dailymail) datasets. The data construction process is described Appendix A of our paper [here](https://arxiv.org/pdf/2109.06379.pdf)

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
