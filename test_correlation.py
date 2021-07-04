import json
import fire
from tqdm import tqdm

from data_utils.data_utils import get_test_examples
from models.discriminative_aligner import DiscriminativeAligner

from scipy.stats.stats import spearmanr, pearsonr, kendalltau


def main(dataset_name='qags_xsum',
         aspect='consistency',
         disc_init=None,
         dialog_context='fact_history',
         aggr_type='mean'):

    aligner = DiscriminativeAligner.load_from_checkpoint(
        aggr_type=aggr_type, checkpoint_path=disc_init).to('cuda')
    aligner.eval()

    examples = get_test_examples(
        dataset_name=dataset_name, aspect=aspect, dialog_context=dialog_context)

    all_preds = []
    pred_scores, true_scores = [], []
    for example in tqdm(examples, desc='Testing'):
        pred_score = aligner.get_score(
            input_text=example.input_text, context=example.context)

        all_preds.append({
            'context': example.context,
            'input_text': example.input_text,
            'pred_score': pred_score
        })

        if pred_score is not None:
            pred_scores.append(pred_score)
            true_scores.append(example.score)

    pearson_score = pearsonr(pred_scores, true_scores)[0]
    spearman_score = spearmanr(pred_scores, true_scores)[0]
    kendall_score = kendalltau(pred_scores, true_scores)[0]

    print(f'#sents: {len(pred_scores)}')
    print(f'pearson: {pearson_score:.4f}')
    print(f'spearman: {spearman_score:.4f}')
    print(f'kendall: {kendall_score:.4f}')

    json.dump(all_preds, open(f'{dataset_name}_{aspect}_{aggr_type}.json', 'w'),
              indent=4)


if __name__ == '__main__':
    fire.Fire(main)