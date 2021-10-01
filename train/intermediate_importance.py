import os
import json
import fire
import numpy as np
from tqdm import tqdm

from data_utils.data_utils import get_test_examples
from models.discriminative_aligner import DiscriminativeAligner
from models.bert_aligner import BERTAligner
from models.bleurt_aligner import BLEURTAligner

from scipy.stats.stats import spearmanr, pearsonr, kendalltau


def get_reference_score(aligner, input_text, context, aligner_type, remove_stopwords): 
    if isinstance(input_text, list): 
        align_list = []
        r_y_tokens = []
        r_y_tokens_scores = []
        for ref in input_text: 
            if aligner_type == 'bleurt': 
                i = context
                c = ref
            else: 
                i = ref
                c = context
            
            score, r_y_tokens_ele, r_y_tokens_scores_ele = aligner.get_sing_score(
                        input_text=i,
                        context=c,
                        remove_stopwords=remove_stopwords)
            align_list.append(score)
            r_y_tokens.append(r_y_tokens_ele)
            r_y_tokens_scores.append(r_y_tokens_scores_ele)
            
        align_score = np.array(align_list).mean()
        
    else: 
        if aligner_type == 'bleurt': 
            i = context
            c = input_text
        else: 
            i = input_text
            c = context
                
        align_score, r_y_tokens, r_y_tokens_scores = aligner.get_sing_score(
                    input_text=i,
                    context=c,
                    remove_stopwords=remove_stopwords)
            
    return align_score, r_y_tokens, r_y_tokens_scores
        


def main(dataset_name='qags_xsum',
         aspect='consistency',
         aligner_type='disc',
         disc_init=None,
         bleurt_init=None,
         relevance_y_x_init=None,
         bert_model_type='roberta-large',
         bert_num_layers=None,
         dialog_context='fact_history',
         aggr_type='mean',
         remove_stopwords=False,
         n_references=11):

    if aligner_type == 'disc':
        aligner = DiscriminativeAligner.load_from_checkpoint(
            aggr_type=aggr_type, checkpoint_path=disc_init).to('cuda')
        aligner.eval()
        if relevance_y_x_init is not None: 
            aligner_y_x = (DiscriminativeAligner
                           .load_from_checkpoint(
                               aggr_type=aggr_type, 
                               checkpoint_path=relevance_y_x_init)
                           .to('cuda'))
            aligner_y_x.eval()
            
    elif aligner_type == 'bert':
        aligner = BERTAligner(
            model_type=bert_model_type,
            num_layers=bert_num_layers,
            aggr_type=aggr_type,
            lang='en',
            device='cuda')
        aligner_y_x = aligner
        
    elif aligner_type == 'bleurt': 
        aligner = BLEURTAligner(
            aggr_type=aggr_type,
            checkpoint=bleurt_init)
        if relevance_y_x_init is not None: 
            aligner_y_x = BLEURTAligner(
                aggr_type=aggr_type,
                checkpoint=relevance_y_x_init)

    examples = get_test_examples(
        dataset_name=dataset_name, 
        aspect=aspect, 
        dialog_context=dialog_context,
        n_references=n_references)

    all_preds = []
    pred_scores, true_scores = [], []
    for example in tqdm(examples, desc='Testing'):
        if isinstance(example, list):
            if aspect == 'relevance':
                align_r_y, r_y_tokens, r_y_tokens_scores = get_reference_score(
                                aligner=aligner, 
                                input_text=example[1].input_text, 
                                context=example[1].context, 
                                aligner_type=aligner_type, 
                                remove_stopwords=remove_stopwords)
                align_y_x, y_x_tokens, y_x_token_scores = aligner_y_x.get_sing_score(
                                input_text=example[0].input_text,
                                context=example[0].context,
                                remove_stopwords=remove_stopwords)
                pred_score = align_r_y * align_y_x
                    

            elif aspect == 'preservation':
                align_y_x, y_x_tokens, y_x_token_scores = aligner.get_sing_score(
                    input_text=example[0].input_text,
                    context=example[0].context,
                    remove_stopwords=remove_stopwords)
                align_x_y, x_y_tokens, x_y_token_scores = aligner.get_sing_score(
                    input_text=example[1].input_text,
                    context=example[1].context,
                    remove_stopwords=remove_stopwords)
                pred_score = (align_y_x * align_x_y) / (align_y_x + align_x_y)

            
            y_x_max_word = y_x_tokens[y_x_token_scores.index(max(y_x_token_scores))]
            y_x_min_word = y_x_tokens[y_x_token_scores.index(min(y_x_token_scores))]
            
            if aspect == 'preservation':
                x_y_max_word = x_y_tokens[x_y_token_scores.index(max(x_y_token_scores))]
                x_y_min_word = x_y_tokens[x_y_token_scores.index(min(x_y_token_scores))]

                x_y_order = sorted(range(len(x_y_token_scores)), key=lambda k: x_y_token_scores[k])

                x_y_sl = [x_y_tokens[ind] for ind in x_y_order]
            else:
                x_y_max_word = None
                x_y_min_word = None
                x_y_order = None
                x_y_sl = None

                x_y_tokens = None
                x_y_token_scores = None

            y_x_order = sorted(range(len(y_x_token_scores)), key=lambda k: y_x_token_scores[k])
                
            y_x_sl = [y_x_tokens[ind] for ind in y_x_order]

            all_preds.append({
                'context_0': example[0].context,
                'input_text_0': example[0].input_text,
                'context_1': example[1].context,
                'input_text_1': example[1].input_text,
                'pred_score': pred_score,
                'true_score': example[0].score,
                'y_x_tokens': y_x_tokens,
                'y_x_token_scores': y_x_token_scores,
                'y_x_tokens_small2large': y_x_sl,
                'y_x_max_word': y_x_max_word,
                'y_x_min_word': y_x_min_word,
                'x_y_tokens': x_y_tokens,
                'x_y_token_scores': x_y_token_scores,
                'x_y_tokens_small2large': x_y_sl,
                'x_y_max_word': x_y_max_word,
                'x_y_min_word': x_y_min_word,
                'r_y_tokens': r_y_tokens if aspect=='relevance' else None,
                'r_y_token_scores': r_y_tokens_scores if aspect=='relevance' else None
            })


            if pred_score is not None:
                pred_scores.append(pred_score)

                assert example[0].score == example[1].score
                true_scores.append(example[0].score)

        else:
            pred_score, y_x_tokens, y_x_token_scores = aligner.get_sing_score(
                input_text=example.input_text,
                context=example.context,
                remove_stopwords=remove_stopwords)

            y_x_max_word = y_x_tokens[y_x_token_scores.index(max(y_x_token_scores))]
            y_x_min_word = y_x_tokens[y_x_token_scores.index(min(y_x_token_scores))]

            y_x_order = sorted(range(len(y_x_token_scores)), key=lambda k: y_x_token_scores[k])
            y_x_sl = [y_x_tokens[ind] for ind in y_x_order]

            all_preds.append({
                'context': example.context,
                'input_text': example.input_text,
                'pred_score': pred_score,
                'true_score': example.score,
                'y_x_tokens': y_x_tokens,
                'y_x_token_scores': y_x_token_scores,
                'y_x_tokens_small2large': y_x_sl,
                'y_x_max_word': y_x_max_word,
                'y_x_min_word': y_x_min_word
            })

            if pred_score is not None:
                pred_scores.append(pred_score)
                true_scores.append(example.score)

    
    pearson_score = pearsonr(pred_scores, true_scores)[0]
    spearman_score = spearmanr(pred_scores, true_scores)[0]
    kendall_score = kendalltau(pred_scores, true_scores)[0]

    os.makedirs(f'eval_results/details/', exist_ok=True)
    output_filename = f'{dataset_name}_{aspect}_{aligner_type}'
    if aligner_type == 'bert':
        output_filename += f'_{bert_model_type}'

    output_path = f'eval_results/details/detail_{output_filename}.json'
    json.dump(all_preds, open(output_path, 'w'), indent=4)

    print(output_filename)
    print(f'#sents: {len(pred_scores)}')
    print(f'pearson: {pearson_score:.4f}')
    print(f'spearman: {spearman_score:.4f}')
    print(f'kendall: {kendall_score:.4f}')


if __name__ == '__main__':
    fire.Fire(main)