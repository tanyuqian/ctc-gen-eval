import fire
from forte import Pipeline

from data_utils.data_utils_forte import TestDatasetReader
from models.ctc_processor import DiscModelProcessor, BertModelProcessor

from scipy.stats.stats import spearmanr, pearsonr, kendalltau


def main(dataset_name='qags_xsum',
         aspect='consistency',
         aligner_type='bert',
         disc_init=None,
         bert_model_type='roberta-large',
         bert_rescale_with_baseline=False,
         dialog_context='fact_history',
         aggr_type='mean', ):
    pred_scores = []
    true_scores = []

    pl = Pipeline()
    pl.set_reader(TestDatasetReader())
    if aligner_type == 'disc':
        pl.add(DiscModelProcessor(ckpt_path=disc_init,
                                  aggr_type=aggr_type,
                                  device='cuda',
                                  aspect=aspect))
    elif aligner_type == 'bert':
        pl.add(BertModelProcessor(model_type=bert_model_type,
                                  rescale_with_baseline=bert_rescale_with_baseline,
                                  aggr_type=aggr_type,
                                  lang='en',
                                  device='cuda',
                                  aspect=aspect,
                                  context=dialog_context))
    else:
        raise ValueError('Aligner type unknown: {}'.format(aligner_type))

    pl.initialize()

    for pack in pl.process_dataset(dataset_name):  # process whole dataset pack
        if aspect in ['consistency', 'relevance']:
            ans_pack = pack.get_pack('summary')
        elif aspect in ['preservation']:
            ans_pack = pack.get_pack('output_sent')
        elif aspect in ['engagingness', 'groundness']:
            ans_pack = pack.get_pack('response')
        generics = list(ans_pack.all_generic_entries)
        gen_dict = dict()
        for each_generic in ans_pack.all_generic_entries:
            gen_dict[each_generic.metric_name] = each_generic.metric_value
        # print(gen_dict)
        if gen_dict['pred_'+aspect] is not None:
            pred_scores.append(gen_dict['pred_'+aspect])
            true_scores.append(gen_dict[aspect])

    pearson_score = pearsonr(pred_scores, true_scores)[0]
    spearman_score = spearmanr(pred_scores, true_scores)[0]
    kendall_score = kendalltau(pred_scores, true_scores)[0]

    print(f'#sents: {len(pred_scores)}')
    print(f'pearson: {pearson_score:.4f}')
    print(f'spearman: {spearman_score:.4f}')
    print(f'kendall: {kendall_score:.4f}')


if __name__ == '__main__':
    fire.Fire(main)

# Disc, qags_xsum, consistency
# sents: 239
# pearson: 0.3166
# spearman: 0.3049
# kendall: 0.2495

# Bert (roberta-large), qags_xsum, consistency
# sents: 239
# pearson: 0.0548
# spearman: 0.0489
# kendall: 0.0400
