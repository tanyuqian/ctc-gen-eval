import fire
from forte import Pipeline

from data_utils.data_utils_forte import TestDatasetReader
from models.ctc_processor import AlignModelProcessor, CorrelationProcessor, BaseMetricProcessor


def main(dataset_name='qags_xsum',
         aspect='consistency',
         aligner_type='bert',
         disc_init=None,
         bert_model_type='roberta-large',
         bert_rescale_with_baseline=False,
         dialog_context=None,
         aggr_type='mean', ):

    pl = Pipeline()
    pl.set_reader(TestDatasetReader())
    # pl.add(AlignModelProcessor(model_type=bert_model_type,
    #                            rescale_with_baseline=bert_rescale_with_baseline,
    #                            aggr_type=aggr_type,
    #                            lang='en',
    #                            device='cuda',
    #                            aspect=aspect,
    #                            context=dialog_context,
    #                            ckpt_path=disc_init,
    #                            aligner_type=aligner_type,
    #                            dataset_name=dataset_name))
    pl.add(BaseMetricProcessor(aspect=aspect))
    pl.add(CorrelationProcessor(aspect=aspect))

    pl.initialize()

    for pack in pl.process_dataset(dataset_name):  # process whole dataset pack
        pass

    print(f'#sents: {len(pl.components[1].pred_scores)}')
    print(f'pearson: {pl.components[1].pearson_score:.4f}')
    print(f'spearman: {pl.components[1].spearman_score:.4f}')
    print(f'kendall: {pl.components[1].kendall_score:.4f}')


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
