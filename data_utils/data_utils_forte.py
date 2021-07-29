import json
import os
from typing import Any, Iterator

from forte.data import MultiPack
from forte.data.base_pack import PackType
from forte.data.base_reader import MultiPackReader
from forte.pipeline import Pipeline
from forte.processors.base import MultiPackProcessor

from .ft.onto.ctc import Metric


class TestDatasetReader(MultiPackReader):
    def _collect(self, file_path: str) -> Iterator[Any]:
        js_path = f'data/{file_path}.json'
        yield from [js_path]

    def _parse_pack(self, file_path: str) -> Iterator[PackType]:
        print(file_path)
        js_file = json.load((open(file_path)))
        dataset_name, extension = os.path.basename(file_path).split('.')

        for each_item in js_file:
            m_pack: MultiPack = MultiPack()  # each multi-pack represents an item in a json file

            if dataset_name in ['qags_xsum', 'qags_cnndm']:
                document_pack = m_pack.add_pack('document')
                summary_pack = m_pack.add_pack('summary')

                document_pack.set_text(each_item['document'])
                summary_pack.set_text(each_item['summary'])

                met_consistency = Metric(summary_pack)
                met_consistency.metric_name = 'consistency'
                met_consistency.metric_value = each_item['consistency']

            elif dataset_name == 'summeval':
                document_pack = m_pack.add_pack('document')
                summary_pack = m_pack.add_pack('summary')
                refs_pack = m_pack.add_pack('references')

                document_pack.set_text(each_item['document'])
                summary_pack.set_text(each_item['summary'])
                refs_pack.set_text(' '.join(each_item['references']))

                met_consistency = Metric(summary_pack)
                met_consistency.metric_name = 'consistency'
                met_consistency.metric_value = each_item['consistency']

                met_relevance = Metric(summary_pack)
                met_relevance.metric_name = 'relevance'
                met_relevance.metric_value = each_item['relevance']
            elif dataset_name == 'yelp':
                input_sent_pack = m_pack.add_pack('input_sent')
                output_sent_pack = m_pack.add_pack('output_sent')

                input_sent_pack.set_text(each_item['input_sent'])
                output_sent_pack.set_text(each_item['output_sent'])

                met_preservation = Metric(output_sent_pack)
                met_preservation.metric_name = 'preservation'
                met_preservation.metric_value = each_item['preservation']
            elif dataset_name in ['persona_chat', 'topical_chat']:
                fact = 'fact: ' + each_item['fact']
                history = each_item['dialog_history']
                fact_history = '\n\n\n'.join([fact.strip(), history.strip()])
                history_fact = '\n\n\n'.join([history.strip(), fact.strip()])

                fact_pack = m_pack.add_pack('fact')
                history_pack = m_pack.add_pack('history')
                fact_history_pack = m_pack.add_pack('fact_history')
                history_fact_pack = m_pack.add_pack('history_fact')
                response_pack = m_pack.add_pack('response')

                fact_pack.set_text(fact)
                history_pack.set_text(history)
                fact_history_pack.set_text(fact_history)
                history_fact_pack.set_text(history_fact)
                response_pack.set_text(each_item['response'])

                engagingness_met = Metric(response_pack)
                groundness_met = Metric(response_pack)

                engagingness_met.metric_name = 'engagingness'
                engagingness_met.metric_value = each_item['engaging']

                groundness_met.metric_name = 'groundness'
                groundness_met.metric_value = each_item['uses_knowledge']

            else:
                raise ValueError('Unsupported dataset type: {}'.format(dataset_name))

            yield m_pack


class DiscModelProcessor(MultiPackProcessor):
    def _process(self, input_pack: MultiPack):
        print('in processor')          # do more computation

    # def initialize(self, model_path):
    #     aligner = DiscriminativeAligner.load_from_checkpoint(aggr_type='mean',
    #                                                          checkpoint_path=model_path).to('cuda')
    #     aligner.eval()


if __name__ == '__main__':
    pl = Pipeline()
    pl.set_reader(TestDatasetReader())
    pl.add(DiscModelProcessor())

    pl.initialize()
    for pack in pl.process_dataset('/home/zyh/CTC_task/ctc-gen-eval/data/summeval.json'):  # process whole dataset pack
        summ_pack = pack.get_pack('summary')
        for each_generics in summ_pack:
            print(each_generics.metric_name, each_generics.metric_value)
        # print(summ_pack.num_generics_entries)
        break

    # out_pack = pl.process_one('data/')  # only process one dataset pack
    # # pl.run()
    # tst_class = TestDatasetReader()
    # for each in tst_class._parse_pack('/home/zyh/CTC_task/ctc-gen-eval/data/summeval.json'):
    #     print(each.get_pack('document_2').)
