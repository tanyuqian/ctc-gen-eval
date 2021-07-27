import json
from typing import Any, Iterator

from forte.data import MultiPack, MultiPackLink
from forte.data.base_pack import PackType
from forte.data.base_reader import MultiPackReader
from forte.pipeline import Pipeline
from forte.processors.base import MultiPackProcessor

from ontology.ft.onto.ctc import metrics


class TestDatasetReader(MultiPackReader):
    def _collect(self, file_path: str) -> Iterator[Any]:
        yield from [file_path]

    def _parse_pack(self, file_path: str) -> Iterator[PackType]:
        print(file_path)
        js_file = json.load((open(file_path)))
        i = 0
        for each_item in js_file:
            m_pack: MultiPack = MultiPack()
            document = each_item['document'] if 'document' in each_item.keys() else None
            summary = each_item['summary'] if 'summary' in each_item.keys() else None
            consistency = each_item['consistency'] if 'consistency' in each_item.keys() else None
            references = each_item['references'] if 'references' in each_item.keys() else None
            # all_elements = [document, summary, consistency, references,relevance,  # qags_xsum/cnndm/summeval
            #                 fact, history, response, engagingness, groundness,  # persona_chat/topical_chat
            #                 input_sent, output_sent, preservation  # yelp
            #                 ]

            document_pack = m_pack.add_pack("document")
            summary_pack = m_pack.add_pack("summary")

            document_pack.set_text(document)
            summary_pack.set_text(summary)
            metrics(summary_pack, 0, len(summary))

            # link_attr = MultiPackLink(m_pack, document_pack)

            # document_span = Document(document_pack, 0, len(document_pack.text))
            # summary_span = Document(summary_pack, 0, len(summary_pack.text))
            # link_doc2sum = MultiPackLink(m_pack, document_span, summary_span)
            # link_doc2sum.get_parent()
            # link_doc2sum.get_child()

            yield m_pack

    # @staticmethod
    # def read_json_item(each_item) -> Dict[str, Any]:
    #     js_item = dict()
    #     document = each_item['document'] if 'document' in each_item.keys() else None
    #     summary = each_item['summary'] if 'summary' in each_item.keys() else None
    #     consistency = each_item['consistency'] if 'consistency' in each_item.keys() else None
    #     references = each_item['references'] if 'references' in each_item.keys() else None  # list
    #     relevance = each_item['relevance'] if 'relevance' in each_item.keys() else None
    #
    #     fact = each_item['fact'] if 'fact' in each_item.keys() else None
    #     history = each_item['dialog_history'] if 'dialog_history' in each_item.keys() else None
    #     response = each_item['response'] if 'response' in each_item.keys() else None
    #     engagingness = each_item['engaging'] if 'engaging' in each_item.keys() else None
    #     groundness = each_item['uses_knowledge'] if 'uses_knowledge' in each_item.keys() else None
    #
    #     input_sent = each_item['input_sent'] if 'input_sent' in each_item.keys() else None
    #     output_sent = each_item['output_sent'] if 'output_sent' in each_item.keys() else None
    #     preservation = each_item['preservation'] if 'preservation' in each_item.keys() else None
    #
    #     return pass


class ExampleProcessor(MultiPackProcessor):
    def _process(self, input_pack: MultiPack):
        for link in input_pack.get(MultiPackLink):
            link.get_parent()  ## sent1
            link.get_child()  ## sent1_tgt
            # do more computation


if __name__ == '__main__':
    pl = Pipeline()
    pl.set_reader(TestDatasetReader())
    pl.add(ExampleProcessor())

    pl.initialize()
    for pack in pl.process_dataset('/home/zyh/CTC_task/ctc-gen-eval/data/summeval.json'):  # process whole dataset pack
        for a in pack.get_pack('document').get(metrics):
            print('this line')
            print(a,'a')
            type(a,'a')
            break
        break

    # out_pack = pl.process_one('data/')  # only process one dataset pack
    # # pl.run()
    # tst_class = TestDatasetReader()
    # for each in tst_class._parse_pack('/home/zyh/CTC_task/ctc-gen-eval/data/summeval.json'):
    #     print(each.get_pack('document_2').)
