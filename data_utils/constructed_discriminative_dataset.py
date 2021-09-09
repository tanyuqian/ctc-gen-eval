import json
from glob import glob

from torch.utils.data import Dataset

from data_utils.data_utils import text_clean, get_discriminative_token_labels,\
    get_context, TokenClassificationExample


class ConstructedDiscriminativeDataset(Dataset):
    def __init__(self, dataset_name, split, dialog_context=''):
        self._examples = []

        for file_path in glob(f'constructed_data/{dataset_name}/*.json'):
            for doc in json.load(open(file_path)):
                if dataset_name == "cnndm_ref": 
                    input_text, labels = get_discriminative_token_labels(
                        template=doc['template'], 
                        answers=doc['answers'],
                        fillings=doc['fillings'])
                    
                    context = text_clean(get_context(
                        constructed_doc=doc,
                        dataset_name=dataset_name,
                        dialog_context=dialog_context))
                    
                    input_text = text_clean(input_text)
                    
                    self._examples.append(TokenClassificationExample(
                        context=context, input_text=input_text, labels=labels))
                else:
                    input_text, labels = get_discriminative_token_labels(
                        template=doc['template'],
                        answers=doc['answers'],
                        fillings=doc['fillings'])

                    context = text_clean(get_context(
                        constructed_doc=doc,
                        dataset_name=dataset_name,
                        dialog_context=dialog_context))

                    input_text = text_clean(input_text)

                    self._examples.append(TokenClassificationExample(
                        context=context, input_text=input_text, labels=labels))

        cut = int(0.9 * len(self._examples))
        self._examples = self._examples[:cut] if split == 'train' \
            else self._examples[cut:]

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]
