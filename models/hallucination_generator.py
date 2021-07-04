import re
import random
import spacy

from nltk.corpus import stopwords
from benepar.spacy_plugin import BeneparComponent

from models.bart import BART


MAX_LENGTH = 256
SAMPLING_TOPP = 0.95
en_stopwords = stopwords.words('english')
re_special_tokens = [
    '.', '^', '$', '*', '+', '?', '|', '(', ')', '{', '}', '[', ']']


class HallucinationGenerator:
    def __init__(self, device):
        self._device = device

        self._tokenizer = spacy.load('en')

        self._parser = spacy.load('en')
        self._parser.add_pipe(BeneparComponent('benepar_en3_large'))

        self._infiller = BART(init='bart.large').to(self._device)

    def parse(self, text):
        return list(self._parser(text).sents)

    def depth_first_search(self, root, sent_length, depth=0):
        children = list(root._.children)

        template = '' if len(children) != 0 else root.text
        layer = 0
        answers = []
        for child in children:
            child_result = self.depth_first_search(
                root=child, sent_length=sent_length, depth=depth+1)

            template = template + ' ' + child_result['template']
            layer = max(layer, child_result['layer'] + 1)
            answers.extend(child_result['answers'])

        if root.text.lower() in en_stopwords or \
                not any([ch.isalnum() for ch in root.text]):
            p_mask = 0.
        else:
            p_mask = len(root.text.split()) / sent_length / (layer + 1)

        if random.random() < p_mask:
            template = '<mask>'
            answers = [root.text]

        return {
            'template': template.strip(),
            'layer': layer,
            'answers': answers
        }

    def hallucinate_sent(self, root):
        for _ in range(5):
            result = self.depth_first_search(
                root=root, sent_length=len(root.text.split()))

            if result['template'] != '<mask>' and \
                    '<mask> <mask>' not in result['template'] and \
                    len(result['answers']) > 0:
                break
            else:
                result = None

        if result is None:
            return None

        gen_text = self._infiller.generate(
            src_texts=[result['template']],
            sampling=True,
            topp=SAMPLING_TOPP,
            max_len=MAX_LENGTH)[0]

        gen_text = self.cleantext(gen_text)

        pattern = result['template']
        for special_token in re_special_tokens:
            pattern = pattern.replace(special_token, f'\{special_token}')
        pattern = pattern.replace('<mask>', '(.*)')
        pattern = pattern + '$'

        try:
            matching = re.match(pattern=pattern, string=gen_text, flags=re.I)
        except:
            matching = None

        if matching is None:
            return None
        else:
            fillings = list(matching.groups())
            result['original_text'] = self.cleantext(root.text)
            result['gen_text'] = gen_text
            result['fillings'] = fillings

            return result

    def cleantext(self, text):
        return ' '.join([token.text for token in self._tokenizer(text)])

    def hallucinate(self, input_text):
        roots = self.parse(input_text)

        result = {key: [] for key in [
            'template', 'original_text', 'gen_text', 'answers', 'fillings']}
        for root in roots:
            sent_result = self.hallucinate_sent(root=root)
            if sent_result is None:
                return None

            for key in result:
                if key in ['answers', 'fillings']:
                    result[key].extend(sent_result[key])
                else:
                    result[key].append(sent_result[key])

        for key in ['template', 'original_text', 'gen_text']:
            result[key] = ' '.join(result[key])

        return result