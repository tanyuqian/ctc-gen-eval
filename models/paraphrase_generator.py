import distance
from nltk import sent_tokenize

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MAX_LENGTH = 256
TOP_K = 120
TOP_P = 0.95


class ParaphraseGenerator:
    def __init__(self, device='cuda'):
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(
            "Vamsi/T5_Paraphrase_Paws")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            "Vamsi/T5_Paraphrase_Paws").to(self._device)

    def generate_sent(self, input_sent):
        raw_input_text = input_sent
        input_text = "paraphrase: " + input_sent + " </s>"

        model_inputs = self._tokenizer.encode_plus(
            input_text,
            max_length=MAX_LENGTH,
            padding='longest',
            return_tensors="pt").to(self._device)

        outputs = self._model.generate(
            **model_inputs,
            max_length=MAX_LENGTH,
            do_sample=True,
            top_k=TOP_K,
            top_p=TOP_P,
            early_stopping=True,
            num_return_sequences=10)

        max_dist, output_sent = -1, ''
        for output in outputs:
            output_text = self._tokenizer.decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)

            dist = distance.levenshtein(raw_input_text, output_text)

            if dist > max_dist:
                output_sent = output_text

        return output_sent

    def generate(self, input_text):
        return ' '.join([self.generate_sent(input_sent=sent)
                         for sent in sent_tokenize(input_text)])