TASKS = [
    'style_transfer',
    'summarization',
    'dialog'
]

ALIGNS = [
    "E-bert",
    "E-roberta",
    "E-roberta-mnli",
    "D-topical_chat",
    "D-persona_chat",
    "D-cnndm",
    "D-xsum",
    "D-yelp",
    "R-topical_chat",
    "R-persona_chat",
    "R-cnndm",
    "R-xsum",
    "R-yelp"
]

E_MODEL_CONFIGS = {
    'roberta': {'model_type': 'roberta-large'},
    'bert': {'model_type': 'bert-base-uncased'},
    'roberta-mnli': {'model_type': 'roberta-large-mnli', 'num_layers': 9}
}

DR_MODEL_LINKS = {
    "D-topical_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1n91TuO1TFsM5ksVuclE2h6tBrHATSgC4/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1wLZEPd5K4H-F_wAAjIjQwyXDrzPO9TIg/view?usp=sharing"
    },
    "D-persona_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1hqF8x4UoPvtCdzTus04UiNVFAZBN32Ak/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1mgtJKxtSCUNqJ1PITVtjLTXpve4woemf/view?usp=sharing"
    },
    "D-cnndm": {
        "doc_to_summ": "https://drive.google.com/file/d/1vahFzi2k74KRtQ6dxXzE5vYLQkartO9S/view?usp=sharing",
        "summ_to_ref": "https://drive.google.com/file/d/1G7-_RDeHNt2vzouK_sUkFRRvepmWiXID/view?usp=sharing"
    },
    "D-xsum": {
        "doc_to_summ": "https://drive.google.com/file/d/11y1FzAb0XcyDi2yDZXWDB3kpfc7BEXK8/view?usp=sharing"
    },
    "D-yelp": {
        "sent_to_sent": "https://drive.google.com/file/d/1Fw9Nibmu8-VhdwhPBJSoICgcnqmrxf6C/view?usp=sharing"
    },
    
    "R-topical_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1viv8deMc0HS2GPfUwCZDhacr0LAkjWrm/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1ThFlhvcUyuJtGZh3B9M42W9d6-YEjQmG/view?usp=sharing"
    },
    "R-persona_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1VuaAZbik8XuosPOjsBzbiaYl3zv-3Iuq/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1VAEKwzkJNxWCaRqLlSu5eoq1TBvqHMqN/view?usp=sharing"
    },
    "R-cnndm": {
        "doc_to_summ": "https://drive.google.com/file/d/1gJ_YcDxz880mCtctOagxRMihBU-HFniS/view?usp=sharing",
        "summ_to_ref": "https://drive.google.com/file/d/16y2ZhBzSVUHNnTEN_vHc_9J5pzAVonKI/view?usp=sharing"
    },
    "R-xsum": {
        "doc_to_summ": "https://drive.google.com/file/d/1-nf7ob2GjR7kbCjhZtIIDB6kT7EY8pHx/view?usp=sharing"
    },
    "R-yelp": {
        "sent_to_sent": "https://drive.google.com/file/d/1mcXvNzeLyTTIEHRsUPmMSB8xvuLXcTrJ/view?usp=sharing"
    }
}
