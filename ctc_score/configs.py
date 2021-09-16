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
        "fact_to_response": "https://drive.google.com/file/d/111a9F482StWnxxQma3AVRWZGeHdNKgTP/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/15syenU4xDel-cR-TbBGtrBXM1CCtvQJO/view?usp=sharing"
    },
    "D-persona_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1g3ZrQRwM-8yFlThCyOiSPVbJCjdc80oy/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1AfecB09cyUko7_NsEDy1vw6CzQPMnbaQ/view?usp=sharing"
    },
    "D-cnndm": {
        "doc_to_summ": "https://drive.google.com/file/d/1XEJpvsZUdEqrcFxmVeEaK8-dKtXZx3gM/view?usp=sharing",
        "summ_to_ref": "https://drive.google.com/file/d/1Jd2axVY2lmkG0NDsj1pJ6ZhbVPoPNxad/view?usp=sharing"
    },
    "D-xsum": {
        "doc_to_summ": "https://drive.google.com/file/d/1T0uhyhjYiCnWKlbPzc2QPHCkiolNVjYJ/view?usp=sharing"
    },
    "D-yelp": {
        "sent_to_sent": "https://drive.google.com/file/d/1o-QiH_MlfJr0VYUc_b5uFtbs2QGaGc82/view?usp=sharing"
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
