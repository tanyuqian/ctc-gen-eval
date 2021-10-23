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
        "fact_to_response": "https://drive.google.com/file/d/1C5__8Xda9GzEQ29W4aFm3F3WReoPn5Zc/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/18oSTjxmLDERCHCNA91CLkcoT4Vi7PTS0/view?usp=sharing"
    },
    "D-persona_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1mQqdVPzO9uIkZkIbj834tXtl3Azi-noA/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1pDAnLmEulJXuq6HYT9-_4U4tCaoLlJL_/view?usp=sharing"
    },
    "D-cnndm": {
        "doc_to_summ": "https://drive.google.com/file/d/1ABC0uNd8GJFBoNVASsl0gGktiNI5-kXW/view?usp=sharing",
        "summ_to_ref": "https://drive.google.com/file/d/1O7X98C60Rt89_WeTfBziJUqMN3tY7bUl/view?usp=sharing"
    },
    "D-xsum": {
        "doc_to_summ": "https://drive.google.com/file/d/1nFH16-7HLtpjxmTUx0QvwIw_NbaYpDsG/view?usp=sharing"
    },
    "D-yelp": {
        "sent_to_sent": "https://drive.google.com/file/d/1deuVD8u6ZFIQZwpaZL9v0zcW4r6d37Vf/view?usp=sharing"
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
