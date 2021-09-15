ALIGNS = [
    'E-bert',
    "E-roberta",
    "D-topical_chat",
    "D-persona_chat",
    "D-cnndm",
    "D-xsum",
    "D-yelp"
    "E-topical_chat",
    "E-persona_chat",
    "E-cnndm",
    "E-xsum",
    "E-yelp"
]

E_MODEL_TYPES = {
    'roberta': 'roberta-large',
    'bert': 'bert-base-uncased'
}

DR_MODEL_LINKS = {
    "D-topical_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1K0esaW4g5renxGeHt5brjPDSgbsO5LVn/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/15syenU4xDel-cR-TbBGtrBXM1CCtvQJO/view?usp=sharing"
    },
    "D-persona_chat": {
        "fact_to_response": "https://drive.google.com/file/d/1C1NkooRdmzPyHW4Ynlboz65wSWYPYCgz/view?usp=sharing",
        "fact_history_to_response": "https://drive.google.com/file/d/1AfecB09cyUko7_NsEDy1vw6CzQPMnbaQ/view?usp=sharing"
    },
    "D-cnndm": {
        "doc_to_summ": "https://drive.google.com/file/d/1XEJpvsZUdEqrcFxmVeEaK8-dKtXZx3gM/view?usp=sharing",
        "summ_to_ref": "https://drive.google.com/file/d/1FAS2_b9elU7MMEgDb9ZCiqJGf1xd0y7e/view?usp=sharing"
    },
    "D-xsum": {
        "doc_to_summ": "https://drive.google.com/file/d/1T0uhyhjYiCnWKlbPzc2QPHCkiolNVjYJ/view?usp=sharing"
    },
    "D-yelp": {
        "sent_to_sent": "https://drive.google.com/file/d/1o-QiH_MlfJr0VYUc_b5uFtbs2QGaGc82/view?usp=sharing"
    }
}
