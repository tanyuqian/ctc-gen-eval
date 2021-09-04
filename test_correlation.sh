# SummEval - Consistency

python test_correlation.py --dataset_name summeval --aspect consistency --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation.py --dataset_name summeval --aspect consistency --aligner_type bert --aggr_type mean
python test_correlation.py --dataset_name summeval --aspect consistency --aligner_type disc --disc_init ckpts/cnndm/disc.ckpt --aggr_type mean
python test_correlation.py --dataset_name summeval --aspect consistency --aligner_type disc --disc_init ckpts/xsum/disc.ckpt --aggr_type mean

# SummEval - Relevance

python test_correlation.py --dataset_name summeval --aspect relevance --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation.py --dataset_name summeval --aspect relevance --aligner_type bert --aggr_type mean
python test_correlation.py --dataset_name summeval --aspect relevance --aligner_type disc --disc_init ckpts/cnndm/disc.ckpt --aggr_type mean
python test_correlation.py --dataset_name summeval --aspect relevance --aligner_type disc --disc_init ckpts/xsum/disc.ckpt --aggr_type mean

# QAGS

python test_correlation.py --dataset_name qags_cnndm --aspect consistency --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation.py --dataset_name qags_cnndm --aspect consistency --aligner_type bert --aggr_type mean
python test_correlation.py --dataset_name qags_cnndm --aspect consistency --aligner_type disc --disc_init ckpts/cnndm/disc.ckpt --aggr_type mean
python test_correlation.py --dataset_name qags_cnndm --aspect consistency --aligner_type disc --disc_init ckpts/xsum/disc.ckpt --aggr_type mean

python test_correlation.py --dataset_name qags_xsum --aspect consistency --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation.py --dataset_name qags_xsum --aspect consistency --aligner_type bert --aggr_type mean
python test_correlation.py --dataset_name qags_xsum --aspect consistency --aligner_type disc --disc_init ckpts/cnndm/disc.ckpt --aggr_type mean
python test_correlation.py --dataset_name qags_xsum --aspect consistency --aligner_type disc --disc_init ckpts/xsum/disc.ckpt --aggr_type mean

# Yelp

python test_correlation.py --dataset_name yelp --aspect preservation --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation.py --dataset_name yelp --aspect preservation --aligner_type bert --aggr_type mean
python test_correlation.py --dataset_name yelp --aspect preservation --aligner_type disc --disc_init ckpts/yelp/disc.ckpt --aggr_type mean

# PersonaChat

python test_correlation.py --dataset_name persona_chat --aspect engaging --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation.py --dataset_name persona_chat --aspect engaging --aligner_type bert --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation.py --dataset_name persona_chat --aspect engaging --aligner_type disc --disc_init ckpts/persona_chat/disc_fact_history.ckpt --remove_stopwords --aggr_type sum --dialog_context fact_history

python test_correlation.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type bert --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type disc --disc_init ckpts/persona_chat/disc_fact.ckpt --remove_stopwords --aggr_type sum --dialog_context fact

# TopicalChat

python test_correlation.py --dataset_name topical_chat --aspect engaging --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation.py --dataset_name topical_chat --aspect engaging --aligner_type bert --aggr_type sum --remove_stopwords --dialog_context fact_history
python test_correlation.py --dataset_name topical_chat --aspect engaging --aligner_type disc --disc_init ckpts/topical_chat/disc_fact_history.ckpt --remove_stopwords --aggr_type sum --dialog_context fact_history

python test_correlation.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type bert --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type disc --disc_init ckpts/topical_chat/disc_fact.ckpt --remove_stopwords --aggr_type sum --dialog_context fact