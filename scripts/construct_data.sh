python construct.py --dataset_name xsum --task_type summ --target_size 10000
python construct.py --dataset_name cnndm --task_type summ --target_size 10000
python construct.py --dataset_name cnndm_ref --task_type summ_ref --target_size 10000
python construct.py --dataset_name yelp --task_type transduction --target_size 10000
python construct.py --dataset_name persona_chat --task_type dialog --target_size 100000
python construct.py --dataset_name persona_chat_fact --task_type dialog_fact --target_size 10000
python construct.py --dataset_name topical_chat --task_type dialog --target_size 100000
python construct.py --dataset_name topical_chat_fact --task_type dialog_fact --target_size 10000
