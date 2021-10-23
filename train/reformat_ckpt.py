import torch
import os,sys
from models.discriminative_aligner import DiscriminativeAligner
import tqdm

for dataset_name in tqdm.tqdm(os.listdir('./ckpts/')):
    disc_model = DiscriminativeAligner(aggr_type=None).to('cuda')
    ckpt_path = os.listdir(os.path.join('./ckpts/', dataset_name))
    for ckpt_name in ckpt_path:
        os.makedirs(os.path.join('/new/ckpts', dataset_name), exist_ok=True)
        ckpt = os.path.join('./ckpts/', dataset_name, ckpt_name)
        disc_model.load_state_dict(torch.load(ckpt)['state_dict'])
        torch.save(disc_model.state_dict(), os.path.join('/new/ckpts', dataset_name,ckpt_name))
    