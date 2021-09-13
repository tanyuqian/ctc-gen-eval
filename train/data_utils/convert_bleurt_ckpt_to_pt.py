import os
import fire
import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn


def convert_tf_ckpt_to_pt_state_dict(checkpoint):
    """ 
    Conversion routine from 
    https://github.com/huggingface/datasets/issues/224#issuecomment-911782388
    """
    imported = tf.saved_model.load_v2(checkpoint)

    state_dict = {}
    for variable in imported.variables:
        n = variable.name
        if n.startswith('global'):
            continue
        data = variable.numpy()
        # if 'dense' in n:
        if 'kernel' in n:  # this is fix #1 - considering 'kernel' layers instead of 'dense'
            data = data.T
        n = n.split(':')[0]
        n = n.replace('/','.')
        n = n.replace('_','.')
        n = n.replace('kernel','weight')
        if 'LayerNorm' in n:
            n = n.replace('beta','bias')
            n = n.replace('gamma','weight')
        elif 'embeddings' in n:
            n = n.replace('word.embeddings','word_embeddings')
            n = n.replace('position.embeddings','position_embeddings')
            n = n.replace('token.type.embeddings','token_type_embeddings')
            n = n + '.weight'
        state_dict[n] = torch.from_numpy(data)
        
    return state_dict

def main(in_ckpt, out_path): 
    state_dict = convert_tf_ckpt_to_pt_state_dict(in_ckpt)
    
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict, out_path)
    
if __name__ == '__main__':
    fire.Fire(main)