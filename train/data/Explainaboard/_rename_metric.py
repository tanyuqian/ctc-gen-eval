import json
from collections import defaultdict
from tqdm import tqdm

with open('./_summeval_out.json') as f:
    old_out = json.load(f)

tree = lambda: defaultdict(tree)
new_out = tree()


for sys_out in tqdm(old_out.keys(), desc='Rename...'):
    new_out[sys_out]['src'] = old_out[sys_out]['src']
    new_out[sys_out]['refs'] = old_out[sys_out]['refs']
    for each_hypo in old_out[sys_out]['hypos'].keys():
        new_out[sys_out]['hypos'][each_hypo]['hypo'] = old_out[sys_out]['hypos'][each_hypo]['hypo']
        new_out[sys_out]['hypos'][each_hypo]['scores']['coherence'] = old_out[sys_out]['hypos'][each_hypo]['scores']['coherence']
        new_out[sys_out]['hypos'][each_hypo]['scores']['consistency'] = old_out[sys_out]['hypos'][each_hypo]['scores']['consistency']
        new_out[sys_out]['hypos'][each_hypo]['scores']['fluency'] = old_out[sys_out]['hypos'][each_hypo]['scores']['fluency']
        new_out[sys_out]['hypos'][each_hypo]['scores']['relevance'] = old_out[sys_out]['hypos'][each_hypo]['scores']['relevance']

        new_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_CNNDM_Consistency'] = old_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_CNNDM_Consistency']
        new_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_XSUM_Consistency'] = old_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_XSUM_Consistency']
        new_out[sys_out]['hypos'][each_hypo]['scores']['CTC_E_RoBERTa_Relevance'] = old_out[sys_out]['hypos'][each_hypo]['scores']['CTC_E_RoBERTa_Relevance']
        new_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_CNNDM_Relevance'] = old_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_CNNDM_Relevance']
        new_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_XSUM_Relevance'] = old_out[sys_out]['hypos'][each_hypo]['scores']['CTC_D_XSUM_Relevance']
        
json.dump(new_out, open('./summeval_out.json', 'w'))