#%%
import os

import sys
sys.path.append('../')
sys.path.insert(0, os.path.expanduser('~') + '/proj/OpenKE/')
try:
	os.chdir(os.path.join('/home/andrey/proj/OpenKE/'))
	print(os.getcwd())
except:
	pass

import os, time
from sfe_ar.tools.tools import SFE
from sfe_ar.tools.helpers import generate_timestamp
from tools.dataset_tools import Dataset



# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')

#%%
dataset = 'NELL186' #[FB13, FB15k, FB15k-237, NELL186, WN11, WN18, WN18RR]
p_folder = os.path.expanduser('~') + f'/proj/XKE_results/{dataset}/'
use_split = 'g_2negrate_old_test1'
use_sm_model = f'{dataset}_Google_news_d300.model'
# use_timestamp = '2109010941'
max_fanout = 1000



#%%
s = SFE(bench_dataset = dataset, 
        project_folder = p_folder,
		split = use_split,
		# time_stamp = use_timestamp,
        max_fanout = max_fanout
        )

s.load_kv_model(use_sm_model)

s.set_params_dict(
    {
        'sfe:node_relv_in': True# ,
        # 'sfe:top_rels_reduced_graph': 180 
    })

s.run(override_features=False)
# s.run(relations_to_run=['r0', 'r4', 'r5', 'r17', 'r20', 'r25', 'r26', 'r31', 'r43', 'r44', 'r46', 
# 'r50', 'r52', 'r59', 'r67', 'r76', 'r90', 'r98', 'r107', 'r113','r115', 'r118', 'r124', 'r126', 
# 'r132', 'r137', 'r148', 'r149', 'r151','r154', 'r156', 'r165', 'r172', 'r176', 'r181', 'r195', 
# 'r200', 'r204','r209', 'r211', 'r216', 'r217', 'r220', 'r222', 'r228', 'r229', 'r233'], override_features=True)

