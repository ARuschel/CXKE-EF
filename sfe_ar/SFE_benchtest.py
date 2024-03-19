# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os

try:
	os.chdir(os.path.join('/home/andrey/proj/OpenKE/'))
	print(os.getcwd())
except:
	pass

import sys
sys.path.append('../')
sys.path.insert(0, os.path.expanduser('~') + '/proj/OpenKE/')

from tools.dataset_tools import Dataset

# from sfe_ar.tools.nim_functions import nim_subgraph

import pandas as pd
import numpy as np
import os, time
from tqdm import tqdm
from sfe_ar.tools.tools import SFE
from sfe_ar.tools.helpers import generate_timestamp
from collections import Counter, defaultdict

# %%
time_stamp = generate_timestamp()

print('Time stamp is: {}'.format(time_stamp))

bench_dataset = 'NELL186' #[FB13, FB15k, FB15k-237, NELL186, WN11, WN18, WN18RR]

model = 'Analogy'
timestamp_emb = '1904121223'

project_folder = os.path.expanduser('~') + '/proj/XKE_results/{}/{}/{}/'.format(bench_dataset, model, timestamp_emb)


# %%
s = SFE(bench_dataset, 
        time_stamp, 
        project_folder)


# %%
# s.load_kv_model('word2vec/fb15k237_Google_news_d300.model')


# %%
s.set_params_dict(
    {
        'sfe:node_relv_in': False,
        # 'sfe:top_rels_reduced_graph':80
    })


# %%
s.run()

