# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import os

try:
	os.chdir(os.path.join('/home/andrey/proj/OpenKE/'))
	print('Current working dir:', os.getcwd())
except:
	pass

import pandas as pd
import numpy as np
from tools.explainer import Explainer
from tqdm import tqdm


# %%
dataset = 'FB15K237'
splits = 'g_2negrate_bern'
kv_model = 'FB15K237_Google_news_d300.model'
tests = [
    #emb_modell, timestamp_emb, timestamp_sfe
    ('TransE', '1906141142', '2010150748'), #test sfe timestamp with 3 relations only
    # ('TransE', '1906141142', '2109010941'),
    # ('TransE', '1906141142', '2012071951'),
    # ('Analogy', '2010091937', '2109010941')#,
    # ('Analogy', '2010091937', '2012071951'),
]


# dataset = 'NELL186'
# splits = 'g_2negrate_old_test1'# 'g_2negrate_bern'
# kv_model = 'NELL186_Google_news_d300.model'
# tests = [
#     #emb_modell, timestamp_emb, timestamp_sfe
#     # ('TransE', '1906141142', '2010150748'), #test sfe timestamp with 3 relations only
#     ('TransE', '1526711822', '2109012008'), 
#     # ('TransE', '1526711822', '2012010611'), #node relv_in = True
#     ('Analogy', '1904121223', '2109012008')
#     # ('Analogy', '1904121223', '2012010611'), #node relv_in = True
# ]


param_grid_logit = [{
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'alpha': [0.01, 0.001, 0.0001],
            'loss': ["log"],
            'penalty': ["elasticnet"],
            'max_iter': [100000],
            'tol': [1e-3],
            'class_weight': ["balanced"],
            'n_jobs': [10]
}]

params_list = [
    # {'pru:prunning':'force', 'xke:evaluate_benchmarks':False, 'pru:top_avg_rel_sim': 0.1},
    # {'pru:prunning':'force', 'xke:evaluate_benchmarks':False, 'pru:top_pop':0.1},
    # {'pru:prunning':'force', 'xke:evaluate_benchmarks':False, 'pru:top_avg_rel_sim': 0.2},
    {'pru:prunning':'force', 'xke:evaluate_benchmarks':False, 'pru:top_pop':0.2},
    # {'pru:prunning': False, 'xke:evaluate_benchmarks':False}

]

experiments = int(len(tests) * len(params_list))

i = 1

for test in tests:

    emb_model = test[0]
    timestamp_emb = test[1]
    timestamp_sfe = test[2]
    

    for params in params_list:

        top_avg = params.get('pru:top_avg_rel_sim', 'na')
        top_pop = params.get('pru:top_pop', 'na')

        print(f'Starting experiment {i}/{experiments}: {emb_model}_{timestamp_emb} / sfe{timestamp_sfe} / pru:top_avg_rel_sim:{top_avg} / pru:top_pop:{top_pop}')

        e = Explainer(dataset, 
                        emb_model, 
                        timestamp_emb, 
                        timestamp_sfe,
                        splits, 
                        method='fast')

        e.load_kv_model(kv_model)
        e.set_param_grid_logit(param_grid_logit)
        e.set_prune_dict(params)
        e.train_test_logit()

        i +=1

