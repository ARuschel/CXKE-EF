#%%
import os

try:
	os.chdir(os.path.expanduser('~') + '/proj/OpenKE/')
	print(os.getcwd())
except:
	pass

import pandas as pd
import time
import logging
from tqdm import tqdm
from sfe_ar.tools.tools import SFE, GRAPH
from sfe_ar.tools.helpers import generate_timestamp
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from sfe_ar.tools.helpers import ensure_dir
import psutil

# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')

def get_dirs(dirpath):
    """Same as `os.listdir()` but ensures that only directories will be returned.
    """
    #fs = init_fs(True)
    dirs = []
    for f in os.listdir(dirpath):
        f_path = os.path.join(dirpath, f)
        dirs.append(f)
    return dirs

#%%
#Generating New Time Stamp
time_stamp = generate_timestamp()
time_stamp = '1911121432'
# time_stamp = 'debug_new'
print('Time stamp is: {}'.format(time_stamp))


#%%

bench_dataset = 'FB15K237' #[FB13, FB15k, FB15k-237, NELL186, WN11, WN18, WN18RR]
max_len = 2
allow_cycles = False
max_fan_out = 100
ignore_initial_max_fo = True #opens start/end node despite degree>max_fan_out

graph_types = [('g', 0)]#, ('g_hat', 3), ('g_hat', 5), ('g_hat', 7)]#

graph_type = 'g'
knn = 7

load_from_splits = False

s = SFE(bench_dataset, 
        graph_type, 
        knn, 
        time_stamp, 
        max_len, 
        allow_cycles, 
        max_fan_out, 
        ignore_initial_max_fo,
        load_from_splits)

#%%
model = 'TransE'
dataset = 'FB15K237'
timestamp_emb = '1906141126_d'


project_folder = os.path.expanduser('~') + '/proj/XKE_lp/results/{}/{}/{}/'.format(dataset, model, timestamp_emb)
s.output_dir = project_folder + 'sfe_features/{}/'.format(time_stamp)

lp_folder = project_folder + 'link_prediction/'

target_relations = get_dirs(lp_folder)

# target_relations = ['people-person-religion']

i = 1
s.load_pre_built_subgraphs = False
s.override_pkl = False

#%%
for rel in target_relations:

        print('\nProcessing relation {}/{}: {}\n'.format(i, len(target_relations), rel))
        i = i + 1
        fold_lp = lp_folder + rel + '/' + 'detailed/'
        tsv_files = [f for f in os.listdir(fold_lp) if f.split('.')[1] == 'tsv']
        print('Found {} tsv files'.format(len(tsv_files)))

        s.processed_triples = set([f.split('.')[0] for f in os.listdir(s.output_dir + rel + '/')])

        j=0
        if psutil.virtual_memory()[3]/1000000000 > 45:
                print('Flushing Loaded Subgraphs\n')
                s.subgraphs = dict()
     
        for f in tsv_files:
                print('Processing file {}/{}'.format(j, len(tsv_files)))
                # print('\nSubgraphs dict with {} entities'.format(len(s.subgraphs.keys())))
                s.run_single_file(fold_lp, f, top_n = 1500)
                j += 1


#%%
