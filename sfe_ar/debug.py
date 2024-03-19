
#%%
import pandas as pd
import os, time
import logging
from tqdm import tqdm, tnrange
from tools.tools import SFE, GRAPH
from tools.helpers import generate_timestamp
from collections import Counter, defaultdict, deque

#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

#%%
def get_triple_ids(ent_dict, text_triple):

    entities = text_triple.split(',')
    
    left = ent_dict[entities[0]]
    right = ent_dict[entities[1]]

    return left, right

def get_path_ids(rel_dict, text_paths):

    text_paths = text_paths.split(',1.0 -#- ')


    paths = []

    for path in text_paths:
        path = path[1:-1]
        rels = path.split('-')
        path_id = ''
        for rel in rels:
            if rel[0] == '_':
                path_id = path_id + '_' + 'i' + str(rel_dict[str(rel[1:])])
            else:
                path_id = path_id + '_' + str(rel_dict[str(rel)])
        paths.append(path_id[1:])

    return paths

def check_path(graph, triple, path):
    
    start = triple[0]
    end = triple[1]
    rel = str(triple[3])
    path = path.split('_')
    path_lenght = len(path)

    step_control = deque()
    next_rel = deque()
    nodes = deque()
    visited = set({start})

    #First it is necessary to check if there is the relation comming out
    #from the start node
    if path[0] not in graph[start].keys():
        return False

    if path_lenght == 1:
        if path[0] == rel:
            return False
        
        if end in graph[start][path[0]]:
            return True
        else:
            return False

    #Initial Expansion:

    #Here we have already checked that there is path[0] comming out
    #from start node
    for node in graph[start][path[0]]:
        if node == end: #this is to avoid appending ending node to the queue
            continue
        else:
            nodes.append(node)
            visited = visited | {node}
            next_rel.append(path[1])
            step_control.append(1)

    while nodes:
        node_to_open = nodes.popleft()
        rel_to_open = next_rel.popleft()
        current_step = step_control.popleft() + 1

        if current_step < path_lenght:

            if rel_to_open in graph[node_to_open].keys():
                for node in graph[node_to_open][rel_to_open]:
                    if node not in visited:
                        nodes.append(node)
                        visited = visited | {node}
                        step_control.append(current_step)
                        next_rel.append(path[current_step])

        else:
            if rel_to_open in graph[node_to_open].keys():
                if end in graph[node_to_open][rel_to_open]:
                    return True

    return False

#%%
bench_dataset = 'NELL186' #[FB13, FB15k, FB15k-237, NELL186, WN11, WN18, WN18RR]
graph_type = 'g'
knn = 7
max_len = 2
time_stamp = 'debug'
relation = 40
allow_cycles = True
max_fan_out = 5000
ignore_initial_max_fo = True


graph = GRAPH(bench_dataset, graph_type, knn)
g = graph.graph
s = SFE(bench_dataset, graph_type, knn, time_stamp, max_len, allow_cycles, max_fan_out, ignore_initial_max_fo)

bench_path = os.path.join('~/proj/XKEc/benchmarks', bench_dataset)
        
ent = pd.read_csv(os.path.join(bench_path, 'entity2id.txt'), 
                            sep='\t', skiprows=1, header=None, 
                            names=['ent', 'id'])

rel = pd.read_csv(os.path.join(bench_path, 'relation2id.txt'), 
                            sep='\t', skiprows=1, header=None, 
                            names=['ent', 'id'])
        
ent_dict = dict(zip(ent.ent, ent.id))
rel_dict = dict(zip(rel.ent, rel.id))

rel_name = list(rel_dict.keys())[list(rel_dict.values()).index(relation)]
print(rel_name)

if graph_type == 'g':
    output_dir = os.path.join('results', bench_dataset, time_stamp, 'g_2negrate_bern__pra', rel_name)
else:
    output_dir = os.path.join('results', bench_dataset, time_stamp, 
        ('ghat_{}nn_2negrate_bern__pra').format(knn), rel_name)
#%%
df = pd.read_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', header=None, 
    names=['triple', 'label', 'features'])

dbug_log = logging.getLogger(__name__)
logging.basicConfig(filename='logs/debug_{}_{}.txt'.format(time_stamp, bench_dataset),
                level=logging.INFO,
                format='%(levelname)s: %(asctime)s %(message)s',
                datefmt='%d/%m/%Y %I:%M:%S')

#%%

dbug_log.info('Starting consitency check for features extracted.')
dbug_log.info('Dataset %s', bench_dataset)
dbug_log.info('Relation %s', rel_name)
dbug_log.info('Reading files in directory %s', output_dir)

triples_with_problem = 0

for _, row in tqdm(df.iterrows()):
    if pd.isnull(row['features']):
        continue
    triple_id = get_triple_ids(ent_dict, row['triple'])[0], get_triple_ids(ent_dict, row['triple'])[1], row['label'], relation
    features = get_path_ids(rel_dict, row['features'][:-4])
    
    inconsistencies = 0
    good = True

    for path in features:
       
        check = s.check_path(triple_id, path)
        check_naive = s.naive_check_path(triple_id, path)
        if not (check & check_naive):
            dbug_log.info('Triple %s / Path: %s / check %s check_naive %s', triple_id, path, check, check_naive)
            inconsistencies += 1
            good = False
        
    if not good:
        triples_with_problem +=1
        dbug_log.info('Triple %s has %s wrong paths out of %s', triple_id, inconsistencies, len(features) )

    
dbug_log.info('Evaluated %s triples and found %s triples with inconsistent paths', len(df), triples_with_problem)


