
import pandas as pd
import time, os
# import git
import numpy as np
import scipy.sparse as sp
import random
import itertools
import logging
import _pickle as pickle
from collections import deque, defaultdict, Counter
from tqdm import tqdm, tnrange
import multiprocessing as mp
from multiprocessing import Queue
from sklearn.feature_extraction import DictVectorizer

import sys
sys.path.insert(0, os.path.expanduser('~') + '/proj/OpenKE/')
sys.path.append('../')
from tools.dataset_tools import Dataset
from tools.tools import get_dirs, write_to_pkl
from sfe_ar.tools.helpers import generate_timestamp


class SFE(Dataset):
    def __init__(self,
                bench_dataset, 
                project_folder,
                split,
                time_stamp = None,
                max_fanout = False):
       
        print("\nInitializing SFE module!\n")
        # self.current_repo = git.Repo(search_parent_directories=True)
        # self.sha = self.current_repo.head.object.hexsha

        if time_stamp == None:
            self.time_stamp = generate_timestamp()
            print(f'sfe_time_stamp created: {self.time_stamp}.\n')
        else:
            self.time_stamp = time_stamp
            print(f'Using provided sfe_time_stamp: {time_stamp}.\n')

        self.bench_dataset = bench_dataset
        self.project_folder = project_folder
        self.bench_path = os.path.expanduser('~') + f'/proj/OpenKE/benchmarks/{self.bench_dataset}/'
        self.splits_folder = self.project_folder + f'splits/{split}/'
        self.output_dir = project_folder + f'sfe_features/{self.time_stamp}/'
        self.sm_folder = self.project_folder + 'sm_models/'
        self.max_fanout = max_fanout
        
        self.subgraphs = dict()
        self.load_pre_built_subgraphs = False
        self.override_pkl = True
        
        super().__init__(bench_dataset)

        self.load_true_sets()
        self.build_graph()
        self.corrupted_queue = Queue()
        self.counter = Queue()
        self.processed_triples = set()

        self.thresholds = dict()
        self.params_dict = dict()
        self.model_info = dict()
        self.emb_model_info = dict()
        # self.load_emb_model_info()
        # self.set_subgraph_engine()

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='logs/{}_{}.txt'.format(self.time_stamp, self.bench_dataset),
                level=logging.INFO,
                format='%(levelname)s: %(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S')
        self.logger.info('\nInitializing SFE class with parameters:\nDataset %s', self.bench_dataset)
        self.logger.info('Results will be saved to: %s', self.output_dir)

    def write_to_pkl(self, filenm, object_to_save):

        file_name = '{}.pkl'.format(filenm)
        # print('Writing to Dict file '.format(filenm))

        with open(file_name, 'wb') as f:
            pickle.dump(object_to_save, f)

    def load_from_pkl(self, filenm):

        file_name = '{}.pkl'.format(filenm)
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        
        return obj

    
    # def set_subgraph_engine(self):

    #     if self.params_dict.get('sfe:node_relv_in', False):
    #         self.subgraph_engine = self.context_aware_subgraph
    #         print('Using context_aware_subgraph to build subgraphs.')
    #     else:
    #         self.subgraph_engine = self.semi_naive_subgraph
    #         print('Using unconstrained subgraph build')

    def build_queue(self, rel, subset):

        self.logger.info('Building Queue...')

        count = 0
        for triple in self.corrupted_dict[rel][subset]:

            triple_id = str(triple[0]) + '_' + str(triple[1]) + '_' + str(triple[2]) + '_' + str(triple[3])
            self.corrupted_queue.put(triple_id)
            count += 1
        
        print('\nQueue filled with {} training examples.'.format(self.corrupted_queue.qsize()))
        self.logger.info('\nQueue filled with %s triples.', self.corrupted_queue.qsize())


    def set_params_dict(self, params_dict):

        self.params_dict = params_dict

    def load_emb_model_info(self):

        self.emb_model_info = dict(pd.read_csv(self.project_folder + 'model_info.tsv', sep='\t').T)[0]

#############################################################################
# New working functions

    def invert_path(self, path):

        spath = path.split('_')

        int_path = []
        for rel in spath:
                if rel[0] == 'i':
                        int_path.append(rel[1:])
                else:
                        int_path.append('i' + rel)

        int_path = int_path[::-1]

        out_path = ''
        for el in int_path:
                out_path = out_path + '_' + el        
        
        return out_path[1:]

    def invert_rel(self, rel):

        rel = rel

        if rel[0] == 'i':
            return rel[1:]
        else:
            return 'i' + rel

    def select_top_similar_rels(self, rel, topn = 20):

        selected_rels = self.sm_model.most_similar(rel, topn = 3000)

        return [e[0] for e in selected_rels if (e[0][0] != 'e') & (e[0][0] != 'i') ][:topn]

    def semi_naive_subgraph(self, start, tail , rel):
        ''' This function expands a subgraph departing from a starting node.
            Each first dict key represents the size of the paths
            Paths are grouped by ending node represented by the second level
            dict key.

        '''

        path_queue = deque()

        output = defaultdict()
        output[1] = defaultdict(list)
        output[2] = defaultdict(list)
        

        #initial expansion
        for key in self.graph[start].keys():
    
            nodes = self.graph[start][key]
            for el in nodes:
                path_queue.append([key] + [el])
                output[1][el] += [[key] ]
                
        while path_queue:
            path_to_expand = path_queue.popleft()
            node_to_open = path_to_expand[-1]
            for key in self.graph[node_to_open].keys():
                frontier = self.graph[node_to_open][key]
                if self.max_fanout:
                    if len(frontier) > self.max_fanout:
                        frontier = random.sample(frontier, self.max_fanout)
                for el in frontier:
                    if el == start:
                        continue
                    output[2][el] += [path_to_expand + [key] ]
   
        return output

    def context_aware_subgraph(self, start, tail, rel, theta=0.5):
        ''' This function expands a subgraph departing from a starting node.
            Each first dict key represents the size of the paths
            Paths are grouped by ending node represented by the second level
            dict key.

            This function uses similarity measures to add or skip
            nodes and relations

        '''

        path_queue = deque()

        output = defaultdict()
        output[1] = defaultdict(list)
        output[2] = defaultdict(list)
        
        try:
            h_t_context = self.ent_similarity_matrix(int(start[1:]), int(tail[1:]))
        except:
            h_t_context = 0

        #initial expansion
        for key in self.graph[start].keys():
            
            nodes = self.graph[start][key]
            for el in nodes:
                try:
                    if ((theta * self.ent_similarity_matrix[int(el[1:]), int(start[1:])] + (1-theta) * self.ent_similarity_matrix[int(el[1:]), int(tail[1:])]) >= h_t_context):
                        path_queue.append([key] + [el])
                        output[1][el] += [[key] ]
                except:
                    continue

        while path_queue:
            path_to_expand = path_queue.popleft()
            node_to_open = path_to_expand[-1]
            for key in self.graph[node_to_open].keys():

                frontier = self.graph[node_to_open][key]

                if self.max_fanout:
                    if len(frontier) > self.max_fanout:
                        frontier = random.sample(frontier, self.max_fanout)

                for el in frontier:
                    try:
                        if ((theta * self.ent_similarity_matrix[int(el[1:]), int(start[1:])] + (1-theta) * self.ent_similarity_matrix[int(el[1:]), int(tail[1:])] ) >= h_t_context):
                            if el == start:
                                continue
                            output[2][el] += [path_to_expand + [key] ]
                    except:
                        continue
   
        return output

    def sim_merge_subgraphs(self, triple_full, process):

        triple = triple_full.split('_')

        head = triple[0]
        tail = triple[1]
        label = triple[2]
        rel = triple[3]

        theta = 0.5
        h_t_similarity = self.ent_similarity_matrix[int(head[1:]), int(tail[1:])]

        left_subgraphs = self.subgraph_engine(head, tail, rel)
        right_subgraphs = self.subgraph_engine(tail, head, rel)

        left_nodes1 = left_subgraphs[1].keys()
        left_nodes2 = left_subgraphs[2].keys() 

        # right_subgraphs = self.semi_naive_subgraph(tail)

        right_nodes1 = right_subgraphs[1].keys()
        right_nodes2 = right_subgraphs[2].keys()

        feature, relv , avg_relv = [], [], []

        #handle paths of size 1
        if tail in left_nodes1:
            for path in left_subgraphs[1][tail]:
                if path[0] == rel:
                    continue

                feature.append(path[0])
                relv.append(1)
                avg_relv.append(1)


        #handle paths of size 2
        if tail in left_nodes2:
            for path in left_subgraphs[2][tail]:
                
                feature.append(path[0] + '_' + path[2])
                n1 = theta * self.ent_similarity_matrix[int(head[1:]), int(path[1][1:])] + (1 - theta) * self.ent_similarity_matrix[int(tail[1:]), int(path[1][1:])]
                relv.append(n1)
                avg_relv.append(n1)

        #handle paths of size 3
        i_nodes = left_nodes2 & right_nodes1
        for node in i_nodes:
            
            for left_path in left_subgraphs[2][node]:
                
                #this will avoid loops passing through the tail
                if left_path[1] == tail: 
                    continue
                
                for right_path in right_subgraphs[1][node]:
                    
                    feature.append(left_path[0] + '_' + left_path[2] + '_'  + self.invert_rel(right_path[0]))
                    n1 = theta * self.ent_similarity_matrix[int(head[1:]), int(left_path[1][1:])] + (1 - theta) * self.ent_similarity_matrix[int(tail[1:]), int(left_path[1][1:])]
                    n2 = theta * self.ent_similarity_matrix[int(head[1:]), int(node[1:])] + (1 - theta) * self.ent_similarity_matrix[int(tail[1:]), int(node[1:])]
                    relv.append(min(n1, n2))
                    avg_relv.append((n1 + n2) / 2)

        #handle paths of size 4
        i_nodes = left_nodes2 & right_nodes2

        for node in i_nodes:
            
            for left_path in left_subgraphs[2][node]:
                
                #this will avoid loops passing throug the tail:
                if left_path[1] == tail: 
                    # if left_path[0] == rel:
                    continue
                
                for right_path in right_subgraphs[2][node]:
                    
                    #this will avoid loops passing through the head
                    if right_path[1] == head:
                        # if right_path[0] == self.invert_rel(rel):
                        continue

                    #this will avoid paths when 1st node is common for both sides
                    if left_path[1] == right_path[1]:
                        continue

                    if left_path[1] == node:
                        continue

                    if right_path[1] == node:
                        continue

                    feature.append(left_path[0] + '_' + left_path[2] + '_' + self.invert_rel(right_path[2]) + '_' + self.invert_rel(right_path[0]))

                    n1 = theta * self.ent_similarity_matrix[int(head[1:]), int(left_path[1][1:])] + (1 - theta) * self.ent_similarity_matrix[int(tail[1:]), int(left_path[1][1:])]
                    n2 = theta * self.ent_similarity_matrix[int(head[1:]), int(node[1:])] + (1 - theta) * self.ent_similarity_matrix[int(tail[1:]), int(node[1:])]
                    n3 = theta * self.ent_similarity_matrix[int(head[1:]), int(right_path[1][1:])] + (1 - theta) * self.ent_similarity_matrix[int(tail[1:]), int(right_path[1][1:])]

                    relv.append(min(n1, n2, n3))
                    avg_relv.append((n1+ n2+ n3) / 3 )

        #Dataframe columns definition:
        cols = ['path',	'min_node_sim',	'avg_node_sim',	'h_t_sim', 'node_relv_in', 'node_relv_out',
        	'min_rel_sim', 	'avg_rel_sim', 	'max_node_sim', 'min_avg_node_sim',	'max_avg_node_sim']


        #If there is no path connecting h and t will return an empty dataframe
        if len(feature) == 0:
            return pd.DataFrame(columns=cols)
        
        df = pd.DataFrame()
        df['path'] = feature
        df['min_node_sim'] = relv
        df['avg_node_sim'] = avg_relv

        df['h_t_sim'] = h_t_similarity
        df['node_relv_in'] = df['min_node_sim'] > h_t_similarity
        df['node_relv_out'] = df['min_node_sim'] < h_t_similarity

        # #Calculate rel similarity measurements
        # unique_paths = list(set(feature))
        # rel_sims = dict()
        # min_rel_path = dict()
        # avg_rel_path = dict()

        # for u_path in unique_paths:

        #     rel_sim_list = []

        #     for edge in u_path.split('_'):

        #         r_sim = rel_sims.get(edge, False)

        #         if not r_sim:
                    
        #             r_sim = self.sm_model.similarity(edge, rel)
        #             rel_sims[edge] = r_sim

        #         rel_sim_list.append(r_sim)
            
        #     min_rel_path[u_path] = min(rel_sim_list)
        #     avg_rel_path[u_path] = np.mean(rel_sim_list)

        # df['min_rel_sim'] = df.path.map(min_rel_path)
        # df['avg_rel_sim'] = df.path.map(avg_rel_path)

        return df

    def fast_merge_subgraphs(self, triple_full, process):

        triple = triple_full.split('_')

        head = triple[0]
        tail = triple[1]
        rel = triple[3]

        left_subgraphs = self.subgraph_engine(head, tail, rel)
        right_subgraphs = self.subgraph_engine(tail, head, rel)

        left_nodes1 = left_subgraphs[1].keys()
        left_nodes2 = left_subgraphs[2].keys() 

        right_nodes1 = right_subgraphs[1].keys()
        right_nodes2 = right_subgraphs[2].keys()

        # feature = set()
        feature = dict()

        #handle paths of size 1
        if tail in left_nodes1:
            for path in left_subgraphs[1][tail]:
                if path[0] == rel:
                    continue

                feature[path[0]] = 1

        #handle paths of size 2
        if tail in left_nodes2:

            for path in left_subgraphs[2][tail]:
                
                feature[path[0] + '_' + path[2]] = 1

        #handle paths of size 3
        i_nodes = left_nodes2 & right_nodes1

        for node in i_nodes:
            
            for left_path in left_subgraphs[2][node]:
                
                #this will avoid loops passing through the tail
                if left_path[1] == tail: 
                    continue
                
                for right_path in right_subgraphs[1][node]:
                    
                    feature[left_path[0] + '_' + left_path[2] + '_'  + self.invert_rel(right_path[0])] = 1

        #handle paths of size 4
        i_nodes = left_nodes2 & right_nodes2

        for node in i_nodes:
            
            for left_path in left_subgraphs[2][node]:
                
                #this will avoid loops passing throug the tail:
                if left_path[1] == tail: 
                    # if left_path[0] == rel:
                    continue
                
                for right_path in right_subgraphs[2][node]:
                    
                    this_path = left_path[0] + '_' + left_path[2] + '_' + self.invert_rel(right_path[2]) + '_' + self.invert_rel(right_path[0])

                    if this_path in feature: continue

                    #this will avoid loops passing through the head
                    if right_path[1] == head:
                        # if right_path[0] == self.invert_rel(rel):
                        continue

                    #this will avoid paths when 1st node is common for both sides
                    if left_path[1] == right_path[1]:
                        continue

                    if left_path[1] == node:
                        continue

                    if right_path[1] == node:
                        continue

                    feature[this_path] = 1

        #If there is no path connecting h and t will return an empty dataframe

        return feature

    def group_features(self, df, triple_id):

        #The following columns will be used to calculate different rwp metrics
        df['max_node_sim'] = df['min_node_sim']
        df['min_avg_node_sim'] = df['avg_node_sim']
        df['max_avg_node_sim'] = df['avg_node_sim']

        # Consolidating all information within a dataframe indexed by path type
        out = df.pivot_table(['min_node_sim', 'max_node_sim', 'min_avg_node_sim', 'max_avg_node_sim', 'h_t_sim', 'node_relv_in', 'node_relv_out'], index='path', 
            aggfunc={'min_node_sim':'min', 'max_node_sim':'max', 'min_avg_node_sim':'min', 'max_avg_node_sim':'max', 'h_t_sim':'first', 'node_relv_in':'sum', 'node_relv_out':'sum'}).reset_index()
        out['triple_id'] = triple_id
        # self.logger.info('Triple: {} | Shape: {}'.format(triple_id, out.shape))
        out = out[['triple_id', 'path', 'min_node_sim', 'max_node_sim', 'min_avg_node_sim', 'max_avg_node_sim', 'h_t_sim', 'node_relv_in', 'node_relv_out']]
        
        return out
           
    def compile_subsets(self, rel_id):

            start_time = time.time()

            rel = self.rel_dict[rel_id]

            print('\n\nFetching features for {} relation:'.format(rel))
            rel_folder = self.output_dir + rel + '/'
            train_triples = pd.read_csv(self.splits_folder + rel + '/train.tsv', sep='\t')
            test_triples = pd.read_csv(self.splits_folder + rel + '/test.tsv', sep='\t')
            valid_triples = pd.read_csv(self.splits_folder + rel + '/valid.tsv', sep='\t')

            #Dataset Statistics
            self.model_info[rel_id]['dat:train_triples_neg'] = len(train_triples[train_triples['label'] == 0]) + len(valid_triples[valid_triples['label'] == 0])
            self.model_info[rel_id]['dat:train_triples_pos'] = len(train_triples[train_triples['label'] == 1]) + len(valid_triples[valid_triples['label'] == 1])
            self.model_info[rel_id]['dat:train_triples_total'] = len(train_triples) + len(valid_triples)
            
            self.model_info[rel_id]['dat:test_triples_neg'] = len(test_triples[test_triples['label'] == 0])
            self.model_info[rel_id]['dat:test_triples_pos'] = len(test_triples[test_triples['label'] == 1])
            self.model_info[rel_id]['dat:test_triples_total'] = len(test_triples)

            #Loading subsets:
            train_features = pd.read_pickle(rel_folder + 'train_df_feature_set.pkl', compression='gzip')
            valid_features = pd.read_pickle(rel_folder + 'valid_df_feature_set.pkl', compression='gzip')
            test_features = pd.read_pickle(rel_folder + 'test_df_feature_set.pkl', compression='gzip')

            #we append train and valid feature for metrics purposes
            train_valid_features = pd.concat([train_features, valid_features])


            train_valid_unique = set(train_valid_features.path.unique())

            self.model_info[rel_id]['sfe:train_triples_pos_w_feat'] = float(train_valid_features[train_valid_features['label'] == 1].triple_id.nunique() / (len(train_triples[train_triples['label'] == 1]) + len(valid_triples[valid_triples['label'] == 1])))
            self.model_info[rel_id]['sfe:train_triples_neg_w_feat'] = float(train_valid_features[train_valid_features['label'] == 0].triple_id.nunique() / (len(train_triples[train_triples['label'] == 0]) + len(valid_triples[valid_triples['label'] == 0])))

            test_unique = set(test_features.path.unique())

            self.model_info[rel_id]['sfe:test_triples_pos_w_feat'] = float(test_features[test_features['label'] == 1].triple_id.nunique() / len(test_triples[test_triples['label'] == 1]))
            self.model_info[rel_id]['sfe:test_triples_neg_w_feat'] = float(test_features[test_features['label'] == 0].triple_id.nunique() / len(test_triples[test_triples['label'] == 0]))
            
            self.model_info[rel_id]['sfe:train_triples_w_feat'] = float(train_valid_features.triple_id.nunique() / (len(train_triples) + len(valid_triples)))
            self.model_info[rel_id]['sfe:test_triples_w_feat'] = float(test_features.triple_id.nunique() / len(test_triples))

            self.model_info[rel_id]['sfe:train_feature_count'] = len(train_valid_unique)
            self.model_info[rel_id]['sfe:test_feature_count'] = len(test_unique)
            self.model_info[rel_id]['sfe:common_feature_count'] = len(train_valid_unique & test_unique)
            self.model_info[rel_id]['sfe:features_triples_ratio'] = float(len(train_valid_unique) / (len(train_triples) + len(valid_triples)))

            combined_features = pd.concat([train_valid_features, test_features], ignore_index = True)
            combined_features['p_len'] = combined_features['path'].apply(lambda x: len(x.split('_')))

            print('Saving combined_features.fset... ', end = ' ')
            combined_features.to_pickle(rel_folder + 'combined_features.fset')
            print('Done!')

            end_time = time.time()

            previous_time = self.model_info[rel_id].get('sfe:feature_compilation2_elapsed_time', 0)
            self.model_info[rel_id]['sfe:feature_compilation2_elapsed_time'] = previous_time + end_time - start_time

            print('\nSaving features to file: {}combined_features.fset'.format(rel_folder))
            print('\nSFE extracted {} unique features for relation {}'.format(combined_features['path'].nunique(), rel))

            return train_features

    def fast_compile_subsets(self, rel_id):

            start_time = time.time()

            rel = self.rel_dict[rel_id]

            print('\n\nFetching features for {} relation:'.format(rel))
            rel_folder = self.output_dir + rel + '/'
            train_triples = pd.read_csv(self.splits_folder + rel + '/train.tsv', sep='\t')
            test_triples = pd.read_csv(self.splits_folder + rel + '/test.tsv', sep='\t')
            valid_triples = pd.read_csv(self.splits_folder + rel + '/valid.tsv', sep='\t')

            #Dataset Statistics
            self.model_info[rel_id]['dat:train_triples_neg'] = len(train_triples[train_triples['label'] == 0]) + len(valid_triples[valid_triples['label'] == 0])
            self.model_info[rel_id]['dat:train_triples_pos'] = len(train_triples[train_triples['label'] == 1]) + len(valid_triples[valid_triples['label'] == 1])
            self.model_info[rel_id]['dat:train_triples_total'] = len(train_triples) + len(valid_triples)
            
            self.model_info[rel_id]['dat:test_triples_neg'] = len(test_triples[test_triples['label'] == 0])
            self.model_info[rel_id]['dat:test_triples_pos'] = len(test_triples[test_triples['label'] == 1])
            self.model_info[rel_id]['dat:test_triples_total'] = len(test_triples)

            #Loading subsets:
            train_features = self.load_from_pkl(rel_folder + 'train_feature_package')
            valid_features = self.load_from_pkl(rel_folder + 'valid_feature_package')
            test_features = self.load_from_pkl(rel_folder + 'test_feature_package')


            #SFE statistics
            train_mask_pos = train_triples.label.values == 1
            train_mask_neg = train_triples.label.values == 0
            valid_mask_pos = valid_triples.label.values == 1
            valid_mask_neg = valid_triples.label.values == 0
            test_mask_pos = test_triples.label.values == 1
            test_mask_neg = test_triples.label.values == 0

            self.model_info[rel_id]['sfe:train_triples_neg_w_feat'] = (np.count_nonzero(train_features['feature_matrix'][train_mask_neg].sum(axis=1) ) + np.count_nonzero(valid_features['feature_matrix'][valid_mask_neg].sum(axis=1) )) / (len(train_triples[train_triples['label'] == 0]) + len(valid_triples[valid_triples['label'] == 0]))
            self.model_info[rel_id]['sfe:train_triples_pos_w_feat'] = (np.count_nonzero(train_features['feature_matrix'][train_mask_pos].sum(axis=1) ) + np.count_nonzero(valid_features['feature_matrix'][valid_mask_pos].sum(axis=1) )) / (len(train_triples[train_triples['label'] == 1]) + len(valid_triples[valid_triples['label'] == 1]))
            self.model_info[rel_id]['sfe:train_triples_w_feat'] = (np.count_nonzero(train_features['feature_matrix'].sum(axis=1) ) + np.count_nonzero(valid_features['feature_matrix'].sum(axis=1) )) / (len(train_triples) + len(valid_triples))
            
            self.model_info[rel_id]['sfe:test_triples_w_feat'] = np.count_nonzero(test_features['feature_matrix'].sum(axis=1) ) / len(test_triples)
            self.model_info[rel_id]['sfe:test_triples_neg_w_feat'] = np.count_nonzero(test_features['feature_matrix'][test_mask_neg].sum(axis=1) ) / len(test_triples[test_triples['label'] == 0])
            self.model_info[rel_id]['sfe:test_triples_pos_w_feat'] = np.count_nonzero(test_features['feature_matrix'][test_mask_pos].sum(axis=1) ) / len(test_triples[test_triples['label'] == 1])

            self.model_info[rel_id]['sfe:train_feature_count'] = len(set(train_features['vectorizer'].feature_names_ + valid_features['vectorizer'].feature_names_))
            self.model_info[rel_id]['sfe:test_feature_count'] = len(set(test_features['vectorizer'].feature_names_))
            self.model_info[rel_id]['sfe:common_feature_count'] = len(set(train_features['vectorizer'].feature_names_ + valid_features['vectorizer'].feature_names_) & set(test_features['vectorizer'].feature_names_))
            self.model_info[rel_id]['sfe:features_triples_ratio'] = len(set(train_features['vectorizer'].feature_names_ + valid_features['vectorizer'].feature_names_)) / (len(train_triples) + len(valid_triples))

            end_time = time.time()

            previous_time = self.model_info[rel_id].get('sfe:feature_compilation2_elapsed_time', 0)
            self.model_info[rel_id]['sfe:feature_compilation2_elapsed_time'] = previous_time + end_time - start_time

            print('\nSFE extracted {} unique features for relation {}'.format(len(set(train_features['vectorizer'].feature_names_ + valid_features['vectorizer'].feature_names_ + test_features['vectorizer'].feature_names_)), rel))

            return 

    def compile_dataset(self, rel_id):

            start_time = time.time()

            rel = self.rel_dict[rel_id]

            print('\n\nFetching features for {} relation:'.format(rel))
            rel_folder = self.output_dir + rel + '/'
            train_triples = pd.read_csv(self.splits_folder + rel + '/train.tsv', sep='\t')
            test_triples = pd.read_csv(self.splits_folder + rel + '/test.tsv', sep='\t')
            valid_triples = pd.read_csv(self.splits_folder + rel + '/valid.tsv', sep='\t')

            #Dataset Statistics
            self.model_info[rel_id]['dat:train_triples_neg'] = len(train_triples[train_triples['label'] == 0]) + len(valid_triples[valid_triples['label'] == 0])
            self.model_info[rel_id]['dat:train_triples_pos'] = len(train_triples[train_triples['label'] == 1]) + len(valid_triples[valid_triples['label'] == 1])
            self.model_info[rel_id]['dat:train_triples_total'] = len(train_triples) + len(valid_triples)
            
            self.model_info[rel_id]['dat:test_triples_neg'] = len(test_triples[test_triples['label'] == 0])
            self.model_info[rel_id]['dat:test_triples_pos'] = len(test_triples[test_triples['label'] == 1])
            self.model_info[rel_id]['dat:test_triples_total'] = len(test_triples)

            train_dict = dict(zip(train_triples.triple_id, train_triples.label))
            test_dict = dict(zip(test_triples.triple_id, test_triples.label))
            valid_dict = dict(zip(valid_triples.triple_id, valid_triples.label))

            print('\nTrain triples:')
            time.sleep(0.2)
            train_features = pd.DataFrame(columns=['triple_id', 'path', 'min_node_sim', 'max_node_sim', 'min_avg_node_sim',
             'max_avg_node_sim', 'avg_rel_sim', 'min_rel_sim', 'h_t_sim', 'node_relv_in', 'node_relv_out', 'label', 'subset'])
            triples_to_load = train_triples['triple_id'].values
            counter_1_neg = 0
            counter_1_pos = 0
            for triple in tqdm(triples_to_load):
                filenm = triple + '.fset'
                df = pd.read_pickle(rel_folder + filenm, compression='gzip')
                if len(df) == 0: continue
                df = self.group_features(df, triple)
                if train_dict[triple] == 1:
                    counter_1_pos += 1
                else:
                    counter_1_neg += 1
                train_features = train_features.append(df, ignore_index = True)
            train_features['label'] = train_features['triple_id'].map(train_dict)
            train_features['subset'] = 'train'

            print('\nValid triples:')
            time.sleep(0.2)
            triples_to_load = valid_triples['triple_id'].values
            valid_features = pd.DataFrame(columns=['triple_id', 'path', 'min_node_sim', 'max_node_sim', 'min_avg_node_sim',
             'max_avg_node_sim', 'avg_rel_sim', 'min_rel_sim','h_t_sim', 'node_relv_in', 'node_relv_out', 'label', 'subset'])
            for triple in tqdm(triples_to_load):
                filenm = triple + '.fset'
                df = pd.read_pickle(rel_folder + filenm, compression='gzip')
                if len(df) == 0: continue
                df = self.group_features(df, triple)
                if valid_dict[triple] == 1:
                    counter_1_pos += 1
                else:
                    counter_1_neg += 1
                valid_features = valid_features.append(df, ignore_index = True)
            if len(valid_features) != 0:
                valid_features['label'] = valid_features['triple_id'].map(valid_dict)
                valid_features['subset'] = 'valid'
            else:
                print('\nFound no feature for Valid Set!')

            train_valid_unique = set(list(train_features.path.unique()) + list(valid_features.path.unique()))

            self.model_info[rel_id]['sfe:train_triples_pos_w_feat'] = counter_1_pos / (len(train_triples[train_triples['label'] == 1]) + len(valid_triples[valid_triples['label'] == 1]))
            self.model_info[rel_id]['sfe:train_triples_neg_w_feat'] = counter_1_neg / (len(train_triples[train_triples['label'] == 0]) + len(valid_triples[valid_triples['label'] == 0]))
            
            print('\nTest triples:')
            time.sleep(0.2)
            triples_to_load = test_triples['triple_id'].values
            test_features = pd.DataFrame(columns=['triple_id', 'path', 'min_node_sim', 'max_node_sim', 'min_avg_node_sim',
             'max_avg_node_sim', 'avg_rel_sim', 'min_rel_sim', 'h_t_sim', 'node_relv_in', 'node_relv_out', 'label', 'subset'])
            counter_2_neg = 0
            counter_2_pos = 0
            for triple in tqdm(triples_to_load):
                filenm = triple + '.fset'
                df = pd.read_pickle(rel_folder + filenm, compression='gzip')
                if len(df) == 0: continue
                df = self.group_features(df, triple)
                if test_dict[triple] == 1:
                    counter_2_pos += 1
                else:
                    counter_2_neg += 1
                test_features = test_features.append(df, ignore_index = True)
            test_features['label'] = test_features['triple_id'].map(test_dict)
            test_features['subset'] = 'test'

            test_unique = set(list(test_features.path.unique()))

            self.model_info[rel_id]['sfe:test_triples_pos_w_feat'] = counter_2_pos / len(test_triples[test_triples['label'] == 1])
            self.model_info[rel_id]['sfe:test_triples_neg_w_feat'] = counter_2_neg / len(test_triples[test_triples['label'] == 0])
            
            self.model_info[rel_id]['sfe:train_triples_w_feat'] = (counter_1_pos + counter_1_neg) / (len(train_triples) + len(valid_triples))
            self.model_info[rel_id]['sfe:test_triples_w_feat'] = (counter_2_pos + counter_2_neg) / len(test_triples)

            self.model_info[rel_id]['sfe:train_feature_count'] = len(train_valid_unique)
            self.model_info[rel_id]['sfe:test_feature_count'] = len(test_unique)
            self.model_info[rel_id]['sfe:common_feature_count'] = len(train_valid_unique & test_unique)
            self.model_info[rel_id]['sfe:features_triples_ratio'] = len(train_valid_unique) / (len(train_triples) + len(valid_triples))

            combined_features = pd.concat([train_features, test_features, valid_features], ignore_index = True)
            combined_features['p_len'] = combined_features['path'].apply(lambda x: len(x.split('_')))

            print('Deleting temporary files... '.format(self.dataset), end = ' ')
            files_list = os.listdir(rel_folder)
            for f in files_list:
                os.remove(rel_folder + f)
            print('Done!')

            print('Saving combined_features.fset... ', end = ' ')
            combined_features.to_pickle(rel_folder + 'combined_features.fset')
            print('Done!')

            end_time = time.time()

            self.model_info[rel_id]['sfe:elapsed_time'] += end_time - start_time

            print('\nSaving features to file: {}combined_features.fset'.format(rel_folder))
            print('\nSFE extracted {} unique features for relation {}'.format(combined_features['path'].nunique(), rel))


############################################################################
# Main flow control functions

    def run_counter(self, cases):

        for _ in tqdm(cases):
            _ = self.counter.get(True)

        return

    def merge_subg_inf(self, process, return_dict, rel_id):
        
        #Defining columns of an empty dataframe in case no feature is found for a given triple
        cols = ['triple_id', 'path', 'min_node_sim', 'max_node_sim', 'min_avg_node_sim', 
        'max_avg_node_sim', 'h_t_sim', 'node_relv_in', 'node_relv_out']

        while True:
            
            try:
                triple = self.corrupted_queue.get(True, timeout = 2)
            except:
                return return_dict
            
            features = self.sim_merge_subgraphs(triple, process)
            if len(features) != 0:
                return_dict[triple] = self.group_features(features, triple)
            else:
                return_dict[triple] = pd.DataFrame(columns=cols)

            self.counter.put(process, True, timeout = 10)
        
        return return_dict

    def fast_merge_subg_inf(self, process, return_dict, rel_id):
        
        while True:
            
            try:
                triple = self.corrupted_queue.get(True, timeout = 2)
            except:
                return return_dict
            
            return_dict[triple] = self.fast_merge_subgraphs(triple, process)

            self.counter.put(process, True, timeout = 10)
        
        return return_dict

    def extract_features(self, rel_id, subset):

        t1 = time.time()

        manager = mp.Manager()
        return_dict = manager.dict()
        features = dict()

        if self.corrupted_queue.qsize() != 0:
            print('The queue has garbage!')
            return False

        self.build_queue(rel_id, subset)

        cases = [1] * self.corrupted_queue.qsize() #just a dummy list with len = training examples to help with tqdm

        if self.corrupted_queue.qsize() == 0:
            print('No training examples in the queue, skipping subset.')
            return features, return_dict

        processes = []


        p = mp.Process(target = self.run_counter, args=(cases,))
        self.logger.info('Starting Counter Process')
        processes.append(p)

        for i in range(mp.cpu_count()):
            p = mp.Process(target = self.merge_subg_inf, args=(i, return_dict, rel_id))
            self.logger.info('Starting process #%s', i)
            processes.append(p)

        print('\nExtracting features for {} subset.'.format(subset))

        [x.start() for x in processes]
        [x.join() for x in processes]

        t2 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_extraction_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_extraction_elapsed_time'] = previous_time + t2 - t1

        t3 = time.time()
        self.model_info[rel_id]['sfe:feature_vectorization_elapsed_time'] = 0

        rel_name = self.names_dict[rel_id]
        rel_dir = self.output_dir + rel_name + '/'

        print('\nCompiling feature dataframe for {} subset...'.format(subset), end=' ')
        time.sleep(0.2)

        feature_dataframe = pd.concat(list(return_dict.values()))
        print('Done!')

        feature_dataframe['subset'] = subset
        feature_dataframe['p_len'] = feature_dataframe['path'].apply(lambda x: len(x.split('_')))
        feature_dataframe['label'] = feature_dataframe['triple_id'].apply(lambda x: int(x.split('_')[2]))
        
        t4 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_dataframe_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_dataframe_elapsed_time'] = previous_time + t4 - t3

        print('Computing rel similarities...', end=' ')

        def compute_avg_rel_similarities(path, rel_id):

            path = path.split('_')
            sims = []
            for edge in path:
                sims.append(self.rel_similarity_matrix[int(rel_id[1:]), int(edge.replace('i', '')[1:])])
            
            return np.mean(np.array(sims))

        def compute_min_rel_similarities(path, rel_id):

            path = path.split('_')
            sims = []
            for edge in path:
                sims.append(self.rel_similarity_matrix[int(rel_id[1:]), int(edge.replace('i', '')[1:])])
            
            return min(np.array(sims))

        feature_dataframe['avg_rel_sim'] = feature_dataframe['path'].apply(lambda x: compute_avg_rel_similarities(x, rel_id))
        feature_dataframe['min_rel_sim'] = feature_dataframe['path'].apply(lambda x: compute_min_rel_similarities(x, rel_id))
        print('Done!')

        t5 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_sim_calculation_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_sim_calculation_elapsed_time'] = previous_time + t5 - t4
                
        feature_dataframe.to_pickle(rel_dir + '{}_df_feature_set.pkl'.format(subset), compression='gzip')
        self.write_to_pkl(rel_dir + '{}_feature_set'.format(subset), dict(return_dict))

        t6 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_storing_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_storing_elapsed_time'] = previous_time + t6 - t5

    def fast_extract_features(self, rel_id, subset):

        t1 = time.time()

        rel_name = self.names_dict[rel_id]
        rel_dir = self.output_dir + rel_name + '/'

        manager = mp.Manager()
        return_dict = manager.dict()
        features = dict()

        if self.corrupted_queue.qsize() != 0:
            print('The queue has garbage!')
            return False

        self.build_queue(rel_id, subset)

        cases = [1] * self.corrupted_queue.qsize() #just a dummy list with len = training examples to help with tqdm

        if self.corrupted_queue.qsize() == 0:
            print('No training examples in the queue, skipping subset.')
            return features, return_dict

        processes = []


        p = mp.Process(target = self.run_counter, args=(cases,))
        self.logger.info('Starting Counter Process')
        processes.append(p)

        for i in range(mp.cpu_count()):
            p = mp.Process(target = self.fast_merge_subg_inf, args=(i, return_dict, rel_id))
            self.logger.info('Starting process #%s', i)
            processes.append(p)

        print('\nExtracting features for {} subset.'.format(subset))

        [x.start() for x in processes]
        [x.join() for x in processes]

        t2 = time.time()

        previous_time = self.model_info[rel_id].get('sfe:feature_extraction_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_extraction_elapsed_time'] = previous_time + t2 - t1

        print('\nCompiling feature dataframe for {} subset...'.format(subset), end=' ')
        time.sleep(0.2)

        features = list(return_dict.values())
        rows = list(return_dict.keys())

        v = DictVectorizer(sparse=True)
        v.fit(features)
        feature_matrix = v.transform(features)
        print('Done!')

        feature_package = {
            'vectorizer': v,
            'row_names': rows,
            'feature_matrix': feature_matrix,
            'feature_extractor': dict(return_dict)
        }

        t3 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_vectorization_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_vectorization_elapsed_time'] = previous_time + t3 - t2

        # #Now we are going to perform the original way of storing to pandas dataframes
        # feature_dataframe = pd.DataFrame.from_dict(dict(return_dict), orient='columns').reset_index()
        # feature_dataframe = pd.melt(feature_dataframe, id_vars='index', var_name='triple_id').rename(columns={'index':'path'}).dropna()

        # feature_dataframe['subset'] = subset
        # feature_dataframe['p_len'] = feature_dataframe['path'].apply(lambda x: len(x.split('_')))
        # feature_dataframe['label'] = feature_dataframe['triple_id'].apply(lambda x: int(x.split('_')[2]))

        t4 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_dataframe_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_dataframe_elapsed_time'] = previous_time + t4 - t3

        # print('Computing rel similarities...', end=' ')

        # def compute_avg_rel_similarities(path, rel_id):

        #     path = path.split('_')
        #     sims = []
        #     for edge in path:
        #         sims.append(self.rel_similarity_matrix[int(rel_id[1:]), int(edge.replace('i', '')[1:])])
            
        #     return np.mean(np.array(sims))

        # def compute_min_rel_similarities(path, rel_id):

        #     path = path.split('_')
        #     sims = []
        #     for edge in path:
        #         sims.append(self.rel_similarity_matrix[int(rel_id[1:]), int(edge.replace('i', '')[1:])])
            
        #     return min(np.array(sims))

        # feature_dataframe['avg_rel_sim'] = feature_dataframe['path'].apply(lambda x: compute_avg_rel_similarities(x, rel_id))
        # feature_dataframe['min_rel_sim'] = feature_dataframe['path'].apply(lambda x: compute_min_rel_similarities(x, rel_id))
        # print('Done!')

        t5 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_sim_calculation_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_sim_calculation_elapsed_time'] = previous_time + t5 - t4

        print('Storing feature dataframe...', end=' ')
        # feature_dataframe.to_pickle(rel_dir + '{}_df_feature_set.pkl'.format(subset), compression='gzip')
        print('Done!')

        print('Storing feature package...', end=' ')
        self.write_to_pkl(rel_dir + '{}_feature_package'.format(subset), feature_package)
        print('Done!')

        t6 = time.time()
        previous_time = self.model_info[rel_id].get('sfe:feature_storing_elapsed_time', 0)
        self.model_info[rel_id]['sfe:feature_storing_elapsed_time'] = previous_time + t6 - t5

    def run_single_relation(self, relation, method):
      
        start_time = time.time()

        relation_name = self.rel_dict[relation]

        check = self.load_rel_splits(relation, self.splits_folder)
  
        if not check:
            print('Skipping {} due to lack of one of the files.\n'.format(relation_name))
            self.logger.info('Skipping relation %s due to lack of training/test/valid files\n',relation_name)
            return

        self.model_info[relation] = dict()

        sub_dir = os.path.join(self.output_dir, relation_name)
        subsets = ['train', 'test', 'valid']
        self.logger.info('Starting Feature Extraction for %s.',relation_name)
        print('\nStarting feature extraction for relation {}'.format(relation_name))

        if not os.path.exists(sub_dir):
            os.makedirs(os.path.join(sub_dir))
            print('Creating folder: {}.'.format(sub_dir))
        
        if method == 'fast': feature_extractor = self.fast_extract_features
        if method == 'book_keeping': feature_extractor = self.extract_features

        for subset in subsets:
            feature_extractor(relation, subset)

        end_time = time.time()

        if method == 'book_keeping': self.compile_subsets(relation)
        if method == 'fast': self.fast_compile_subsets(relation)

        # self.model_info[relation]['emb:model'] = self.emb_model_info.get('model', '')
        # self.model_info[relation]['emb:timestamp'] = self.emb_model_info.get('timestamp', '')
        # self.model_info[relation]['emb:overall_test_acc'] = self.emb_model_info.get('acc', '')

        self.model_info[relation]['sfe:timestamp'] = self.time_stamp
        self.model_info[relation]['dat:dataset'] = self.bench_dataset
        self.model_info[relation]['dat:total_rels'] = int(len(self.rel_dict.keys()) / 2)
        self.model_info[relation]['dat:tested_rels'] = 1
        self.model_info[relation]['sfe:extraction_method'] = method
        self.model_info[relation]['sfe:sim_model'] = self.sm_model_name
        self.model_info[relation]['sfe:node_threshold'] = self.thresholds.get(relation, {}).get('node_threshold', 'nan')
        self.model_info[relation]['sfe:rel_threshold'] = self.thresholds.get(relation, {}).get('rel_threshold', 'nan')
        self.model_info[relation]['sfe:elapsed_time'] = end_time - start_time

        self.model_info[relation]['sfe:node_threshold'] = self.params_dict.get('sfe:node_threshold', 'no')
        self.model_info[relation]['sfe:rel_threshold'] = self.params_dict.get('sfe:rel_threshold', 'no')
        self.model_info[relation]['sfe:node_relv_in'] = self.params_dict.get('sfe:node_relv_in', 'no')
        self.model_info[relation]['sfe:top_rels_reduced_graph'] = self.params_dict.get('sfe:top_rels_reduced_graph', 'no')

        self.logger.info('Finished Feature Extraction for %s.\n',relation_name)

    def run_single_file(self, file_folder, file_name, top_n = False):
        '''This method gets a file with a set of triples and extracts sfe features, it is important
        that the file contains ONLY triples of the same relation to avoid confusion. It ouputs
        the features in the pickle format (the same as run_single_relation). '''

        #load file
        triple_set = pd.read_csv(os.path.join(file_folder, file_name), sep='\t')
        
        if top_n:
            triple_set = triple_set.head(top_n)
        
        triple_set['label'] = 0
                
        if triple_set['rel'].nunique() != 1:
            raise ValueError('File contais triples with more than one relation.')

        relation = triple_set['rel'][0]
        relation_name = self.rel_dict[relation]
        # print(relation_name)

        #Find the ouput dir and get all triples already processed
        rel_dir = self.output_dir + relation_name + '/'
        # existing_triples = set([f.split('.')[0] for f in os.listdir(rel_dir)])
        triples = set(triple_set['triple_id'])
        if not self.override_pkl:
            triples = triples - self.processed_triples

        if len(triples) > 0:
           
            # print('Processing {} triples'.format(len(triples_to_process)))

            entities_to_expand = []
            triples_to_process = []
            for triple in triples:
                head = triple.split('_')[0]
                tail = triple.split('_')[1]
                rel = triple.split('_')[2]
                triples_to_process.append((head, tail, 0, rel))
                entities_to_expand.append(head)
                entities_to_expand.append(tail)

            dict_to_process = dict()
            dict_to_process[relation] = triples_to_process
            self.corrupted_dict = {'ad_hoc': dict_to_process }

            #Build a list of subgraphs to expand from the triples to process list
            start = time.time()
            
            entities_to_expand = set(entities_to_expand)
            entities_to_expand = entities_to_expand - self.subgraphs.keys()

            if len(entities_to_expand) > 0 and self.load_pre_built_subgraphs:
                self.batch_load_subgraphs(entities=entities_to_expand)

            self.extract_features(relation, 'ad_hoc')

            self.processed_triples = self.processed_triples | triples

        else:
            pass
            # print('No triple to process, skiping file!')

    def run(self, relations_to_run = [], method='fast', override_features = False):

        start = time.time()

        if relations_to_run == []:
            relations_to_run = [rel for rel in self.rel_dict.keys() if rel[0] != 'i']

        try:
            existing_metrics = pd.read_csv(self.output_dir + 'sfe_model_info.tsv', sep='\t', index_col=0)
            self.model_info = dict(existing_metrics)
            print('Using existing SFE timestamp.')
            if not override_features:
                relations_to_run = [rel for rel in relations_to_run if not rel in existing_metrics.columns]
        except:
            pass

        if self.params_dict.get('sfe:node_relv_in', False):
            self.subgraph_engine = self.context_aware_subgraph
            print('Using context_aware_subgraph to build subgraphs.')
        else:
            self.subgraph_engine = self.semi_naive_subgraph
            print('Using unconstrained subgraph build')

        self.logger.info('Starting Feature Extraction %s Dataset\n', self.bench_dataset)

        counter = 1
        for relation in relations_to_run:

            if self.params_dict.get('sfe:top_rels_reduced_graph', False):
                selected_rels = self.select_top_similar_rels(relation, self.params_dict['sfe:top_rels_reduced_graph'])
                print('Building Reduced Graph for {}'.format(self.names_dict[relation]))
                self.build_reduced_graph(selected_rels)
                
            print('\nProcessing relation {}/{}.'.format(counter, len(relations_to_run)))
            self.run_single_relation(relation, method)

            counter += 1
            os.system('spd-say "finished {}"'.format(relation))

            #If a metrics template is found then it organizes model_info tsv file
            try:
                metrics_template = pd.read_csv('metrics_template.tsv', sep='\t', index_col=0)
                metrics = pd.merge(left=metrics_template, right=pd.DataFrame(self.model_info), how='left', left_index=True, right_index=True)
                metrics.sort_values(by='idx', ascending=True, inplace=True)
                metrics.drop(columns=['idx', 'metric_type'], inplace=True)
                metrics = metrics.head(49) #filtering out only sfe and emb metrics from template
                metrics.to_csv(self.output_dir + 'sfe_model_info.tsv', sep='\t')
            except:
                pd.DataFrame(self.model_info).to_csv(self.output_dir + 'sfe_model_info.tsv', sep='\t')

        end = time.time()
        print('\nFinished in {} seconds.'.format(end-start))
        self.logger.info('Finished Pipeline in %s minutes.', (end-start)/60)
