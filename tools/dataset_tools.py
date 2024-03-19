import os, re
import pandas as pd
import numpy as np
import itertools

from tools.tools import get_dirs
from tqdm import tqdm as tqdm
from collections import defaultdict, deque, Counter

from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, euclidean_distances, pairwise_distances
from sklearn.feature_extraction import DictVectorizer

import gensim


class Dataset(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.bench_path = os.path.expanduser('~') + f'/proj/OpenKE/benchmarks/{self.dataset}/'
        self.ent = self.load_entities()
        self.rel = self.load_relations()
        

        self.ent_dict = dict(zip(self.ent.id, self.ent.ent))
        self.rel_dict = dict(zip(self.rel.id, self.rel.rel))
        self.names_dict = {**self.ent_dict, **self.rel_dict}
        self.ent_dict_rev = dict(zip(self.ent.ent, self.ent.id))
        self.rel_dict_rev = dict(zip(self.rel.rel, self.rel.id))

        self.train_set = None
        self.test_set = None
        self.valid_set = None
        self.complete_set = None
        self.raw_graph = None
        self.graph = None
        self.adj_list = None
        self.corrupted_train_set = None
        self.corrupted_valid_set = None
        self.corrupted_test_set = None
        self.corrupted_total_set = None
        self.ent_context_matrix = None
        self.rel_context_matrix = None
        self.sm_model = None
        self.sm_model_name = None
        self.ent_similarity_matrix = None
        self.rel_similarity_matrix = None


        print('Loaded {} Dataset with {} entities and {} relations.\n'.format(dataset, len(self.ent), len(self.rel)))
    
    def compute_similarity_matrix(self):

        rels = self.rel_dict.keys()
        rels = [rel for rel in rels if rel[0] != 'i']
        rel_vectors = np.zeros((len(rels), self.sm_model.vector_size))
        for rel in rels:
            rel_vectors[int(rel[1:])] = self.sm_model.get_vector(rel)
        self.rel_similarity_matrix = cosine_similarity(rel_vectors, rel_vectors)

        entities = self.ent_dict.keys()
        ent_vectors = np.zeros((len(entities), self.sm_model.vector_size))
        for ent in entities:
            ent_vectors[int(ent[1:])] = self.sm_model.get_vector(ent)
        self.ent_similarity_matrix = cosine_similarity(ent_vectors, ent_vectors)

        print('Computed rel and ent similarity matrices.')        

    def load_kv_model(self, model):

        self.sm_model = gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.load(self.sm_folder + model)
        self.sm_model_name = model

        print('Loaded Keyed-Vectors Similarity Model.')

        self.compute_similarity_matrix()

    def load_w2v_model(self, model):

        self.sm_model = gensim.models.Word2Vec.load(model)
        self.sm_model_name = model

        print('Loaded Word2Vec Model.')

        self.compute_similarity_matrix()

    def build_rel_vocab(self):

        rels = self.rel

        rel_description = [rel for rel in rels.rel.values if rel[:2] != 'i*']

        words = []

        for rel in rel_description:

            words += (re.split("\s+", re.sub('[-._]', ' ', rel), flags=re.UNICODE))

        words = list(set(words))

        return words

    def load_entities(self):

        ent = pd.read_csv(os.path.join(self.bench_path, 'entity2id.txt'), 
                            sep='\t', skiprows=1, header=None, 
                            names=['ent', 'id'])

        if (self.dataset == 'FB15K') or (self.dataset == 'FB15K237'):
            mids_index = pd.read_csv(os.path.join(self.bench_path, 'real_name_index.csv'), 
                                sep=',', header=None,
                                names = ['mid', 'ent_description'])
            ent = pd.merge(ent, mids_index, how='left', left_on='ent', right_on='mid')
            ent = ent[['id', 'ent_description']].rename(columns={'ent_description': 'ent'})

        ent['id'] = 'e' + ent['id'].astype(str)

        return ent[['id', 'ent']]
        
    def load_relations(self):

        rel = pd.read_csv(os.path.join(self.bench_path, 'relation2id.txt'), 
                            sep='\t', skiprows=1, header=None, 
                            names=['rel', 'id']) 

        rel['id'] = 'r'+ rel['id'].astype(str)

        if (self.dataset == 'FB15K') or (self.dataset == 'FB15K237'):
            rel['rel'] = rel['rel'].apply(self.trim_relation_name)

        inv_rel = pd.DataFrame()
        inv_rel['id'] = 'i' + rel['id']
        inv_rel['rel'] = 'i*' + rel['rel']

        rel = rel.append(inv_rel, ignore_index = True)

        return rel[['id', 'rel']]

    def invert_rel(self, rel):

        rel = str(rel)

        if rel[0] == 'i':
            return rel[1:]
        else:
            return 'i' + rel

    def trim_relation_name(self, relation_name):
        '''just a simple function to remove forward slashes from relation names
        it trims the first forward slash and replaces the others
        '''

        if relation_name[0] == '/':
            relation_name = relation_name[1:]
        
        return relation_name.replace('/', '-')

    def load_true_sets(self):
        '''Method to load datasets with true facts
        '''

        print('Loading {} true facts... '.format(self.dataset), end = ' ')

        train_set = pd.read_csv(self.bench_path + 'train2id.txt', sep=' ',
                        skiprows=1,
                        header=None, 
                        names=['e1', 'e2', 'rel'])

        raw_train_set = train_set.copy()

        train_set['e1'] = 'e' + train_set['e1'].astype(str)
        train_set['e2'] = 'e' + train_set['e2'].astype(str)
        train_set['rel'] = 'r' + train_set['rel'].astype(str)
        self.train_set = train_set

        test_set = pd.read_csv(self.bench_path + 'test2id.txt', sep=' ',
                        skiprows=1,
                        header=None, 
                        names=['e1', 'e2', 'rel'])

        

        test_set['e1'] = 'e' + test_set['e1'].astype(str)
        test_set['e2'] = 'e' + test_set['e2'].astype(str)
        test_set['rel'] = 'r' + test_set['rel'].astype(str)
        self.test_set = test_set
        
        valid_set = pd.read_csv(self.bench_path + 'valid2id.txt', sep=' ',
                        skiprows=1,
                        header=None, 
                        names=['e1', 'e2', 'rel'])

        raw_valid_set = valid_set.copy()

        valid_set['e1'] = 'e' + valid_set['e1'].astype(str)
        valid_set['e2'] = 'e' + valid_set['e2'].astype(str)
        valid_set['rel'] = 'r' + valid_set['rel'].astype(str)
        self.valid_set = valid_set
        
        self.raw_graph = pd.concat([raw_train_set, raw_valid_set])

        print('Done!\n')

        
        print('Train set has {} triples'.format(train_set.shape[0]))
        print('Test set has {} triples'.format(test_set.shape[0]))
        print('Valid set has {} triples'.format(valid_set.shape[0]))

    def build_complete_true_set(self):

        if not isinstance(self.train_set, pd.DataFrame):
            self.load_true_sets()

        train = self.train_set
        train['subset'] = 'train'
        test = self.test_set
        test['subset'] = 'test'
        valid = self.valid_set
        valid['subset'] = 'valid'

        self.complete_set = pd.concat([train, test, valid], ignore_index = True)

    def build_type_constraints(self):
        '''
        This method takes as input the complete_set (from the build_complete_true_set method, actually) file and  builds a file with type constraints, 
        getting all three subsets and returning a dict where:

        key level #1 -> relation
        key level #2 -> head / tail
        values -> entities appearing in head or tail of the given relation

        '''
        raw = lambda x: int(x[2:]) if x[0] == 'i' else int(x[1:])

        if not isinstance(self.complete_set, pd.DataFrame):
            self.build_complete_true_set()

        df = self.complete_set

        rels = df.rel.unique().tolist()

        type_constraints = dict()

        for rel in rels:

            type_constraints[rel] = dict()

            this_rel = df[df['rel'] == rel]

            heads = this_rel.e1.unique().tolist()
            tails = this_rel.e2.unique().tolist()

            type_constraints[rel]['head'] = heads
            type_constraints[rel]['head_int'] = [raw(e) for e in heads]
            type_constraints[rel]['tail'] = tails
            type_constraints[rel]['tail_int'] = [raw(e) for e in tails]


        return type_constraints

    def build_graph(self):
        '''This function loads train and valid true sets and builds a graph
        of the form:
        Key level #1: starting node
        Key level #2: outgoing relations
        Values: connected nodes for each relation

        Since the graph can be traversed regardless of the direction of the
        relation, we append the inverse relation as outgoing relation.
        '''

        print('\nBuilding {} graph...'.format(self.dataset), end = ' ')

        if not isinstance(self.train_set, pd.DataFrame):
            print('loading triples...', end = ' ')
            self.load_true_sets()   

        triples_df = self.train_set.append(self.valid_set, ignore_index = True)

        #Creating inverse triples to find reverse edges
        inversed_triples = pd.DataFrame()
        inversed_triples['e1'] = triples_df['e2']
        inversed_triples['e2'] = triples_df['e1']
        inversed_triples['rel'] = "i" + triples_df['rel'] #inversed relations have an i on the end

        #Appending the normal and inverse dfs
        complete_df = triples_df.append(inversed_triples, ignore_index=True, sort=False)
        complete_df['rel'] = complete_df['rel']

        e1 = list(complete_df['e1'])
        e2 = list(complete_df['e2'])
        rel = list(complete_df['rel'])

        output = defaultdict(dict)

             
        for i in range(len(e1)):

            item1 = e1[i]
            item2 = rel[i]
            item3 = e2[i]

            if output.get(item1, {}).get(item2) == None:
                output[item1][item2] = []

            output[item1][item2].append(item3)

        self.graph = output

        counter = 0
        for k1 in output.keys():
            for k2 in output[k1].keys():
                counter += len(output[k1][k2])

        print('Done!\n')
        print('Graph built with {} edges.'.format(int(counter)))

    def build_reduced_graph(self, rels):
        '''This function loads train and valid true sets and builds a graph
        of the form:
        
        Key level #1: starting node
        Key level #2: ougoing relations
        Values: connected nodes for each relation

        Since the graph can be traversed regardless of the direction of the
        relation, we append the inverse relation as outgoing relation.
        '''

        triples_df = self.train_set.append(self.valid_set, ignore_index = True)

        triples_df = triples_df[triples_df['rel'].isin(rels)]

        #Creating inverse triples to find reverse edges
        inversed_triples = pd.DataFrame()
        inversed_triples['e1'] = triples_df['e2']
        inversed_triples['e2'] = triples_df['e1']
        inversed_triples['rel'] = "i" + triples_df['rel'].astype(str) #inversed relations have an i on the end

        #Appending the normal and inverse dfs
        complete_df = triples_df.append(inversed_triples, ignore_index=True)
        complete_df['rel'] = complete_df['rel'].astype(str)

        e1 = list(complete_df['e1'])
        e2 = list(complete_df['e2'])
        rel = list(complete_df['rel'])

        output = defaultdict(dict)

             
        for i in range(len(e1)):

            item1 = e1[i]
            item2 = rel[i]
            item3 = e2[i]

            if output.get(item1, {}).get(item2) == None:
                output[item1][item2] = []

            output[item1][item2].append(item3)

        self.graph = output

        counter = 0
        for k1 in output.keys():
            for k2 in output[k1].keys():
                counter += len(output[k1][k2])

        print('Graph built with {} edges.'.format(counter / 2))

        # return output

    def build_dsp_subgraph(self, source):
        '''This function builds subgraphs from source with l up to two
        The ouput is a dict with features as keys and counts as values
        
        '''

        path_queue = deque()
        dsp_features = []

        #Initial expansion:
        for key in self.graph[source].keys():
            
            nodes = self.graph[source][key]
            for node in nodes:
                path_queue.append([key] + [node])
                dsp_features.append(key)

        
        while path_queue:

            path_to_expand = path_queue.popleft()
            node_to_open = path_to_expand[-1]

            for key in self.graph[node_to_open].keys():
                nodes = self.graph[node_to_open][key]
                for node in nodes:
                    if node == source:
                        continue
                    dsp_features.append(path_to_expand[0] + '_' + key)

        return Counter(set(dsp_features))

    def build_dsp_features(self):


        print('Building DSP Features\n')
        ents = list(self.ent.id.values)

        dsp_features = []

        for ent in tqdm(range(len(ents))):
            dsp_features.append(dict(self.build_dsp_subgraph(ent)))


        return dsp_features

    def fit_dsp_features(self):

        dsp_features = self.build_dsp_features()

        print('Fitting DictVectorizer.')

        v = DictVectorizer(sparse=True)
        dsp = v.fit_transform(dsp_features)

        return dsp

    def build_w2v_sentences(self, source):

        path_queue = deque()
        sentences = []

        #Initial expansion:
        for key in self.graph[source].keys():
            
            nodes = self.graph[source][key]
            for node in nodes:
                path = [source, key, node]
                path_queue.append(path)
                sentences.append('_'.join(path))
                

        
        while path_queue:

            path_to_expand = path_queue.popleft()
            node_to_open = path_to_expand[-1]

            for key in self.graph[node_to_open].keys():
                nodes = self.graph[node_to_open][key]
                for node in nodes:
                    if node == source:
                        continue
                    sentences.append('_'.join(path_to_expand + [key, node]))

        return set(sentences)

    def calculate_node_degrees(self):

        print('Calculating node degrees...')

        df = self.train_set.append(self.valid_set, ignore_index=True)

        return dict(Counter(list(df.e1.values) + list(df.e2.values)))
    
    def build_corpus(self):

        print('Building Corpus.')

        ents_df = self.ent
        ents_df['ent_id'] = 'e' + ents_df['id'].astype(str)
        ents = list(ents_df.ent_id.values)
        path_queue = deque()

        file1 = open('corpus.txt', 'a')

        #Initial expansion:
        for ent in tqdm(ents):
            for key in self.graph[ent].keys():
                
                nodes = self.graph[ent][key]
                for node in nodes:
                    path = [ent, key, node]
                    path_queue.append(path)
                    # file1.write('{} {} {}\n'.format(ent, key, node))
                    

            
            while path_queue:

                path_to_expand = path_queue.popleft()
                node_to_open = path_to_expand[-1]

                for key in self.graph[node_to_open].keys():
                    nodes = self.graph[node_to_open][key]
                    for node in nodes:
                        if node == ent:
                            continue
                        file1.write('{} {} {} {} {}\n'.format(path[0], path[1], path[2], key, node))

        file1.close()

    def build_adjacency_list(self):
        ''' This function builds an adjacency list for all edges of the 
        graph, regardless of the relation.

        The output is a dict where:
        Key level #1: node
        Values: connected nodes (set)
        '''

        triples_df = self.train_set.append(self.valid_set, ignore_index = True)

        #Creating inverse triples to find reverse edges
        inversed_triples = pd.DataFrame()
        inversed_triples['e1'] = triples_df['e2']
        inversed_triples['e2'] = triples_df['e1']

        #Appending the normal and inverse dfs
        complete_df = triples_df.append(inversed_triples, ignore_index=True)

        e1 = list(complete_df['e1'])
        e2 = list(complete_df['e2'])

        output = defaultdict(set)

        for i in range(len(e1)):
            current = output.get(e1[i], set())
            output[e1[i]] = current | {e2[i]}

        self.adj_list = output
        
    def load_rel_splits(self, rel, splits_path):

        rel_name = self.rel_dict[rel]

        print('Building corrupted set for relation {}:'.format(rel_name))
        
        try:
            print('Test Subset... ', end = ' ')
            test_file_to_open = splits_path + '/' + rel_name + '/test.tsv'
            test_split = pd.read_csv(test_file_to_open, sep= '\t')
            print('Done')
        except:
            print('Not found!')
            return False

        try:
            print('Valid Subset... ', end = ' ')
            valid_file_to_open = splits_path + '/' + rel_name + '/valid.tsv'
            valid_split = pd.read_csv(valid_file_to_open, sep= '\t')
            print('Done')
        except:
            print('Not found!')
            return False

        try:
            print('Train Subset... ', end = ' ')
            train_file_to_open = splits_path + '/' + rel_name + '/train.tsv'
            train_split = pd.read_csv(train_file_to_open, sep= '\t')
            print('Done')

        except:
            return False

        corrupted_dict = dict({})
        corrupted_dict[rel] = {}
        corrupted_dict[rel]['train'] = []
        corrupted_dict[rel]['test'] = []
        corrupted_dict[rel]['valid'] = []

        test_split['e1'] = 'e' + test_split['e1'].astype(str)
        test_split['e2'] = 'e' + test_split['e2'].astype(str)
        test_split['rel'] = 'r' + test_split['rel'].astype(str)
        corrupted_dict[rel]['test'] = list(zip(test_split.e1, test_split.e2, test_split.label, test_split.rel))

        train_split['e1'] = 'e' + train_split['e1'].astype(str)
        train_split['e2'] = 'e' + train_split['e2'].astype(str)
        train_split['rel'] = 'r' + train_split['rel'].astype(str)
        corrupted_dict[rel]['train'] = list(zip(train_split.e1, train_split.e2, train_split.label, train_split.rel))

        valid_split['e1'] = 'e' + valid_split['e1'].astype(str)
        valid_split['e2'] = 'e' + valid_split['e2'].astype(str)
        valid_split['rel'] = 'r' + valid_split['rel'].astype(str)
        corrupted_dict[rel]['valid'] = list(zip(valid_split.e1, valid_split.e2, valid_split.label, valid_split.rel))

        self.corrupted_dict = corrupted_dict

        return True
        
    def load_splits(self, splits_path):
        ''' Method to load splits, ie, files containing positive and 
        negative examples '''

        target_relations = get_dirs(splits_path)

        #Train Set

        corrupted_train_set = pd.DataFrame()

        print('\nLoading Train Splits...', end = ' ')

        for rel in tqdm(target_relations):
            try:
                file_to_open = splits_path + '/' + rel + '/train.tsv'
                temp_set = pd.read_csv(file_to_open, sep = '\t')
            except:
                continue

            corrupted_train_set = corrupted_train_set.append(temp_set, ignore_index=True)

        corrupted_train_set['e1'] = 'e' + corrupted_train_set['e1'].astype(str)
        corrupted_train_set['e2'] = 'e' + corrupted_train_set['e2'].astype(str)
        corrupted_train_set['rel'] = 'r' + corrupted_train_set['e1'].astype(str)

        self.corrupted_train_set = corrupted_train_set

        # print('\nFinshed Loading Train Corrupted Set with {} triples and {} columns.'.format(self.corrupted_train_set.shape[0], self.corrupted_train_set.shape[1]))

        #Test Set

        corrupted_test_set = pd.DataFrame()

        print('\nLoading Test Splits')

        for rel in target_relations:
            try:
                file_to_open = splits_path + '/' + rel + '/test.tsv'
                temp_set = pd.read_csv(file_to_open, sep = '\t')
            except:
                continue

            corrupted_test_set = corrupted_test_set.append(temp_set, ignore_index=True)

        corrupted_test_set['e1'] = 'e' + corrupted_test_set['e1'].astype(str)
        corrupted_test_set['e2'] = 'e' + corrupted_test_set['e2'].astype(str)
        corrupted_test_set['rel'] = 'r' + corrupted_test_set['e1'].astype(str)
        self.corrupted_test_set = corrupted_test_set

        print('\nFinished Loading Test Corrupted Set with {} triples and {} columns.'.format(self.corrupted_test_set.shape[0], self.corrupted_test_set.shape[1]))

        #Valid Set

        corrupted_valid_set = pd.DataFrame()

        print('\nLoading Valid Splits')

        for rel in target_relations:
            try:
                file_to_open = splits_path + '/' + rel + '/valid.tsv'
                temp_set = pd.read_csv(file_to_open, sep = '\t')
            except:
                continue

            corrupted_valid_set = corrupted_valid_set.append(temp_set, ignore_index=True)

        corrupted_valid_set['e1'] = 'e' + corrupted_valid_set['e1'].astype(str)
        corrupted_valid_set['e2'] = 'e' + corrupted_valid_set['e2'].astype(str)
        corrupted_valid_set['rel'] = 'r' + corrupted_valid_set['e1'].astype(str)
        self.corrupted_valid_set = corrupted_valid_set

        print('\nFinshed Loading Valid Corrupted Set with {} triples and {} columns.'.format(self.corrupted_valid_set.shape[0], self.corrupted_valid_set.shape[1]))
    
    def convert_df_to_dict(self, dataset):
        '''This function take as input the training, test and validation datasets
        from the corrupted files and the output is a dict:
        keys: relations
        values: a list of tuples containing e1, e2, label and relations
        '''

        relations = list(dataset['rel'].unique())
        relations.sort()

        output_dict = defaultdict(list)
        
        for rel in tqdm(relations):
            df = dataset[dataset['rel'] == rel]
            output_dict[rel] = list(zip(df.e1, df.e2, df.label, df.rel))

        return output_dict

    def build_corrupted_dicts(self):

        corrupted_dict = dict()

        corrupted_dict['train'] = self.convert_df_to_dict(self.corrupted_train_set)
        corrupted_dict['test'] = self.convert_df_to_dict(self.corrupted_test_set)
        corrupted_dict['valid'] = self.convert_df_to_dict(self.corrupted_valid_set)

        print('\nCorrupted Files:')
        print('Training set:   {} triples.'.format(len(self.corrupted_train_set)))
        print('Test set:       {} triples.'.format(len(self.corrupted_test_set)))
        print('Validation set: {} triples.'.format(len(self.corrupted_valid_set))) 

        return corrupted_dict

    def group_corrupted_subsets(self):
        ''' This method appends corrupted train, valid and test set in a single
        dataframe '''

        self.corrupted_total_set = pd.DataFrame()

        temp_df = self.corrupted_train_set
        temp_df['dataset'] = 'train'
        self.corrupted_total_set = self.corrupted_total_set.append(temp_df, ignore_index=True)

        temp_df = self.corrupted_test_set
        temp_df['dataset'] = 'test'
        self.corrupted_total_set = self.corrupted_total_set.append(temp_df, ignore_index=True)

        temp_df = self.corrupted_valid_set
        temp_df['dataset'] = 'valid'
        self.corrupted_total_set = self.corrupted_total_set.append(temp_df, ignore_index=True)

    def explain_path(self, path):
        ''' This method takes a path as input and return the relation names
        '''

        explained_path = ''

        rels = path.split('_')

        for rel in rels:

            explained_path = explained_path + ' | ' + self.names_dict[rel]

        return explained_path[3:]
    
    def compute_ent_similarities_for_rel(self, rel):

        df = self.train_set.append(self.valid_set, ignore_index = True)
        df = df[df['rel'] == rel]
        heads = df.e1.values
        tails = df.e2.values
        
        similarities = []
        for h, t in zip(heads, tails):
            similarities.append(self.sm_model.similarity(h, t))

        df['similarities'] = similarities

        return df

    def compute_context(self):
        '''
        This method computes entity and relation context vectors and stores
        to self class objetcs
        '''


        triples_df = self.train_set.append(self.valid_set, ignore_index = True)
        heads = triples_df.e1.values
        tails = triples_df.e2.values
        rels = triples_df.rel.values

        context_matrix = sparse.lil_matrix((len(self.rel_dict.keys()), len(self.ent_dict.keys())))
        
        for h, t, r in zip(heads, tails, rels):
            context_matrix[int(r[1:]), int(h[1:])] = 1
            context_matrix[int(r[1:]), int(t[1:])] = 1

        self.ent_context_matrix = context_matrix.T
        self.rel_context_matrix = context_matrix

    def get_context_matrix(self):

        triples_df = self.train_set.append(self.valid_set, ignore_index = True)
        heads = triples_df.e1.values
        tails = triples_df.e2.values
        rels = triples_df.rel.values

        context_matrix = sparse.lil_matrix((len(self.rel_dict.keys()), len(self.ent_dict.keys())))
        
        for h, t, r in zip(heads, tails, rels):
            context_matrix[int(r[1:]), int(h[1:])] = 1
            context_matrix[int(r[1:]), int(t[1:])] = 1

        return context_matrix.T

    def get_ent_similarity_matrix(self):

        matrix = self.get_context_matrix()

        return cosine_similarity(matrix, matrix)

    def get_rel_similarity_matrix(self):

        matrix = self.get_context_matrix().T

        return cosine_similarity(matrix, matrix)
    
    def find_topn_similar_ents(self, index, top_n = 5, threshold = -10.0):

        query = index
        cosine_similarities = cosine_similarity(self.ent_context_matrix[index:index+1], self.ent_context_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

        return [(self.ent_dict[query], index, self.ent_dict[index], cosine_similarities[index]) for index in related_docs_indices if cosine_similarities[index] > threshold][0:top_n]

    def find_topn_similar_rels(self, index, top_n = 5, threshold = -10.0):

        query = index
        cosine_similarities = cosine_similarity(self.rel_context_matrix[index:index+1], self.rel_context_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

        return [(self.rel_dict[query], index, self.rel_dict[index], cosine_similarities[index]) for index in related_docs_indices if cosine_similarities[index] > threshold][0:top_n]

    def print_triple(self, triple):

        triple = triple.split('_')
        print('e1: {}'.format(self.ent_dict[triple[0]]))
        print('rel: {}'.format(self.rel_dict[triple[3]]))
        print('e2: {}'.format(self.ent_dict[triple[1]]))

    def rebuild_path(self, triple, path):

        triple = triple.split('_')
        head = triple[0]
        tail = triple[1]
        rel = triple[3]

        path = path.split('_')
        path_len = len(path)

        pipe = deque()

        output = []

        #initial expansion

        neighbor_nodes = self.graph.get(head, dict()).get(path[0], [])
        for node in neighbor_nodes:
            if (node != head) & (node != tail):
                pipe.append([node])

        while pipe:

            current_path = pipe.popleft()
            current_len = len(current_path)

            neighbor_nodes = self.graph.get(current_path[-1], dict()).get(path[current_len], [])
            if current_len < path_len -1:
                for node in neighbor_nodes:
                    if (node != head) & (node != tail):
                        pipe.append(current_path + [node])
                    continue
            else:
                if tail in neighbor_nodes:
                    output.append(current_path)

        complete_path = []
        if output:
            for path in output:
                complete_path.append([head] + path + [tail])

        return complete_path

    def textual_path_explanation(self, tr, pt):

        rels_list = self.explain_path(pt).split(' | ')

        intermediate_nodes = self.rebuild_path(tr, pt)

        output = []

        for feature in intermediate_nodes:
            ent_names = ['(' + self.ent_dict[n] + ')' for n in feature]
            complete_path = list(itertools.chain.from_iterable(zip(ent_names,rels_list)))
            complete_path.append(ent_names[-1])
            output.append(complete_path)

        return output

    def compute_node_context(self, path, theta = 0.5):

        head = path[0]
        tail = path[-1]
        int_nodes = path[1:-1]

        sim_h_t = self.sm_model.similarity(head, tail)

        similarities = []

        for node in int_nodes:
            similarities.append(theta * self.sm_model.similarity(node, head) + (1 -theta) * self.sm_model.similarity(node, tail))
        
        return [sim_h_t] + similarities + [sim_h_t]

    def get_most_similar(self, e1, topn = 10):

        print('Query: {} {}'.format(e1, self.names_dict[e1]))

        return [(e[0], self.names_dict[e[0]], e[1]) for e in self.sm_model.wv.most_similar(e1, topn = topn ) ]

    def get_similar_rels(self, e1, topn = 10):

        print('Query: {} {}'.format(e1, self.names_dict[e1]))

        return [(e[0], self.names_dict[e[0]], e[1]) for e in self.sm_model.wv.most_similar(e1, topn=1000) if e[0][0] == 'r'][:topn]

    def compute_avg_rel_sim(self, rel, path):
        '''
        This method computes the average relation similarity for a given path
        '''

        path = path.split('_')
        sim = 0
        for pt in path:
            sim += self.sm_model.similarity(rel, pt)
        
        return sim / len(path)

    def compute_inpath_similarity(self, path):
        '''
        This method computes the in_path dissimilarity between rels within a given path
        '''
        
        path = path.split('_')

        if len(path) == 1: return 1

        if len(path) == 2: return self.sm_model.similarity(path[0], path[1])

        if len(path) == 3: return ((1 - self.sm_model.similarity(path[0], path[1])) + (1-self.sm_model.similarity(path[1], path[2]))) / 2

        if len(path) == 4: return ((1 - self.sm_model.similarity(path[0], path[1])) + (1-self.sm_model.similarity(path[1], path[2])) + (1-self.sm_model.similarity(path[3], path[3]))) / 3

#####################################################################
#
# Similarity Model Auxiliary Functions

    def build_adjacent_nodes_for_rel(self, rel_id):
        '''
        This model returns a list of adjacent nodes (head and tail) for a 
        given relation
        
        Methods using this method:
        extract_features

        '''

        df = self.train_set.append(self.valid_set, ignore_index=True)
        df = df[df['rel'] == rel_id]

        return list(df.e1.values) + list(df.e2.values)

    def extract_features_3(self, rel_id, count = False):
        '''
        This method returns a dict containing all the adjacent rels for the given relation,
        the output is a dict whose keys are adjacent relations and values are the counts
        If not_count is True, return 1, else returns the number of times each rel appears.
        '''
        # we get the all the heads and tails (train + valid datasets)
        nodes = self.build_adjacent_nodes_for_rel(rel_id)

        feature = []
        
        # for each node we build a set of features
        for node in nodes:
            feature += list(self.graph[node].keys())
        if not count:
            feature = list(set(feature))

        return dict(Counter(feature))

    def compute_avg_rel_similarities(self, path, rel_id):

        path = path.split('_')
        sims = []
        for edge in path:
            sims.append(self.rel_similarity_matrix[int(rel_id[1:]), int(edge.replace('i', '')[1:])])
        
        return np.mean(np.array(sims))

    def compute_min_rel_similarities(self, path, rel_id):

        path = path.split('_')
        sims = []
        for edge in path:
            sims.append(self.rel_similarity_matrix[int(rel_id[1:]), int(edge.replace('i', '')[1:])])
        
        return min(np.array(sims))

#####################################################################
# Similarity Models

    ''' Model #3

    This similarity model considers a vector for each relation, considering
    all relations that are connected to heads and tails, without making any distinction
    wheter it is connected to head or tail

    '''
    def fit_model_3(self):

        # get a list of the rels for this dataset
        rel_ids = list(self.rel_dict.keys())

        vectors = []

        for rel_id in tqdm(rel_ids):

            vectors += [self.extract_features_3(rel_id, count = False)]

        v = DictVectorizer(sparse = True)

        rel_vectors = v.fit_transform(vectors)
        
        kv = gensim.models.keyedvectors.WordEmbeddingsKeyedVectors(rel_vectors.shape[1])

        kv.add(rel_ids ,np.array(rel_vectors.todense()))

        return kv



