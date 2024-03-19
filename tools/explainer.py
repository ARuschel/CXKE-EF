
from __future__ import division
import os
import time
import pandas as pd
import numpy as np
import logging
import math


from io import StringIO
from csv import writer

import config
from tools.tools import get_dirs, write_to_pkl, load_file, restore_model
from tools.dataset_tools import Dataset
from tqdm import tqdm
from collections import Counter, deque, defaultdict
import _pickle as pickle

# from lazypredict.Supervised import LazyClassifier

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, euclidean_distances, pairwise_distances

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
from sklearn.dummy import DummyClassifier

from sfe_ar.tools.helpers import generate_timestamp

class Explainer(Dataset):

    def __init__(self, dataset, emb_model, timestamp_emb, timestamp_sfe, splits, method='fast'):

        self.dataset = dataset
        self.emb_model = emb_model
        self.timestamp_emb = timestamp_emb
        self.timestamp_sfe = timestamp_sfe
        self.method = method
        self.project_folder = os.path.expanduser('~') + f'/proj/XKE_results/{dataset}/'
        self.splits_folder = self.project_folder + f'splits/{splits}/'
        self.feature_folder = self.project_folder + f'sfe_features/{timestamp_sfe}/'
        self.embeddings_folder = self.project_folder + f'embeddings/{emb_model}/{timestamp_emb}/'
        self.logit_results_folder = self.project_folder + f'xke_explain/{emb_model}_{timestamp_emb}_sfe{timestamp_sfe}/'
        self.emb_results_folder = self.project_folder + f'emb_results/{emb_model}_{timestamp_emb}_{splits}/'
        self.sm_folder = self.project_folder + 'sm_models/'

        super().__init__(self.dataset) 

        self.model_info = dict()
        self.sfe_model_info = dict()
        self.emb_model_info = dict()
        self.emb_metrics_info = dict()
        self.load_model_info()
        self.load_emb_model_info()

        self.len_train_triples = 0
        self.len_valid_triples = 0
        self.prune_dict = dict()
        self.param_grid_logit = dict()
        self.metrics = dict()
        self.logit_models = dict()
        self.log_original_path = None

        self.emb = False
        self.g_hat = None
        self.g_hat_dict = None

        if method == 'fast':
            self.load_data_engine = self.fast_load_data
        if method == 'book_keeping':
            self.load_data_engine = self.load_data

        self.logger = logging.getLogger(__name__)

    def write_to_pkl(self, filenm, object_to_save):

        file_name = '{}.pkl'.format(filenm)

        with open(file_name, 'wb') as f:
            pickle.dump(object_to_save, f)

    def load_from_pkl(self, filenm):

        file_name = '{}.pkl'.format(filenm)
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        
        return obj

    def set_prune_dict(self, prune_dict):

        self.prune_dict = prune_dict

    def set_param_grid_logit(self, param_grid_logit):

        self.param_grid_logit = param_grid_logit 

    def load_g_hat(self):

        file_name = self.emb_results_folder + 'g_hat'

        print('Loadding g_hat file...', end=' ')
        self.g_hat = self.load_from_pkl(file_name)
        print('Done!')

    def build_g_hat_dict(self):

        print('Building g_hat_dict...', end=' ')
        time.sleep(0.2)

        self.g_hat_dict = dict()

        for rel in tqdm(self.g_hat.keys()):

            inv_rel = 'i' + rel

            self.g_hat_dict[rel] = defaultdict(list)
            self.g_hat_dict[inv_rel] = defaultdict(list)

            for h, t in zip(self.g_hat[rel].nonzero()[0], self.g_hat[rel].nonzero()[1]):
                self.g_hat_dict[rel][h].append(t)
                self.g_hat_dict[inv_rel][t].append(h)

        # self.g_hat = None

        print('Done!')

    def load_model_info(self):
        
        try:
            self.sfe_model_info = dict(pd.read_csv(self.feature_folder + 'sfe_model_info.tsv', sep='\t', index_col=0))
            print('\nLoaded sfe_model_info!')
        except:
            print('\nCould not load sfe_model_info.')
            self.model_info = dict()

    def load_emb_model_info(self):

        try:
            emb_model_info = pd.read_csv(self.embeddings_folder + 'model_info.tsv', sep='\t')
            self.emb_model_info = {
                'emb:model':                emb_model_info.loc[0, 'model'],
                'emb:timestamp':            emb_model_info.loc[0, 'timestamp'],
                'emb:overall_test_acc':     emb_model_info.loc[0, 'acc']
            }
            print(f'Loaded emb_model_info for {self.timestamp_emb} timestamp!')
            

        except:
            print(f'FAILED to load emb_model_info for {self.timestamp_emb} timestamp!')
            pass

        try:
            self.emb_metrics_info = dict(pd.read_csv(self.emb_results_folder + 'emb_metrics.tsv', 
            sep='\t', index_col=0).T)
        except:
            pass

        print(self.emb_model_info)
        
    def load_splits_for_rel(self, rel):
        '''
        This method loads splits files - it is different from the Dataset similar method
        because it returns the files instead of converting to dicts and saving
        to attributes
        '''

        rel_splits_folder = self.splits_folder + rel + '/'

        train_df = pd.read_csv(rel_splits_folder + 'train.tsv', sep='\t')
        train_df['subset'] = 'train'
        self.len_train_triples = len(train_df)
        valid_df = pd.read_csv(rel_splits_folder + 'valid.tsv', sep='\t')
        valid_df['subset'] = 'valid'
        self.len_valid_triples = len(valid_df)
        test_df = pd.read_csv(rel_splits_folder + 'test.tsv', sep='\t')
        test_df['subset'] = 'test'
        splits_df = pd.concat([train_df, test_df, valid_df])
        # print(splits_df.shape)
        # print(splits_df.head())

        return splits_df

    def load_sfe_features(self, rel):

        file_to_load = self.feature_folder + rel + '/combined_features.fset'
        # print(file_to_load)

        sfe = load_file(file_to_load)

        # sfe['p_len'] = sfe['p_len'].astype(str)

        sfe['pos'] = sfe['label'].apply(lambda x: 1 if x==1 else 0)
        sfe['neg'] = sfe['label'].apply(lambda x: 1 if x==0 else 0)

        print('\nLoaded SFE Matrix with {} unique features.'.format(sfe.path.nunique()))
  
        return sfe

    def prune_features(self, sfe):
        '''
        This method gets sfe matrix and performs a set of prunnings according to parameters
        stored in the params_dict. 

        It returns a list of selected features that will be later used to filter the 
        sfe features again.
        '''
        print('Performing prunning process.')

        #Calculating the proportions of positive and negative triples per feature
        props =  sfe[ (sfe['subset'] == 'train') | (sfe['subset'] == 'valid')].pivot_table(index='path', values=['pos', 'neg'], aggfunc='sum').reset_index().fillna(0)
        props['p_len'] = props['path'].apply(lambda x: len(x.split('_')))
        props['triples'] = props.pos + props.neg
        props['rate'] = 100 * props.pos / (props.pos + props.neg)
        # props['prop'] = 100 * props.triples / (self.len_train_triples + self.len_valid_triples)

        #First we select train and validation subsets, then we make a pivot table to get the triple count per feature (popularity)
        ft_sel = sfe[ (sfe['subset'] == 'train') | (sfe['subset'] == 'valid')].pivot_table(values=['triple_id', 'min_rel_sim', 'avg_rel_sim'], index='path', aggfunc={'triple_id':'count', 'min_rel_sim':'first', 'avg_rel_sim':'first'}).rename(columns={'triple_id':'triple_count'})
        ft_sel = pd.merge(ft_sel, props, how='left', on='path')

        n = self.len_train_triples + self.len_valid_triples
        selected_features = []

        #Here we apply prunning by popularity
        if self.prune_dict.get('pru:top_pop', False):
            df_pop = ft_sel.sort_values(by=['triple_count', 'rate'], ascending=[False, False])
            df_pop = df_pop.head(int(self.prune_dict['pru:top_pop'] * n))
            selected_features += list(df_pop.path.values)
            self.metrics['pru:top_pop'] = self.prune_dict['pru:top_pop']
        else:
            self.metrics['pru:top_pop'] = 'no'

        #Here we apply prunning by top negative features (proportion of neg triples x total)        
        if self.prune_dict.get('pru:top_neg', False):
            df_top_neg = ft_sel.sort_values(by=['rate', 'triple_count'], ascending=[True, False])
            df_top_neg = df_top_neg.head(int(self.prune_dict['pru:top_neg'] * n))
            selected_features += list(df_top_neg.path.values)
            self.metrics['pru:top_neg'] = self.prune_dict['pru:top_neg']
        else:
            self.metrics['pru:top_neg'] = 'no'

        #Here we select top positive features (proportion of pos triples x total)
        if self.prune_dict.get('pru:top_pos', False):
            df_top_pos = ft_sel.sort_values(by=['rate', 'triple_count'], ascending=[False, False])
            df_top_pos = df_top_pos.head(int(self.prune_dict['pru:top_pos'] * n))
            selected_features += list(df_top_pos.path.values)
            self.metrics['pru:top_pos'] = self.prune_dict['pru:top_pos']
        else:
            self.metrics['pru:top_pos'] = 'no'

        #Here we select features with greatest min rel similarity
        if self.prune_dict.get('pru:top_avg_rel_sim', False):
            df_top_rel_sim = ft_sel.sort_values(by=['avg_rel_sim', 'triple_count'], ascending=[False, False])
            df_top_rel_sim = df_top_rel_sim.head(int(self.prune_dict['pru:top_avg_rel_sim'] * n))
            selected_features += list(df_top_rel_sim.path.values)
            self.metrics['pru:top_avg_rel_sim'] = self.prune_dict['pru:top_avg_rel_sim']
        else:
            self.metrics['pru:top_avg_rel_sim'] = 'no'


        #Removing duplicated entries from selected features
        selected_features = list(set(selected_features))

        print('Prunning resulted in {} unique features.\n'.format(len(selected_features)))

        return selected_features

    def load_emb_predictions(self, rel_name):

        train = pd.read_csv(self.emb_results_folder + rel_name + '/train.tsv', sep='\t', index_col=0)
        
        try:
            valid = pd.read_csv(self.emb_results_folder + rel_name + '/valid.tsv', sep='\t', index_col=0)
        except:
            valid = pd.DataFrame(columns=['e1',	'e2', 'label', 'rel', 'triple_id', 'rel_thres',	'emb_score', 'emb_pred'])
        
        test = pd.read_csv(self.emb_results_folder + rel_name + '/test.tsv', sep='\t', index_col=0)

        return pd.concat([train, valid, test])

    def fast_load_sfe_features(self, rel_name):


        rel_folder = self.feature_folder + rel_name

        train_pack = self.load_from_pkl(rel_folder + '/train_feature_package')
        valid_pack = self.load_from_pkl(rel_folder + '/valid_feature_package')
        test_pack = self.load_from_pkl(rel_folder + '/test_feature_package')

        return train_pack, valid_pack, test_pack

    def fast_prune_features(self, sfe):
        '''
        This method gets sfe matrix and performs a set of prunnings according to parameters
        stored in the params_dict. 

        It returns a list of selected features that will be later used to filter the 
        sfe features again.
        '''
        print('Performing prunning process.')

        n = self.len_train_triples + self.len_valid_triples
        selected_features = []

        #Here we apply prunning by popularity
        if self.prune_dict.get('pru:top_pop', False):
            df_pop = sfe.sort_values(by=['triple_count'], ascending=[False])
            df_pop = df_pop.head(int(self.prune_dict['pru:top_pop'] * n))
            selected_features += list(df_pop.path.values)
            self.metrics['pru:top_pop'] = self.prune_dict['pru:top_pop']
        else:
            self.metrics['pru:top_pop'] = 'no'

        #Here we select features with greatest min rel similarity
        if self.prune_dict.get('pru:top_avg_rel_sim', False):
            df_top_rel_sim = sfe.sort_values(by=['avg_rel_sim'], ascending=[False])
            df_top_rel_sim = df_top_rel_sim.head(int(self.prune_dict['pru:top_avg_rel_sim'] * n))
            selected_features += list(df_top_rel_sim.path.values)
            self.metrics['pru:top_avg_rel_sim'] = self.prune_dict['pru:top_avg_rel_sim']
        else:
            self.metrics['pru:top_avg_rel_sim'] = 'no'

        #Removing duplicated entries from selected features
        selected_features = list(set(selected_features))

        print('Prunning resulted in {} unique features.'.format(len(selected_features)))

        return selected_features

    def get_feature_statistics(self, rel_id, train, valid):

        #First we get the train fetures and triple counts
        tf = pd.DataFrame()
        tf['path'] = train['vectorizer'].feature_names_
        tf['train_triples'] = np.array(train['feature_matrix'].sum(axis=0))[0]

        #Then, the same for the valid features
        vf = pd.DataFrame()
        vf['path'] = valid['vectorizer'].feature_names_
        vf['valid_triples'] = np.array(valid['feature_matrix'].sum(axis=0))[0]

        #Then we merge both dataframes to obtain a single dataframe
        df = pd.merge(tf, vf, how='outer', left_on='path', right_on='path').fillna(0)
        df['triple_count'] = df['train_triples'] + df['valid_triples']

        #Now we compute statistics        
        df['p_len'] = df['path'].apply(lambda x: len(x.split('_')))
        df['avg_rel_sim'] = df['path'].apply(lambda x: self.compute_avg_rel_similarities(x, rel_id))
        df['min_rel_sim'] = df['path'].apply(lambda x: self.compute_min_rel_similarities(x, rel_id))

        return df

    def fast_load_data(self, rel_name):

        # print('Loading data for relation: {}'.format(rel_name))

        rel_id = self.rel_dict_rev[rel_name]

        splits_df = self.load_splits_for_rel(rel_name)
        emb_pred = self.load_emb_predictions(rel_name)


        '''Each pack contains:
         - vectorizer: an object used to build sparse matrices using a list of dicts
         - row_names: a list with the name of each triple of the subset, in the same order as feature_extractor
         - feature_matrix: a sparse csr matrix with triples as rows and features (paths) as columns
         - feature_extractor: a list of dicts containing the raw features extracted by sfe
        '''
        train_pack, valid_pack, test_pack = self.fast_load_sfe_features(rel_name)


        ## Filtering and Prunning process:
        #Now we get the feature statistics for the train and valid subsets:
        sfe = self.get_feature_statistics(rel_id, train_pack, valid_pack)
        n = self.len_train_triples + self.len_valid_triples
        
        #Here we can prune paths containing rels with similarity to the current rel less 
        #than the desired level
        min_rel_sim = self.prune_dict.get('pru:min_rel_sim', False)
        if min_rel_sim:
            print('Prunning min_rel_sim')
            sfe = sfe[sfe['min_rel_sim'] > min_rel_sim]
            self.metrics['pru:min_rel_sim'] = self.prune_dict.get('min_rel_sim', False)
        else:
            self.metrics['pru:min_rel_sim'] = 'no'

        #In fast implementation we do not carry node_relv_in values, so this is only to set pru:node_relv_in to no
        self.metrics['pru:node_relv_in'] = 'no'

        #Same fashion for top_pos and top_neg
        self.metrics['pru:top_pos'] = 'no'
        self.metrics['pru:top_neg'] = 'no'

        #Here we can limit p_len
        if self.prune_dict.get('pru:max_l', False):
            print('Selecting features with p_len up to {} rels.'.format(self.prune_dict['pru:max_l']))
            sfe = sfe[sfe['p_len'] <= self.prune_dict['pru:max_l']]
            self.metrics['pru:max_l'] = self.prune_dict['pru:max_l']
        else:
            self.metrics['pru:max_l'] = 'no'

        #Here we call prunning function when pru:prunning is set to force
        if self.prune_dict.get('pru:prunning', False) == 'force':
            selected_features = self.fast_prune_features(sfe)
            sfe = sfe[sfe['path'].isin(selected_features)]
            self.metrics['pru:prunning'] = 'force'
        else:
            # if prunning is not forced, we must creat a list with all features that latter will prune up to the max_features parameter
            selected_features = []
            df_pop = sfe.sort_values(by=['triple_count'], ascending=[False])
            selected_features = list(df_pop.path.values)

        # If prunning is not forced, here we call prunning again just to make sure we do not pass a matrix 
        
        max_features = 100000
        print(f'Training Examples: {n} / unique_paths: {sfe.path.nunique()} / max_features set {max_features}\n')

        if sfe.path.nunique() > max_features:
            print('Activated poka-yoke!\n')
            # self.prune_dict['pru:top_pop'] = 0
            # self.prune_dict['pru:top_avg_rel_sim'] = 'forced'
            # selected_features = self.fast_prune_features(sfe)
            sfe = sfe[sfe['path'].isin(selected_features[:max_features])]
            self.metrics['pru:prunning'] = f'up to {max_features}'

        # Finally build a counter dict containing only the selected features that later fill be fed
        # into de DictVectorizer constructor
        final_selected_features = [Counter(sfe.path.values)]

        # Let us make some training cases alignment just for sure, here we maintain train and valid subsets together
        # because we are going to make cross-validation for hyperparameter tunning
        # We merge splits with embeddings predictions resulting in a dataframe with triple_id, subset, label, and emb_pred
        splits_df = pd.merge(splits_df, emb_pred[['triple_id', 'emb_pred']], how='left', on='triple_id').drop(columns=['e1', 'e2', 'rel'])
        train_split = splits_df[splits_df['subset'] != 'test']
        test_split = splits_df[splits_df['subset'] == 'test']

        features = {**train_pack['feature_extractor'], **valid_pack['feature_extractor'], **test_pack['feature_extractor']}

        train_features = []
        for triple in train_split.triple_id.values:
            train_features.append(features[triple])

        test_features = []
        test_triples = list(test_split.triple_id.values)
        for triple in test_triples:
            test_features.append(features[triple])
        

        #Now we use Dictvectorizer to build train and test subsets, using selected_features:
        v = DictVectorizer(sparse=True)
        v.fit(final_selected_features)
        feature_names = v.feature_names_

        X_train = v.transform(train_features)
        X_test = v.transform(test_features)

        y_train = train_split.label.values
        y_train_emb = train_split.emb_pred.values
        y_test = test_split.label.values
        y_test_emb = test_split.emb_pred.values

        # From here, everything is only for statistics calculations and rather unnecessary, but...
        train_pos_triples = train_split[train_split['label'] == 1].triple_id.values
        train_pos_features = []
        for triple in train_pos_triples:
            train_pos_features.append(features[triple])
        X_train_pos = v.transform(train_pos_features)

        train_neg_triples = train_split[train_split['label'] == 0].triple_id.values
        train_neg_features = []
        for triple in train_neg_triples:
            train_neg_features.append(features[triple])
        X_train_neg = v.transform(train_neg_features)

        test_pos_triples = test_split[test_split['label'] == 1].triple_id.values
        test_pos_features = []
        for triple in test_pos_triples:
            test_pos_features.append(features[triple])
        X_test_pos = v.transform(test_pos_features)

        test_neg_triples = test_split[test_split['label'] == 0].triple_id.values
        test_neg_features = []
        for triple in test_neg_triples:
            test_neg_features.append(features[triple])
        X_test_neg = v.transform(test_neg_features)

        self.metrics['pru:train_triples_pos_w_feat'] = float(np.count_nonzero(X_train_pos.todense().sum(axis=1)) / X_train_pos.shape[0])
        self.metrics['pru:train_triples_neg_w_feat'] = float(np.count_nonzero(X_train_neg.todense().sum(axis=1)) / X_train_neg.shape[0])
        self.metrics['pru:train_triples_w_feat'] = float(np.count_nonzero(X_train.todense().sum(axis=1)) / X_train.shape[0])

        self.metrics['pru:test_triples_pos_w_feat'] = float(np.count_nonzero(X_test_pos.todense().sum(axis=1)) / X_test_pos.shape[0])
        self.metrics['pru:test_triples_neg_w_feat'] = float(np.count_nonzero(X_test_neg.todense().sum(axis=1)) / X_test_neg.shape[0])
        self.metrics['pru:test_triples_w_feat'] = float(np.count_nonzero(X_test.todense().sum(axis=1)) / X_test.shape[0])

        self.metrics['pru:train_feature_count'] = np.count_nonzero(X_train.todense().sum(axis=0))
        self.metrics['pru:test_feature_count'] = np.count_nonzero(X_test.todense().sum(axis=0))
        self.metrics['pru:common_feature_count'] = np.count_nonzero(np.multiply(X_test.todense().sum(axis=0), X_train.todense().sum(axis=0)))
        
        return X_train, y_train, y_train_emb, X_test, y_test, y_test_emb, X_train_pos, X_train_neg, X_test_pos, X_test_neg, feature_names, test_triples

    def load_data(self, rel_name):

        print('\nLoading data for {} relation.'.format(rel_name))

        splits_df = self.load_splits_for_rel(rel_name)
        sfe = self.load_sfe_features(rel_name)
        emb_pred = self.load_emb_predictions(rel_name)

        sfe['counts'] = 1

        #now we split train+valid subsets from test subset:
        #all prunning process is performing taking into account only features from train+valid subsets

        sfe_test = sfe[sfe['subset'] == 'test']
        sfe = sfe[sfe['subset'] != 'test']

        n = self.len_train_triples + self.len_valid_triples
        

        #In case sfe was extract without node_relv_in, nodes with node_relv less than head/tail similarity
        #can be prunned here
        if self.prune_dict.get('pru:node_relv_in', False):
            print('Prunning Relv!')
            sfe = sfe[sfe['node_relv_in'] > 0]
            self.metrics['pru:node_relv_in'] = self.prune_dict.get('pru:node_relv_in', False)
        else:
            self.metrics['pru:node_relv_in'] = 'no'

        #Here we can prune paths containing rels with similarity to the current rel less 
        #than the desired level
        min_rel_sim = self.prune_dict.get('pru:min_rel_sim', False)
        if min_rel_sim:
            print('Prunning min_rel_sim')
            sfe = sfe[sfe['min_rel_sim'] > min_rel_sim]
            self.metrics['pru:min_rel_sim'] = self.prune_dict.get('min_rel_sim', False)
        else:
            self.metrics['pru:min_rel_sim'] = 'no'

        #Here we can limit p_len
        if self.prune_dict.get('pru:max_l', False):
            print('Selecting features with p_len up to {} rels.'.format(self.prune_dict['pru:max_l']))
            sfe = sfe[sfe['p_len'] <= self.prune_dict['pru:max_l']]
            self.metrics['pru:max_l'] = self.prune_dict['pru:max_l']
        else:
            self.metrics['pru:max_l'] = 'no'

        #Here we call prunning function when pru:prunning is set to force
        if self.prune_dict.get('pru:prunning', False) == 'force':
            selected_features = self.prune_features(sfe)
            sfe = sfe[sfe['path'].isin(selected_features)]
            self.metrics['pru:prunning'] = 'force'

        # If prunning is not forced, here we call prunning again just to make sure we do not pass a matrix 
        # with p greater than n to the logistic regressor
        # As a last resort we override pru:top_pop to 0.8
        
        max_features = int(0.8 * n)
        print('\nTraining Examples: {} / max_features set {}'.format(n, max_features))

        if sfe.path.nunique() > max_features:
            self.prune_dict['pru:top_pop'] = 0.8
            selected_features = self.prune_features(sfe)
            sfe = sfe[sfe['path'].isin(selected_features)]
            self.metrics['pru:prunning'] = 'automatic'

        #Now we select from the test subset the same features obtained for train+valid 
        sfe_test = sfe_test[sfe_test['path'].isin(list(sfe.path.unique()))]

        feature_matrix = pd.concat([sfe, sfe_test])
        
        wide_feature_matrix = feature_matrix.pivot(index='triple_id', columns='path', values='counts').fillna(0)

        logit_matrix = pd.merge(splits_df[['subset', 'triple_id', 'label']], wide_feature_matrix, how='left', on='triple_id').fillna(0)
        logit_matrix = pd.merge(logit_matrix, emb_pred[['triple_id', 'emb_pred']], how='left', on='triple_id').fillna(0)
        logit_matrix['label'] = logit_matrix['label'].apply(lambda x: x if x ==1 else 0)

        df_train = logit_matrix[(logit_matrix['subset'] == 'train') | (logit_matrix['subset'] == 'valid')].copy().set_index('triple_id')
        df_train_pos = df_train[df_train['label'] == 1].copy()
        df_train_pos.drop(columns=['subset', 'label', 'emb_pred'], inplace=True)
        X_train_pos = df_train_pos.values

        df_train_neg = df_train[df_train['label'] == 0].copy()
        df_train_neg.drop(columns=['subset', 'label', 'emb_pred'], inplace=True)
        X_train_neg = df_train_neg.values

        pos_triples_w_feat, pos_triples, neg_triples_w_feat, neg_triples= self.count_triples_w_feat(df_train)
        self.metrics['pru:train_triples_pos_w_feat'] = pos_triples_w_feat / pos_triples
        self.metrics['pru:train_triples_neg_w_feat'] = neg_triples_w_feat / neg_triples
        self.metrics['pru:train_triples_w_feat'] = (pos_triples_w_feat + neg_triples_w_feat) / (pos_triples + neg_triples)

        y_train = df_train.label.values
        y_train_emb = df_train.emb_pred.values
        df_train.drop(columns=['subset', 'label', 'emb_pred'], inplace=True)
        X_train = df_train.values

        feature_names = list(df_train.columns)

        df_test = logit_matrix[logit_matrix['subset'] == 'test'].copy().set_index('triple_id')
        
        df_test_pos = df_test[df_test['label'] == 1].copy()
        df_test_pos.drop(columns=['subset', 'label', 'emb_pred'], inplace=True)
        X_test_pos = df_test_pos.values

        df_test_neg = df_test[df_test['label'] == 0].copy()
        df_test_neg.drop(columns=['subset', 'label', 'emb_pred'], inplace=True)
        X_test_neg = df_test_neg.values

        pos_triples_w_feat, pos_triples, neg_triples_w_feat, neg_triples= self.count_triples_w_feat(df_test)
        self.metrics['pru:test_triples_pos_w_feat'] = pos_triples_w_feat / pos_triples
        self.metrics['pru:test_triples_neg_w_feat'] = neg_triples_w_feat / neg_triples
        self.metrics['pru:test_triples_w_feat'] = (pos_triples_w_feat + neg_triples_w_feat) / (pos_triples + neg_triples)

        y_test = df_test.label.values
        y_test_emb = df_test.emb_pred.values
        df_test.drop(columns=['subset', 'label', 'emb_pred'], inplace=True)
        X_test = df_test.values

        self.metrics['pru:train_feature_count'], self.metrics['pru:test_feature_count'], self.metrics['pru:common_feature_count'] = self.count_features_per_subset(df_train, df_test)
        
        return X_train, y_train, y_train_emb, X_test, y_test, y_test_emb, X_train_pos, X_train_neg, X_test_pos, X_test_neg, feature_names

    def count_features_per_subset(self, df_train, df_test):

        features = np.array(df_train.columns)

        train_features = set(features[df_train.values.sum(axis = 0) != 0])
        test_features = set(features[df_test.values.sum(axis = 0) != 0])

        return len(train_features), len(test_features), len(train_features & test_features)

    def count_triples_w_feat(self, df):

        pos = df[df['label'] == 1].copy()
        pos.drop(columns=['subset', 'label'], inplace=True)
        pos_triples_w_feat = np.count_nonzero(np.count_nonzero(pos.values, axis=1))

        neg = df[df['label'] == 0].copy()
        neg.drop(columns=['subset', 'label'], inplace=True)
        neg_triples_w_feat = np.count_nonzero(np.count_nonzero(neg.values, axis=1))

        return pos_triples_w_feat, len(pos), neg_triples_w_feat, len(neg)

    def process_metrics(self, method_metric, model, feature_names, X_train, y_train, X_test, y_test, X_train_pos, X_train_neg, X_test_pos, X_test_neg):

        method = method_metric[0]
        xke_metric = method_metric[1]

        self.metrics['{}:{}train_acc'.format(method, xke_metric)] = float(model.score(X_train, y_train))
        self.metrics['{}:{}train_f1'.format(method, xke_metric)] = float(f1_score(y_true = y_train, y_pred = model.predict(X_train)))
        self.metrics['{}:{}train_tn'.format(method, xke_metric)], self.metrics['{}:{}train_fp'.format(method, xke_metric)], self.metrics['{}:{}train_fn'.format(method, xke_metric)], self.metrics['{}:{}train_tp'.format(method, xke_metric)] = confusion_matrix(y_true = y_train, y_pred = model.predict(X_train), labels=[0,1]).ravel()

        self.metrics['{}:{}test_acc'.format(method, xke_metric)] = float(model.score(X_test, y_test))
        self.metrics['{}:{}test_f1'.format(method, xke_metric)] = float(f1_score(y_true = y_test, y_pred = model.predict(X_test)))
        self.metrics['{}:{}test_AP'.format(method, xke_metric)] = float(average_precision_score(y_test, model.decision_function(X_test)))
        self.metrics['{}:{}test_tn'.format(method, xke_metric)], self.metrics['{}:{}test_fp'.format(method, xke_metric)], self.metrics['{}:{}test_fn'.format(method, xke_metric)], self.metrics['{}:{}test_tp'.format(method, xke_metric)] = confusion_matrix(y_true = y_test, y_pred = model.predict(X_test), labels=[0,1]).ravel()

        self.metrics['{}:train_triples_pos_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_train_pos, model.coef_.T)) / X_train_pos.shape[0])
        self.metrics['{}:train_triples_neg_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_train_neg, model.coef_.T)) / X_train_neg.shape[0])
        self.metrics['{}:test_triples_pos_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_test_pos, model.coef_.T)) / X_test_pos.shape[0])
        self.metrics['{}:test_triples_neg_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_test_neg, model.coef_.T)) / X_test_neg.shape[0])
        self.metrics['{}:train_triples_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_train, model.coef_.T)) / X_train.shape[0])
        self.metrics['{}:test_triples_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_test, model.coef_.T)) / X_test.shape[0])

        train_pos_feat_per_triple = (np.multiply(X_train_pos, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:train_pos_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(train_pos_feat_per_triple[np.nonzero(train_pos_feat_per_triple)]))

        train_neg_feat_per_triple = (np.multiply(X_train_neg, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:train_neg_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(train_neg_feat_per_triple[np.nonzero(train_neg_feat_per_triple)]))

        test_pos_feat_per_triple = (np.multiply(X_test_pos, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:test_pos_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(test_pos_feat_per_triple[np.nonzero(test_pos_feat_per_triple)]))

        test_neg_feat_per_triple = (np.multiply(X_test_neg, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:test_neg_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(test_neg_feat_per_triple[np.nonzero(test_neg_feat_per_triple)]))

        train_tot_feat_per_triple = (np.multiply(X_train, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:train_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(train_tot_feat_per_triple[np.nonzero(train_tot_feat_per_triple)]))

        test_tot_feat_per_triple = (np.multiply(X_test, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:test_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(test_tot_feat_per_triple[np.nonzero(test_tot_feat_per_triple)]))

        self.metrics['{}:neg_active_feat'.format(method)] = (model.coef_ < 0).sum()
        self.metrics['{}:pos_active_feat'.format(method)] = (model.coef_ > 0).sum()
        self.metrics['{}:active_feat'.format(method)] = (model.coef_ != 0).sum()

        mask = model.coef_[0] != 0
        feature_names = np.array(feature_names)
        active_features = feature_names[mask]

        active_rels = []
        p_len_active_features = []
        for feature in active_features:
            active_rels += feature.split('_')
            p_len_active_features.append(len(feature.split('_')))
        active_rels = list(set(active_rels))
        self.metrics['{}:active_rels'.format(method)] = len(active_rels)
        
        p_lens = Counter(p_len_active_features)
        self.metrics['{}:active_feat_count_p_len_1'.format(method)] = p_lens[1]
        self.metrics['{}:active_feat_count_p_len_2'.format(method)] = p_lens[2]
        self.metrics['{}:active_feat_count_p_len_3'.format(method)] = p_lens[3]
        self.metrics['{}:active_feat_count_p_len_4'.format(method)] = p_lens[4]

    def process_XKEe_metrics(self, model, X_test, y_test, false_negatives_corrected, false_positives_corrected):

        method = 'XKEe'
        xke_metric = 'fidelity_'

        self.metrics['XKEe:false_negatives_corrected'] = false_negatives_corrected
        self.metrics['XKEe:false_positives_corrected'] = false_positives_corrected

        self.metrics['{}:{}test_acc'.format(method, xke_metric)] = float(model.score(X_test, y_test))
        self.metrics['{}:{}test_f1'.format(method, xke_metric)] = float(f1_score(y_true = y_test, y_pred = model.predict(X_test)))
        self.metrics['{}:{}test_AP'.format(method, xke_metric)] = float(average_precision_score(y_test, model.decision_function(X_test)))
        self.metrics['{}:{}test_tn'.format(method, xke_metric)], self.metrics['{}:{}test_fp'.format(method, xke_metric)], self.metrics['{}:{}test_fn'.format(method, xke_metric)], self.metrics['{}:{}test_tp'.format(method, xke_metric)] = confusion_matrix(y_true = y_test, y_pred = model.predict(X_test), labels=[0,1]).ravel()

        self.metrics['{}:test_triples_w_active_feat'.format(method)] = float(np.count_nonzero(np.matmul(X_test, model.coef_.T)) / X_test.shape[0])

        test_tot_feat_per_triple = (np.multiply(X_test, model.coef_) !=0).sum(axis=1)
        self.metrics['{}:test_avg_feat_per_triple_w_active_feat'.format(method)] = float(np.mean(test_tot_feat_per_triple[np.nonzero(test_tot_feat_per_triple)]))

    def compute_interpretability_index(self, method, rel_id, feature_names, model_coefs, X_test):

        #first we build a list with all path similarity index 
        feature_similarities = []
        for path in feature_names:
            feature_similarities.append(self.compute_avg_rel_sim(rel_id, path))

        #applying abs to make all coefficients positive
        model_coefs = np.abs(model_coefs)

        #now we multiply X_test by coefs
        weighted_params = np.multiply(X_test, model_coefs)

        #normalizing each row, so we get the importance of each explanatory sentence by triple
        weighted_params = normalize(weighted_params, axis=1, norm='l1')

        #now we multiply each feature vector by the similarities vector
        weighted_params = weighted_params * np.array(feature_similarities)

        self.metrics['{}:overall_interpretability_index'.format(method)] = weighted_params.sum(1).mean()

        #now we select the features that have at least one active feature
        weighted_params_nonzero = weighted_params.sum(1)
        weighted_params_nonzero = weighted_params_nonzero[weighted_params_nonzero != 0]

        self.metrics['{}:overall_interpretability_index(triples_with_feature)'.format(method)] = weighted_params_nonzero.mean()

    def compute_interpretability_index_2(self, rel_id, feature_names, model_coefs, X_test):

        if (X_test > 0).sum() == 0:
            print('empty X_test file!')
            return 0, 0

        #first we build a list with all path similarity index 
        feature_similarities = []
        for path in feature_names:
            feature_similarities.append(self.compute_avg_rel_sim(rel_id, path))

        #applying abs to make all coefficients positive
        model_coefs = np.abs(model_coefs)

        #now we multiply X_test by coefs
        weighted_params = np.multiply(X_test, model_coefs)

        #normalizing each row, so we get the importance of each explanatory sentence by triple
        weighted_params = normalize(weighted_params, axis=1, norm='l1')

        #now we multiply each feature vector by the similarities vector
        weighted_params = weighted_params * np.array(feature_similarities)

        #now we select the features that have at least one active feature
        weighted_params_nonzero = weighted_params.sum(1)
        weighted_params_nonzero = weighted_params_nonzero[weighted_params_nonzero != 0]

        if (weighted_params.sum(1) > 0).sum() > 0:
            wp = weighted_params.sum(1).mean()
        else:
            wp = 0

        if (weighted_params_nonzero >0).sum() > 0:
            wpn = weighted_params_nonzero.mean()

        else:
            wpn = 0

        return wp, wpn

    def train_test_logit(self, logit_timestamp = False, override_results = False):

        t1 = time.time()

        if not logit_timestamp:
            logit_timestamp = generate_timestamp()
            print('Generated Explainer timestamp: {}'.format(logit_timestamp))
        else:
            print('Using existing Explainer timestamp: {}'.format(logit_timestamp))

        logit_timestamp_folder = self.logit_results_folder + logit_timestamp + '/'

        if not os.path.exists(logit_timestamp_folder):
            os.makedirs(os.path.join(logit_timestamp_folder))
            print('Creating folder: {}'.format(logit_timestamp_folder))

        logging.basicConfig(filename='{}/log.txt'.format(logit_timestamp_folder),
                level=logging.INFO,
                format='%(levelname)s: %(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S')

        self.logger.info('Initializing logger for explainer.')

        try:
            self.model_info = dict(pd.read_csv(logit_timestamp_folder + 'detailed_results.tsv', sep='\t', index_col=0))
        except:
            self.model_info = dict()
            print('Could not find any model_info file.')
            using_existing_metrics = False

        metrics_template = pd.read_csv('metrics_template.tsv', sep='\t', index_col=0)

        rels = get_dirs(self.feature_folder)

        i = 1
        total_rels = len(rels)

        for rel in rels:

            t2 = time.time()
            rel_id = self.rel_dict_rev[rel]

            print('\n==============================================================================')
            print('\nProcessing relation {}/{}.\n'.format(i, total_rels))
            print('{}: {}'.format(rel_id, rel))


            explain_rel_fname = logit_timestamp_folder + '{}_coefs.tsv'.format(rel_id)
            explanations_fname = logit_timestamp_folder + '{}_explanations.tsv'.format(rel_id)

            if not override_results:
                if os.path.exists(explain_rel_fname):
                    print('Skipping relation!')
                    i += 1
                    continue
                else:
                    self.model_info[rel_id] = dict()
            else:
                self.model_info[rel_id] = dict()

            self.metrics = dict() #clearing metrics dict for each relation
            self.logit_models[rel_id] = dict()

            X_train, y_train, y_train_emb, X_test, y_test, y_test_emb, X_train_pos, X_train_neg, X_test_pos, X_test_neg, feature_names, test_triples = self.load_data_engine(rel) 

            if len(np.unique(y_train)) < 2:
                print('Relation contains less than two classes for training, skipping!')
                continue

            if len(np.unique(y_train_emb)) < 2:
                print('Relation contains less than two classes for training, skipping!')
                continue

            t3 = time.time()

            #Here we perform pure SFE, ie, fitting the regression into ground truth labels
            print('Fitting SFE...', end = ' ')
            time.sleep(0.2)

            # if X_train.shape[-1] == 0:
            #     print('Using Dummy Classifier')
            #     logit_model = PriorClassifier()
            #     logit_model.fit(X_train, y_train)
            # else:
            gs = GridSearchCV(SGDClassifier(), self.param_grid_logit, n_jobs=10, refit=True, cv=5, verbose=0)
            time.sleep(0.2)
            gs.fit(X_train, y_train)
            logit_model = gs.best_estimator_
            print('Done!')

            time.sleep(0.2)
            self.process_metrics(('sfe', ''), logit_model, feature_names, X_train.todense(), y_train, X_test.todense(), y_test, X_train_pos.todense(), X_train_neg.todense(), X_test_pos.todense(), X_test_neg.todense())

            #evaluating overall intepretability index
            self.compute_interpretability_index('sfe', rel_id, feature_names, logit_model.coef_, X_test.todense())

            t4 = time.time()
            self.metrics['sfe:reg_elapsed_time'] = t4 - t3

            #Here we perform XKE, ie, fitting the regression with embedding labels
            print('Fitting XKE...', end = ' ')
            time.sleep(0.2)

            gs = GridSearchCV(SGDClassifier(), self.param_grid_logit, n_jobs=10, refit=True, cv=5, verbose=0)
            time.sleep(0.2)
            gs.fit(X_train, y_train_emb)
            xke_model = gs.best_estimator_
            xke_y_test_pred = xke_model.predict(X_test.todense())
            print('Done!')

            time.sleep(0.2)
            self.process_metrics(('xke', 'fidelity_'), xke_model, feature_names, X_train.todense(), y_train_emb, X_test.todense(), y_test_emb, X_train_pos.todense(), X_train_neg.todense(), X_test_pos.todense(), X_test_neg.todense())
            self.process_metrics(('xke', 'accuracy_'), xke_model, feature_names, X_train.todense(), y_train, X_test.todense(), y_test, X_train_pos.todense(), X_train_neg.todense(), X_test_pos.todense(), X_test_neg.todense())


            #evaluating overall intepretability index
            self.compute_interpretability_index('xke', rel_id, feature_names, xke_model.coef_, X_test.todense())

            t5 = time.time()
            self.metrics['xke:reg_elapsed_time'] = t5 - t4

            #Here we perform XKEe calculations
            XKEe_X_test, false_negatives_corrected, false_positives_corrected = self.build_X_test_pred_full(xke_model, X_test, test_triples, feature_names, y_test_emb)
            self.process_XKEe_metrics(xke_model, XKEe_X_test.todense(), y_test_emb, false_negatives_corrected, false_positives_corrected)
            xkee_y_test_pred = xke_model.predict(XKEe_X_test.todense())

            #evaluating overall intepretability index
            self.compute_interpretability_index('XKEe', rel_id, feature_names, xke_model.coef_, XKEe_X_test.todense())

            t6 = time.time()
            self.metrics['xke:total_elapsed_time'] = t6 - t1

            self.model_info[rel_id] = {**self.sfe_model_info[rel_id], **self.model_info[rel_id], **self.metrics, **self.emb_metrics_info[rel_id], **self.emb_model_info}
            # self.model_info[rel_id]['emb:timestamp'] = self.timestamp_emb

            #Now we store sfe and xke regression models as well as feature names for later evalutation
            self.logit_models[rel_id]['sfe'] = logit_model
            self.logit_models[rel_id]['xke'] = xke_model
            self.logit_models[rel_id]['feature_names'] = feature_names
            self.logit_models[rel_id]['test_triples'] = test_triples
            self.logit_models[rel_id]['X_test'] = X_test
            self.logit_models[rel_id]['y_test'] = y_test
            self.logit_models[rel_id]['xke_y_test_pred'] = xke_y_test_pred
            self.logit_models[rel_id]['y_test_emb'] = y_test_emb
            self.logit_models[rel_id]['XKEe_X_test'] = XKEe_X_test
            self.logit_models[rel_id]['XKEe_y_test_pred'] = xkee_y_test_pred
            self.write_to_pkl(logit_timestamp_folder + 'logit_models', self.logit_models)

            #This chunk builds a report of coeficients per relation (xke logit)
            explain_model = pd.DataFrame()
            explain_model['path'] = feature_names
            explain_model['coef'] = xke_model.coef_[0]
            explain_model['XKE_train_pos_triples'] = np.array(X_train[y_train_emb == 1].sum(axis=0))[0]
            explain_model['XKE_train_neg_triples'] = np.array(X_train[y_train_emb == 0].sum(axis=0))[0]
            explain_model['XKE_test_pos_triples'] = np.array(X_test[y_test_emb == 1].sum(axis=0))[0]
            explain_model['XKE_test_neg_triples'] = np.array(X_test[y_test_emb == 0].sum(axis=0))[0]
            explain_model['XKEe_test_pos_triples'] = np.array(XKEe_X_test[y_test_emb == 1].sum(axis=0))[0]
            explain_model['XKEe_test_neg_triples'] = np.array(XKEe_X_test[y_test_emb == 0].sum(axis=0))[0]
            explain_model['avg_rel_sim'] = explain_model.path.apply(lambda x: self.compute_avg_rel_sim(rel_id, x))
            explain_model['inpath_sim'] = explain_model.path.apply(lambda x: self.compute_inpath_similarity(x))
            explain_model['path_names'] = explain_model.path.apply(lambda x: self.explain_path(x))
            explain_model.sort_values(by='coef', ascending=False, inplace=True)
            explain_model.reset_index(inplace=True)
            explain_model.drop(columns='index', inplace=True)
            explain_model.loc[-1] = ['bias', logit_model.intercept_[0], '', '', '', '', '', '', '', '', '']
            explain_model.index = explain_model.index + 1
            explain_model.sort_index(inplace=True)
            explain_model.to_csv(explain_rel_fname, sep='\t')


            #Here we build a file with all textual explanations for each relation, including any feature that has been found
            #during XKEe procedure
            cols = ['idx', 'triple_id', 'triple', 'label', 'emb_label', 'XKE_label', 'XKEe_label', 'sim_index', 'coef', 'g_hat', 'explanation']
            feature_names = np.array(feature_names)
            feature_df = explain_model.copy()
            feature_df.set_index('path', inplace=True)


            output = StringIO()
            csv_writer = writer(output)

            j = 0
            for triple_id, emb_label, xke_label, xkee_label in zip(test_triples, y_test_emb, xke_y_test_pred, xkee_y_test_pred):

                triple = triple_id.split('_')
                triple_descr = str(self.ent_dict[triple[0]]) + ' | ' + str(self.ent_dict[triple[1]])
                label = triple[2]

                mask = np.array((X_test[j].toarray() != 0) & (xke_model.coef_ != 0))[0]

                active_features = feature_names[mask]

                for feature in active_features:
                    csv_writer.writerow([j]+[triple_id]+[triple_descr]+[label]+[emb_label]+[xke_label]+[xkee_label]+[feature_df.loc[feature, 'avg_rel_sim']]+[feature_df.loc[feature, 'coef']] + [0] + [self.explain_path(feature)])

                new_mask = np.array((XKEe_X_test.toarray()[j] > X_test.toarray()[j]) & (xke_model.coef_ != 0))[0]

                active_features = feature_names[new_mask]

                for feature in active_features:
                    csv_writer.writerow([j]+[triple_id]+[triple_descr]+[label]+[emb_label]+[xke_label]+[xkee_label]+[feature_df.loc[feature, 'avg_rel_sim']]+[feature_df.loc[feature, 'coef']]+ [1] + [self.explain_path(feature)])
                j += 1 
   
            output.seek(0)
            explanations = pd.read_csv(output, sep=',',names=cols)
            explanations.sort_values(by=['idx', 'g_hat', 'coef'], ascending=[True, True, False], inplace=True)
            explanations.to_csv(explanations_fname, sep='\t')



            #This chunk adds metrics from each run to the detailed_results file, it is within the loop in order to store results
            #after each run, in case of process interruption it can be started again only for the missing rels
            metrics = pd.merge(left=metrics_template, right=pd.DataFrame.from_dict(self.model_info, orient='index').T, how='left', left_index=True, right_index=True)
            metrics.sort_values(by='idx', ascending=True, inplace=True)
            metrics.drop(columns=['idx', 'metric_type'], inplace=True)
            metrics.to_csv(logit_timestamp_folder + 'detailed_results.tsv', sep='\t')



            i+=1

        #Building a single file calculating overall_metrics from all the single sfe/xke/XKEe runs
        metrics = pd.merge(left=metrics_template, right=pd.DataFrame.from_dict(self.model_info, orient='index').T, how='left', left_index=True, right_index=True)
        metrics.sort_values(by='idx', ascending=True, inplace=True)
        metrics.drop(columns=['idx', 'metric_type'], inplace=True)
        overall_metrics = pd.DataFrame(self.process_overall_metrics(metrics_template, metrics)).rename(columns={0:'{}'.format(logit_timestamp)})
        overall_metrics.to_csv(logit_timestamp_folder + 'overall_metrics.tsv', sep='\t')

        print('\nFinished XKE Pipeline!')

        os.system('spd-say "pipeline finished"')

        return
           
    def process_overall_metrics(self, metrics_template, metrics):

        overall = pd.Series(index=metrics_template.index)

        # descriptors
        col = metrics.iloc[:, 0]
        for el in metrics_template[metrics_template['metric_type'] == 'unique'].index:
            overall[el] = col[el]

        # metrics to be summed over each relation
        for metric in metrics_template[metrics_template['metric_type'] == 'sum'].index:
            overall.at[metric] = int(metrics.loc[metric].astype(float).sum())

        # micro avg metrics
        for metric in ['emb:rel_train_', 'emb:rel_test_', 'sfe:train_', 'sfe:test_', 'xke:fidelity_train_', 'xke:fidelity_test_', 'xke:accuracy_train_', 'xke:accuracy_test_', 'XKEe:fidelity_test_']:
            overall.at['{}acc'.format(metric)] = (overall.loc['{}tn'.format(metric)] + overall.loc['{}tp'.format(metric)]) /(overall.loc['{}tn'.format(metric)] + overall.loc['{}tp'.format(metric)] + overall.loc['{}fn'.format(metric)] + overall.loc['{}fp'.format(metric)])
            overall.rename({'{}acc'.format(metric):'{}acc (micro_avg)'.format(metric)}, inplace=True)
            overall.at['{}f1'.format(metric)] = overall.loc['{}tp'.format(metric)] / (overall.loc['{}tp'.format(metric)] + 0.5 * (overall.loc['{}fn'.format(metric)] + overall.loc['{}fp'.format(metric)]))
            overall.rename({'{}f1'.format(metric):'{}f1 (micro_avg)'.format(metric)}, inplace=True)
        
        # averaged metrics
        for metric in metrics_template[metrics_template['metric_type'] == 'avg'].index:
            overall[metric] = np.nanmean(metrics.loc[metric].astype(float).values)
        overall.rename({'sfe:test_AP':'sfe:test_MAP', 'xke:fidelity_test_AP':'xke:fidelity_test_MAP', 'xke:accuracy_test_AP':'xke:accuracy_test_MAP'}, inplace=True)

        # metrics that are weighted averages using train triples to weight
        for metric in metrics_template[metrics_template['metric_type'] == 'weighted_by_train'].index:
            try:
                overall[metric] = (metrics.loc[metric].astype(float).fillna(0).values @ metrics.loc['dat:train_triples_total'].astype(float).values) / metrics.loc['dat:train_triples_total'].astype(float).values.sum()
            except:
                overall[metric] = metrics.loc[metric][0]

        # metrics that are weighted averages using test triples to weight
        for metric in metrics_template[metrics_template['metric_type'] == 'weighted_by_test'].index:
            try:
                overall[metric] = (metrics.loc[metric].astype(float).fillna(0).values @ metrics.loc['dat:test_triples_total'].astype(float).values) / metrics.loc['dat:test_triples_total'].astype(float).values.sum()
            except:
                overall[metric] = metrics.loc[metric][0]

        return overall

    def train_rel_grid_search(self, grid_params): #Deprecated
        
        logit_timestamp = generate_timestamp()
        local_model_info = dict()
        metrics_template = pd.read_csv('metrics_template.tsv', sep='\t', index_col=0)
        logit_timestamp_folder = self.logit_results_folder + logit_timestamp + '/'

        if not os.path.exists(logit_timestamp_folder):
            os.makedirs(os.path.join(logit_timestamp_folder))
            print('Creating folder: {}'.format(logit_timestamp_folder))

        for param in grid_params.keys():

            self.set_prune_dict(grid_params[param])

            rel_id = grid_params[param]['rel']
            rel = self.names_dict[rel_id]
            
            grid_id = rel_id + '.' + param

            local_model_info[grid_id] = {**self.model_info[rel_id], **self.emb_metrics_info[rel_id]}
            self.metrics = dict() #clearing metrics dict for each relation
            self.logit_models[grid_id] = dict() #creating rel_id key for regression models

            X_train, y_train, y_train_emb, X_test, y_test, y_test_emb, X_train_pos, X_train_neg, X_test_pos, X_test_neg, feature_names, df_test = self.load_data(rel) 

            #Here we perform pure SFE, ie, fitting the regression into ground truth labels
            gs = GridSearchCV(SGDClassifier(), self.param_grid_logit, n_jobs=10, refit=True, cv=5, verbose=1)
            time.sleep(0.2)
            gs.fit(X_train, y_train)
            logit_model = gs.best_estimator_
            time.sleep(0.2)
            self.process_metrics(('sfe', ''), logit_model, df_test, X_train, y_train, X_test, y_test, X_train_pos, X_train_neg, X_test_pos, X_test_neg)

            #Here we perform XKE, ie, fitting the regression with embedding labels
            gs = GridSearchCV(SGDClassifier(), self.param_grid_logit, n_jobs=10, refit=True, cv=5, verbose=1)
            time.sleep(0.2)
            gs.fit(X_train, y_train_emb)
            xke_model = gs.best_estimator_
            time.sleep(0.2)
            self.process_metrics(('xke', 'fidelity_'), xke_model, df_test, X_train, y_train_emb, X_test, y_test_emb, X_train_pos, X_train_neg, X_test_pos, X_test_neg)
            self.process_metrics(('xke', 'accuracy_'), xke_model, df_test, X_train, y_train, X_test, y_test, X_train_pos, X_train_neg, X_test_pos, X_test_neg)

            #evaluating overall intepretability index
            self.compute_interpretability_index(rel_id, feature_names, xke_model.coef_, X_test)

            #Just a sanity-check with LazyClassifier
            if self.prune_dict.get('xke:evaluate_benchmarks', True):
                self.evaluate_lazyclassifier(X_train, y_train_emb, X_test, y_test_emb)

            local_model_info[grid_id] = {**local_model_info[grid_id], **self.metrics, **self.emb_metrics_info[rel_id]}

            #Now we store sfe and xke regression models as well as feature names for later evalutation
            self.logit_models[grid_id]['sfe'] = logit_model
            self.logit_models[grid_id]['xke'] = xke_model
            self.logit_models[grid_id]['feature_names'] = feature_names
            self.logit_models[grid_id]['df_test'] = df_test
            self.logit_models[grid_id]['y_test'] = y_test
            self.logit_models[grid_id]['y_test_emb'] = y_test_emb

            explain_model = pd.DataFrame()
            explain_model['path'] = feature_names
            explain_model['coef'] = xke_model.coef_[0]
            explain_model['avg_rel_sim'] = explain_model.path.apply(lambda x: self.compute_avg_rel_sim(rel_id, x))
            explain_model['path_names'] = explain_model.path.apply(lambda x: self.explain_path(x))
            explain_model.sort_values(by='coef', ascending=False, inplace=True)
            explain_model.reset_index(inplace=True)
            explain_model.drop(columns='index', inplace=True)
            explain_model.loc[-1] = ['bias', logit_model.intercept_[0], '', '']
            explain_model.index = explain_model.index + 1
            explain_model.sort_index(inplace=True)
            explain_model.to_csv(logit_timestamp_folder + '{}_{}_{}_{}_coefs.tsv'.format(self.timestamp_sfe, logit_timestamp, rel_id, param), sep='\t')
            # os.system('play -nq -t alsa synth .05 sine 880')
            os.system('spd-say "finished {}"'.format(rel_id))

        self.write_to_pkl(logit_timestamp_folder + '{}_{}_models'.format(self.timestamp_sfe, logit_timestamp), self.logit_models)

        metrics = pd.merge(left=metrics_template, right=pd.DataFrame(local_model_info), how='left', left_index=True, right_index=True)
        metrics.sort_values(by='idx', ascending=True, inplace=True)
        metrics.drop(columns='idx', inplace=True)
        metrics.to_csv(logit_timestamp_folder + 'metrics.tsv', sep='\t')

        print('\nFinished XKE Pipeline!')
        os.system('spd-say "pipeline finished"')



        return df_test, feature_names, X_test, xke_model

    def load_kbe(self):

        con = config.Config()
        self.emb = restore_model(con, self.embeddings_folder)

        # this is just a dummy operation to kickstart embedding model
        self.emb.init_triple_classification()
        self.emb.classify_triples([0, 2, 7], [1, 3, 5], [0, 1, 2])
        self.emb.classify_triples([0, 2, 7], [1, 3, 5], [0, 1, 2])

    def build_X_test_pred_full(self, xke_model, X_test, test_triples, feature_names, y_test_emb):

        print('\nStarting to build XKEe X_test set!\n')
        # self.logger.info('build_X_test_pred_full called')

        XKEe_X_test = X_test.tolil(copy=True)
        if len(feature_names) == 0: #for DummyClassifier we don't need to search for paths in the embedding
            return XKEe_X_test, 0, 0


        coefs = xke_model.coef_[0]
        y_pred_xke = xke_model.predict(X_test.todense())

        if not isinstance(self.graph, dict): self.build_graph()
        if not isinstance(self.g_hat, dict): self.load_g_hat()
        if not isinstance(self.g_hat_dict, dict): self.build_g_hat_dict()

        self.g_hat = dict()

        #Selecting only features with non_zero coefs
        features = pd.DataFrame(index=feature_names)
        features['coefs'] = coefs
        features.reset_index(inplace=True)
        features.rename(columns={'index':'path'}, inplace=True)
        features['idx'] = features.index
        features.set_index('path', inplace=True)
        features.sort_values(by='coefs', ascending=False, inplace=True)
        features = features[features['coefs'] != 0]

        false_positives_corrected = 0
        false_negatives_corrected = 0

        self.logger.info('Starting to loop over triples and features')

        time.sleep(0.2)

        for triple, j in tqdm(zip(test_triples, range(len(test_triples))), total=len(test_triples)):

            # self.logger.info('   ')

            check = 0

            #1st, we check if it is a correct prediciton, if True we skip this iteration
            if y_test_emb[j] == y_pred_xke[j]:

                # if y_test_emb[j] == 1:
                #     self.logger.info('Triple: %s is TP.', triple)
                # else:
                #     self.logger.info('Triple: %s is TN.', triple)

                continue

            else:

                if y_test_emb[j] > y_pred_xke[j]: #means that emb_pred is 1 and xke pred is 0
                    # self.logger.info('Triple: %s is FN.', triple) #From XKE point of view
                    FN = True
                else:
                    # self.logger.info('Triple: %s is FP.', triple)
                    FN = False

                #Let us gather a list of active features for the current triple that are not already
                #active, ie, the potential ones to be searched in g_hat

                #active features for current triple:
                x = XKEe_X_test[j].toarray()[0]

                #we will select features with active coeficients that were not found in the graph
                # if FN:
                #     mask = (x == 0) & (coefs > 0)
                # else:
                #     mask = (x == 0) & (coefs < 0)
                mask = (x == 0) & (coefs != 0)

                selected_features = list(np.array(feature_names)[mask])

                search_features = features[features.index.isin(selected_features)]

                f_features = list(search_features.index.values)
                f_idx = list(search_features.idx.values)

                #now we loop over all features
                while f_features:
                    
                    check += 1

                    this_feature = f_features.pop(0)
                    this_idx = f_idx.pop(0)

                    if self.path_builder(triple, this_feature):
                        XKEe_X_test[j, this_idx] = 1

                #here we check if XKEe was able to correct the feature after checking for all paths
                #in g_hat
                if y_test_emb[j] == xke_model.predict(XKEe_X_test[j]):
                    if FN == True:
                        false_negatives_corrected += 1
                    else:
                        false_positives_corrected += 1
                    # self.logger.info(f'Corrected prediction for triple: {triple} after checking {check} features.')
                else:
                    # if FN == True:
                    #     self.logger.info(f'Could not correct FN triple {triple}.')
                    # else:
                    #     self.logger.info(f'Could not correct FP triple {triple}.')
                    pass
            
        return XKEe_X_test, false_negatives_corrected, false_positives_corrected

    def expand_rel_nodes(self, nodes, rel):
        '''
        This method takes a list of nodes and a relation and expands 
        all the nodes removing entity descriptor.
        '''

        output = []

        for node in nodes:
            output += self.graph.get(node, dict()).get(rel, [])

        output = [int(x[1:]) for x in output]

        return output

    def get_ghat_tails(self, heads, rel):

        mask = np.zeros(shape=(len(self.ent_dict.keys())), dtype=np.bool)

        mask[list(heads)] = 1

        return np.nonzero(self.g_hat[rel][mask].sum(axis=0))[0].tolist()

    def get_ghat_heads(self, tails, rel):

        mask = np.zeros(shape=(len(self.ent_dict.keys())), dtype=np.bool)

        mask[list(tails)] = 1

        return np.nonzero(self.g_hat[rel][:, mask].sum(axis=1))[0].tolist()

    def build_emb_path(self, triple_in, path_in):

        if not isinstance(self.g_hat, dict):
            self.load_g_hat()

        # print(f'Expanding triple {triple_in} / path {path_in}')
        self.logger.info('========================================================')
        self.logger.info('Starting evaluation of triple {} / path {}'.format(triple_in, path_in)) 

        t0 = time.time()

        inv = lambda x: x[1:] if x[0] == 'i' else 'i' + x
        raw = lambda x: int(x[2:]) if x[0] == 'i' else int(x[1:])
        

        path = path_in.split('_')
        triple = triple_in.split('_')
        source, target = int(triple[0][1:]), int(triple[1][1:])

        #here we load the relation thresholds obtained from embeddings, whether from OpenKE or CV for
        #relations without valid/test subsets
        # thresh_dict = dict()

        # for p in path:
        #     if p[0] != 'i':
        #         thresh_dict[p] = self.emb_metrics_info[p]['emb:rel_threshold']
        #         if thresh_dict[p] == 0:
        #             thresh_dict[p] = self.emb_metrics_info[p]['emb:cv_rel_threshold']
        #     else:
        #         thresh_dict[p[1:]] = self.emb_metrics_info[p[1:]]['emb:rel_threshold']
        #         if thresh_dict[p[1:]] == 0:
        #             thresh_dict[p[1:]] = self.emb_metrics_info[p[1:]]['emb:cv_rel_threshold']

        # t1 = time.time()

        # self.logger.info(f'Build rel_thresholds dict in {t1 - t0:.4f}s.')

        #########################################

        l_int_nodes = []
        r_int_nodes = []
        left_frontier = []
        right_frontier = []

        #########################################
        #1st expansion for left side


        l_int_nodes += self.expand_rel_nodes([triple[0]], path[0])

        # t2 = time.time()

        # self.logger.info(f'g l_int_nodes {len(l_int_nodes)} expansion {t2-t1:.4f}s')

        if path[0][0] != 'i':
            # l_int_nodes += list(np.nonzero(self.emb.get_true_tails_np([source], raw(path[0]), thresh_dict[path[0]]))[0])
            l_int_nodes += self.get_ghat_tails([source], path[0])
        else:
            # l_int_nodes += list(np.nonzero(self.emb.get_true_heads_np([source], raw(path[0]), thresh_dict[path[0][1:]]))[0])
            l_int_nodes += self.get_ghat_heads([source], path[0][1:])

        # t3 = time.time()

        # self.logger.info(f'emb l_int_nodes {len(l_int_nodes)} expansion {t3-t2:.4f}s')

        r_int_nodes += self.expand_rel_nodes([triple[1]], inv(path[-1]))

        t4 = time.time()

        # self.logger.info(f'g r_int_nodes {len(r_int_nodes)} expansion {t4-t3:.4f}s')

        if path[-1][0] != 'i':
            # r_int_nodes += list(np.nonzero(self.emb.get_true_heads_np([target], raw(path[-1]), thresh_dict[path[-1]]))[0])
            r_int_nodes += self.get_ghat_heads([target], path[-1])
        else:
            # r_int_nodes += list(np.nonzero(self.emb.get_true_tails_np([target], raw(path[-1]), thresh_dict[path[-1][1:]]))[0])
            r_int_nodes += self.get_ghat_tails([target], path[-1][1:])

        t5 = time.time()

        # self.logger.info(f'emb r_int_nodes {len(r_int_nodes)} expansion {t5-t4:.4f}s')

        l_int_nodes = set(l_int_nodes)
        r_int_nodes = set(r_int_nodes)

        t6 =time.time()

        # self.logger.info(f'l_int_nodes {len(l_int_nodes)} / r_int_nodes {len(r_int_nodes)} set conversion {t6-t5:.4f}s')

        #########################################
        # If path is size 1 compare source with r_int_nodes or target with l_int_nodes

        if len(path) == 1:
            if (source in r_int_nodes) or (target in l_int_nodes):
                self.logger.info('True | p_len= 1 | source in r_int_nodes or target in l_int_nodes')
                return True
            else:
                self.logger.info('False | p_len=1 | source not in r_int_nodes or target not in l_int_nodes')
                return False

        # If path is size 2, compare l_int_nodes with r_int_nodes

        if len(path) == 2:
            if not set(l_int_nodes).isdisjoint(r_int_nodes):
                self.logger.info(f'True | p_len =2 | l_int_nodes {len(l_int_nodes)} or r_int_nodes {len(r_int_nodes)} have matching nodes.')
                return True

        #if a p_len =4, and one of the int stages is empty, no path can be connected
        if len(path) == 4:
            if (len(l_int_nodes) == 0) or (len(r_int_nodes) == 0):
                self.logger.info(f'False | p_len = 4 | l_int_nodes {len(l_int_nodes)} or r_int_nodes {len(r_int_nodes)} are empty for a path of length 4.')
                return False

        ########################################

        # let us now expand l_int_nodes

        # t7 = time.time()

        left_frontier += self.expand_rel_nodes(l_int_nodes, path[1])

        # t8 = time.time()

        # self.logger.info(f'g left_frontier {len(left_frontier)} expansion {t8-t7:.4f}s')

        # let us try to avoid having to use embeddings here
        if len(path) == 3:
            if not r_int_nodes.isdisjoint(left_frontier):
                return True
            # if r_int_nodes and left_frontier are disjoint we must expand right_frontier down the road

        if path[1][0] != 'i':
            # left_frontier += list(np.nonzero(self.emb.get_true_tails_np(l_int_nodes, raw(path[1]), thresh_dict[path[1]]))[0])
            left_frontier += self.get_ghat_tails(l_int_nodes, path[1])
        else:
            # left_frontier += list(np.nonzero(self.emb.get_true_heads_np(l_int_nodes, raw(path[1]), thresh_dict[path[1][1:]]))[0])
            left_frontier += self.get_ghat_heads(l_int_nodes, path[1][1:])

        t9 = time.time()
        # self.logger.info(f'emb left_frontier {len(left_frontier)} expansion {t9-t8:.4f}s')


        left_frontier = set(left_frontier)

        t10 = time.time()

        # self.logger.info(f'{len(left_frontier)} left_frontier set conversion {t10-t9:.4f}s')

        #at this point, if p_len=4 and left_frontier is empty, we can exit
        if len(path) == 4:
            if len(left_frontier) == 0:
                self.logger.info(f'False | p_len = 4 | exiting due to an empty left_frontier.')
                return False


        # we can try to find some paths of len=2 and 3 without expanding right_frontier
        if len(path) == 2:
            if target in left_frontier:
                self.logger.info(f'True | p_len = 2 | target in left_frontier, expanded emb_left_frontier')
                return True
                
        if len(path) == 3:
            if not r_int_nodes.isdisjoint(left_frontier):
                self.logger.info(f'True | p_len = 3 | r_int_nodes in left_frontier, expanded emb_left_frontier')
                return True

        t10 = time.time()

        # if r_int_nodes and left_frontier are disjoint we must expand right_frontier down the road

        right_frontier += self.expand_rel_nodes(r_int_nodes, inv(path[-2]))

        t11 = time.time()

        # self.logger.info(f'g r_int_nodes {len(r_int_nodes)} expansion {t11-t10:.4f}s')

        # now we check again to se wether there is a path without having to expand right_frontier through embeddings
        if len(path) == 2:
            if source in right_frontier:
                self.logger.info(f'True | p_len = 2 | source in right_frontier (g)')
                return True
        
        if len(path) == 3:
            if not l_int_nodes.isdisjoint(right_frontier):
                self.logger.info(f'True | p_len = 3 | l_int_nodes in right_frontier (g)')
                return True

        if len(path) == 4:
            if not set(right_frontier).isdisjoint(left_frontier):
                self.logger.info(f'True | p_len = 4 | right_frontier (only g) in left_frontier (g + g_hat)')
                return True

        t12 = time.time()
        # self.logger.info(f'l_int_nodes {len(l_int_nodes)} / right_frontier {len(right_frontier)} check {t12 - t11:.4f}s')

        #if we fail we need to expand right_frontier via embeddings

        if path[-2][0] != 'i':
            # right_frontier += list(np.nonzero(self.emb.get_true_heads_np(r_int_nodes, raw(path[-2]), thresh_dict[path[-2]]))[0])
            right_frontier += self.get_ghat_heads(r_int_nodes, path[-2])
        else:
            # right_frontier += list(np.nonzero(self.emb.get_true_tails_np(r_int_nodes, raw(path[-2]), thresh_dict[path[-2][1:]]))[0])
            right_frontier += self.get_ghat_tails(r_int_nodes, path[-2][1:])

        t13 = time.time()
        # self.logger.info(f'{len(r_int_nodes)} => {len(right_frontier)} nodes from emb right_frontier expansion {t13-t12:.4f}s')

        right_frontier = set(right_frontier)

        t14 = time.time()
        # self.logger.info(f'right_frontier {len(right_frontier)} set conversion {t14-t13:.4f}s')

        if len(path) == 2:
            if source in right_frontier:
                self.logger.info(f'True | p_len = 2 | source in right_frontier | full expansion.')
                return True
            if target in left_frontier:
                self.logger.info(f'True | p_len = 2 | target in left_frontier | full expansion.')
                return True

        if len(path) == 3:
            if not l_int_nodes.isdisjoint(right_frontier):
                self.logger.info(f'True | p_len = 3 | l_int_nodes in right_frontier | full expansion.')
                return True

        if len(path) == 4:
            if len(right_frontier) <= len(left_frontier):
                if not right_frontier.isdisjoint(left_frontier):
                    self.logger.info(f'True | p_len = 4, right_frontier / left_frontier | full expansion.')
                    return True
            else:
                if not left_frontier.isdisjoint(right_frontier):
                    self.logger.info(f'True | p_len = 4, left_frontier / right_frontier | full expansion.')
                    return True

        self.logger.info(f'False | Reached the end of the pipeline!')

        return False

    def path_builder(self, triple_in, path_in):

        
        # print(f'Expanding triple {triple_in} / path {path_in}')
        # self.logger.info('========================================================')
        # self.logger.info('Starting evaluation of triple {} / path {}'.format(triple_in, path_in)) 

        t0 = time.time()
        max_iter = 1000

        inv = lambda x: x[1:] if x[0] == 'i' else 'i' + x
        raw = lambda x: int(x[2:]) if x[0] == 'i' else int(x[1:])
        
        path = path_in.split('_')
        triple = triple_in.split('_')
        source, target = int(triple[0][1:]), int(triple[1][1:])

        if len(path) == 4:

            l_int_nodes = self.graph.get(source, dict()).get(path[0], [])
            l_int_nodes += self.g_hat_dict[path[0]][source]
            l_int_nodes = deque(set(l_int_nodes))

            r_int_nodes = self.graph.get(target, dict()).get(inv(path[3]), [])
            r_int_nodes += self.g_hat_dict[inv(path[3])][target]
            r_int_nodes = deque(set(r_int_nodes))

            left_frontier = []
            right_frontier = []

            # self.logger.info(f'l_int_nodes {len(l_int_nodes)} | r_int_nodes {len(r_int_nodes)}')

            if (len(l_int_nodes) == 0) or (len(r_int_nodes) == 0):
                # self.logger.info(f'False | p_len = 4 | empty l_int_nodes or r_int_nodes')
                return False 

            k = 0
            while l_int_nodes or r_int_nodes:

                try:
                    left = l_int_nodes.pop()
                    left_frontier += self.graph.get(left, dict()).get(path[1], [])
                    left_frontier += self.g_hat_dict[path[1]].get(left, [])
                except:
                    pass

                try:
                    right = r_int_nodes.pop()
                    right_frontier += self.g_hat_dict[inv(path[2])][right]
                    right_frontier += self.g_hat_dict[inv(path[2])].get(right, [])
                except:
                    pass

                k += 1
                if k == max_iter: break

            if len(left_frontier) > len(right_frontier):

                if not set(right_frontier).isdisjoint(left_frontier):
                    # self.logger.info(f'True | p_len = 4 | found path after {k} iterations.')
                    return True

            else:
                if not set(left_frontier).isdisjoint(right_frontier):
                    # self.logger.info(f'True | p_len = 4 | found path after {k} iterations.')
                    return True
                # if k == max_iter:
                #     # self.logger.info(f'False | p_len = 4 | Reached max_iter: {k}.')
                #     return False

            # self.logger.info(f'False| p_len = 4 | found no path after {k} iterations.')

            return False

        if len(path) == 3:
            
            l_int_nodes = self.graph.get(source, dict()).get(path[0], [])
            l_int_nodes += self.g_hat_dict[path[0]][source]
            l_int_nodes = deque(set(l_int_nodes))

            r_int_nodes = self.graph.get(target, dict()).get(inv(path[2]), [])
            r_int_nodes += self.g_hat_dict[inv(path[2])][target]
            r_int_nodes = deque(set(r_int_nodes))

            left_frontier = []

            # self.logger.info(f'l_int_nodes {len(l_int_nodes)} | r_int_nodes {len(r_int_nodes)}')

            if (len(l_int_nodes) == 0) or (len(r_int_nodes) == 0):
                # self.logger.info(f'False | p_len = 3 | empty l_int_nodes or r_int_nodes')
                return False 

            k = 0
            while l_int_nodes:

                left = l_int_nodes.pop()
                left_frontier += self.g_hat_dict[path[1]].get(left, [])

                k += 1
                if k == max_iter: break

            if len(left_frontier) > len(r_int_nodes):

                if not set(r_int_nodes).isdisjoint(left_frontier):
                    # self.logger.info(f'True | p_len = 4 | found path after {k} iterations.')
                    return True

            else:
                if not set(left_frontier).isdisjoint(r_int_nodes):
                    # self.logger.info(f'True | p_len = 4 | found path after {k} iterations.')
                    return True

                # if k == max_iter:
                #     # self.logger.info(f'False | p_len = 3 | Reached max_iter: {k}.')
                #     return False

            # self.logger.info(f'False| p_len = 3 | found no path after {k} iterations.')

            return False

        if len(path) == 2:

            l_int_nodes = self.graph.get(source, dict()).get(path[0], [])
            l_int_nodes += self.g_hat_dict[path[0]][source]
            l_int_nodes = set(l_int_nodes)

            r_int_nodes = self.graph.get(target, dict()).get(inv(path[1]), [])
            r_int_nodes += self.g_hat_dict[inv(path[1])][target]
            r_int_nodes = set(r_int_nodes)

            # self.logger.info(f'l_int_nodes {len(l_int_nodes)} | r_int_nodes {len(r_int_nodes)}')

            if not set(l_int_nodes).isdisjoint(r_int_nodes):
                # self.logger.info(f'True | p_len = 2 | l_int_nodes and r_int_nodes with intersections.')
                return True
            else:
                # self.logger.info(f'False | p_len = 2 | l_int_nodes and r_int_nodes with no intersections.')
                
                return False

        if len(path) == 1:

            l_int_nodes = self.graph.get(source, dict()).get(path[0], [])
            l_int_nodes += self.g_hat_dict[path[0]][source]
            l_int_nodes = set(l_int_nodes)

            if target in l_int_nodes:
                # self.logger.info(f'True | p_len = 1 | target is in l_int_nodes')
                return True
            else:
                # self.logger.info(f'False | p_len = 1 | target is not in l_int_nodes')

                return False

        return False


class PriorClassifier(object):
    """Returns a dummy classifier that is used when no features are present in training/validation.
    """
    def __init__(self):
        self.clf = DummyClassifier(strategy='prior')

    def fit(self, X, y):
        self.clf.fit(self._adapt_X(X), y)
        # self.coef_ = np.array([[0]])
        self.coef_ = np.array([])
        proba = self.clf.class_prior_[-1] # get probability of predicting 1, which will be always the same regardless of X
        self.intercept_ = math.log(proba/(1-proba)) # apply logit to probability in order to get the intercept

    def _adapt_X(self, X):
        if X.shape[-1] == 0:
            X = np.zeros(X.shape[:-1] + (1,))
        return X

    def predict(self, X):
        return self.clf.predict(self._adapt_X(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._adapt_X(X))

    def score(self, X, y, sample_weight=None):
        return self.clf.score(self._adapt_X(X), y, sample_weight)