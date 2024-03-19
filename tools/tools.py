import datetime
import os
import math
import pandas as pd
import numpy as np
import random
import csv
import json
import logging
import _pickle as pickle
import multiprocessing
from scipy import sparse
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier, LinearRegression, ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.dummy import DummyClassifier


def generate_timestamp():
  return datetime.datetime.now().strftime('%y%m%d%H%M')

def write_to_pkl(filenm, object_to_save):

    file_name = '{}.pkl'.format(filenm)
    # print('Writing to Dict file '.format(filenm))

    with open(file_name, 'wb') as f:
        pickle.dump(object_to_save, f)

def load_pkl(filenm):

    file_name = '{}.pkl'.format(filenm)
    with open(file_name, "rb") as f:
        object_to_load = pickle.load(f)
    
    return object_to_load

def load_file(filenm):

    with open(filenm, 'rb') as f:
        object_to_load = pickle.load(f)

    return object_to_load

def read_model_info(folder):
    '''This function just gets an embedding directory and retrieves the model info
    file. It returns a dict containg overall information about the model 
    obtained during training phase'''

    md_info = pd.read_csv(folder+'model_info.tsv', sep='\t')
    model_info = md_info.to_dict('record')
    
    return model_info[0]

def regression_metrics(y_pred, y_true, p=False):

    n = len(y_pred)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    rss = ((y_pred - y_true) ** 2).sum()
    tss = ((y_true - y_true.mean()) ** 2).sum()
    mse = ((y_pred - y_true) ** 2).sum() / n
    
    if p:
        rse = np.sqrt( (1 / (n-p-1)) * rss)
        adjusted_r2 = 1 - ((rss / (n-p-1) ) / (tss / (n-1)))
    else:
        rse = np.sqrt(((1 / (n-2)) * rss))
        adjusted_r2 = 0

    r2 = 1 - rss/tss

    out_dict = {
        'mse' : mse,
        'rss' : rss,
        'tss' : tss,
        'rse' : rse, #only make sense to training set 
        'r2'  : r2, #theoretically also makes sense only to training set, function of p
        'adjusted_r2' : adjusted_r2
    }

    return out_dict

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.join(file_path))
        print('Creating folder: {}.'.format(file_path))

def save_model_info(model_info, export_path):
    
  results = pd.DataFrame([model_info])

  #store embedding results within the timestamp folder
  results.to_csv('{}/model_info.tsv'.format(export_path), sep='\t', index=False)
    
  #store historic of all embedding runs
  file_to_save = os.path.expanduser('~') + '/proj/OpenKE/logs/model_info_history.csv'
  if not os.path.isfile(file_to_save):
    print('creating file')
    results.to_csv(file_to_save, sep='\t', index=False)
  else:
    print('appending results to existing file')
    df = pd.read_csv(file_to_save, sep='\t')
    df = df.append(results, ignore_index=True)
    df.to_csv(file_to_save, sep='\t', index=False)

def save_training_log(training_log, export_path):
  
  results = pd.DataFrame([training_log])

  #store embedding results within the timestamp folder
  results.to_csv('{}/training_log.csv'.format(export_path), index=False)

def get_mid2name(dict_path, map_path='./mid2name.tsv'):
  """ Decode Machine Id (mid) to its correspondent name. 
      It is needed to interpret Freebase entities.
  """
    
  # Load dictionary for dataset and mid to name mapping
  entity2id_df = pd.read_csv(dict_path, sep='	', names=['mid', 'id'], skiprows=[0])
  mid2name_df = pd.read_csv('./mid2name.tsv', sep='	', names=['mid', 'name'])

  # Filter only the intersection of entity2id and mid2name to reduce computation
  mid2name_df = mid2name_df.loc[mid2name_df['mid'].isin(entity2id_df['mid'])]

  # Group multiple names for same mid (mid2name_df is now a dictionary)
  mid2name_sf = mid2name_df.groupby('mid').apply(lambda x: "%s" % '| '.join(x['name']))
  mid2name_df = pd.DataFrame({'mid':mid2name_sf.index, 'name':mid2name_sf.values})

  return pd.merge(entity2id_df, mid2name_df, how='left', on=['mid'])

def read_type_constrain_file(filepath):
    """Parses a type_constrain file into a dict. Each key in the dict is a relation and each value
    another dict. This second dict has two keys: head and tail. Each value of this second dict is a
    set of entities that were observed in that position (head or tail) of the graph for the
    respective relation.
    """
    with open(filepath, 'r') as f:
        content = f.read()
        content = content.split('\n')[:-1]
        del content[0] # remove the first line
        content = [line.split('\t') for line in content]
        content = [[int(item) for item in line] for line in content]
        output = {}
        last_rel = None
        for line in content:
            rel = line[0]
            entities = line[2:]
            if not rel == last_rel:
                output[rel] = {'head': set(entities)}
            else:
                output[rel]['tail'] = set(entities)
            last_rel = rel
    return output

def get_bern_prob_corrupt_tail(type_constrain_dict):
    """Gets the probability of the tail be corrupted when generating negative examples using the
    Bernoulli distribution proposed by Wang et al. (2014).

    Arguments:
    - type_constrain_dict: A dictionary where each key in the dict is a relation and each value
    another dict. This second dict has two keys: head and tail. Each value of this second dict is a
    set of entities that were observed in that position (head or tail) of the graph for the
    respective relation. See `read_type_constrain_file()` function.
    """
    relations = range(len(type_constrain_dict))
    prob_corrupt_tail = {}
    for r in relations:
        tph = float(len(type_constrain_dict[r]['tail'])) / len(type_constrain_dict[r]['head'])
        hpt = tph**(-1) # head_per_tail is the inverse of tail_per_head
        prob_corrupt_tail[r] = hpt / (hpt + tph)
    return prob_corrupt_tail

def generate_corrupted_training_examples(dataset_path, neg_proportion=1, bern=True,
                                         output_include_pos=True):
    """Generates negative examples for training following the Bernoulli sampling procedure proposed
    by Wang el al. (2014) if `bern=True` or using a uniform distribution if `bern=False`. A list is
    returned, where each element is a dict representing a triple (with keys = head, tail and
    relation).

    Arguments:
    - dataset_path: path of the dataset for which positive training examples will be corrupted
    - neg_proportion: proportion of negative examples for each positive one
    - bern: flag indicating that the bernoulli distribution will be used to corrupt head vs tail
    """
    train_triples = pd.read_csv(dataset_path + '/train2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    tc_dict = read_type_constrain_file(dataset_path + '/type_constrain.txt')
    # get a set of all entities present in test data
    ents = set()
    ents.update(train_triples['head'].unique())
    ents.update(train_triples['tail'].unique())

    if bern:
        prob_corrupt_tail = get_bern_prob_corrupt_tail(tc_dict)
    else:
        relations = train_triples['relation'].unique()
        prob_corrupt_tail = {rel: 0.5 for rel in relations}

    output = []
    for idx, row in train_triples.iterrows():
        r = row['relation']
        h = row['head']
        t = row['tail']
        if output_include_pos:
            output.append({'head': h,
                           'tail': t,
                           'relation': r,
                           'label': 1})
        for _ in range(neg_proportion):
            if prob_corrupt_tail[r] < random.random():
                t_ = random.sample(ents, 1)[0]
                h_ = h
            else:
                h_ = random.sample(ents, 1)[0]
                t_ = t
            output.append({'head': h_,
                           'tail': t_,
                           'relation': r,
                           'label': -1})
    return output

def restore_model(con, model_info_path):

    model_info_df = pd.read_csv('{}model_info.tsv'.format(model_info_path), sep='\t')
    # transform model info into dict with only one "row"

    model_info = model_info_df.to_dict()
    for key,d in model_info.items():
        model_info[key] = d[0]
    # add timestamp to model if not present
    if not 'timestamp' in model_info:
        model_info['timestamp'] = os.path.abspath(model_info_path).split(os.sep)[-1]

    #Input training files from benchmarks/FB15K/ folder.
    dataset_path = './benchmarks/{}/'.format(model_info['dataset_name'])
    if not os.path.exists(dataset_path):
        raise ValueError('ERROR: Informed dataset path `{}` could not be found.'.format(dataset_path))

    con.set_in_path(dataset_path)

    # Set Parameters
    con.set_test_link_prediction(model_info['test_link_prediction'])
    con.set_test_triple_classification(model_info['test_triple_classification'])

    con.set_work_threads(model_info['work_threads'])
    con.set_train_times(model_info['train_times'])
    con.set_nbatches(model_info['nbatches'])
    if 'alpha' in model_info: con.set_alpha(model_info['alpha'])
    if 'margin' in model_info: con.set_margin(model_info['margin'])
    con.set_bern(model_info['bern'])
    con.set_dimension(model_info['dimension'])
    con.set_ent_neg_rate(model_info['ent_neg_rate'])
    con.set_rel_neg_rate(model_info['rel_neg_rate'])
    con.set_opt_method(model_info['opt_method'])

    #Models will be exported via tf.Saver() automatically.
    emb_model_path = model_info_path + 'tf_model/'
    con.set_import_files(emb_model_path + 'model.vec.tf')
    #Initialize experimental settings.
    con.init()
    
    print('emb_model_path', emb_model_path)
    print('set_import_files', emb_model_path + 'model.vec.tf')

    #Set the knowledge embedding model
    models_path = './models/'
    models_names = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
    sel_model_name = '{}.py'.format(model_info['model']) 
    if not any(sel_model_name in model_name for model_name in models_names):
        raise ValueError('ERROR: Informed model `{}` is unkown'.format(model_info['model']))
    _model = getattr(__import__('models'), model_info['model'])
    con.set_model(_model)

    return con

def load_from_pkl(filenm):

    file_name = '{}.pkl'.format(filenm)
    with open(file_name, "rb") as f:
        obj = pickle.load(f)
    
    return obj

def build_anyrel_features(features_pra):

    path_types = list(features_pra.keys())
    features_anyrel = dict()
    for path in path_types:
        current = path.split('_')
        lenght = len(current)
        qty = features_pra[path]
        if lenght == 1:
            features_anyrel['ANR' + path] = qty
        if lenght == 2:
            new_path = 'ANR' + current[0] + '_any'
            existing_qty = features_anyrel.get(new_path, 0)
            features_anyrel[new_path] = existing_qty + qty
            new_path = 'ANRany_' + current[1]
            existing_qty = features_anyrel.get(new_path, 0)
            features_anyrel[new_path] = existing_qty + qty
        if lenght == 3:
            new_path = 'ANR' + current[0] + '_any_any'
            existing_qty = features_anyrel.get(new_path, 0)
            features_anyrel[new_path] = existing_qty + qty
            new_path = 'ANRany_any_' + current[2]
            existing_qty = features_anyrel.get(new_path, 0)
            features_anyrel[new_path] = existing_qty + qty
            # new_path = current[0] + '_any_' + current[2]
            # existing_qty = features_anyrel.get(new_path, 0)
            # features_anyrel[new_path] = existing_qty + qty
        if lenght == 4:
            new_path = 'ANR' + current[0] + '_any_any_any'
            existing_qty = features_anyrel.get(new_path, 0)
            features_anyrel[new_path] = existing_qty + qty
            new_path = 'ANRany_any_any_' + current[3]
            existing_qty = features_anyrel.get(new_path, 0)
            features_anyrel[new_path] = existing_qty + qty
            # new_path = current[0] + '_any_any_' + current[3]
            # existing_qty = features_anyrel.get(new_path, 0)
            # features_anyrel[new_path] = existing_qty + qty

    return features_anyrel

def prune_features(features, max_len, feature_type=False):
    '''This function gets a dict with paths (features) and prunes if path lenght
    is greather than max_len. It returns a dict in the same way'''

    paths = list(features.keys())
    output_dict = dict()

    if feature_type:
        for path in paths:
            if len(path.split('_')) > max_len:
                continue
            else:
                output_dict[path] = features[path]
                # output_dict[feature_type + path] = 1.0

    else:
        for path in paths:
            if len(path.split('_')) > max_len:
                continue
            else:
                output_dict[path] = features[path]
                # output_dict[path] = 1.0


            
    return output_dict

def load_splits(splits_path):
    'This function gets a splits folder and loads the corrupted triples'

    train_fpath = splits_path + 'train.tsv'
    valid_fpath = splits_path + 'valid.tsv'
    test_fpath  = splits_path + 'test.tsv'

    train_set = pd.read_csv(train_fpath, sep='\t')
    valid_set = pd.read_csv(valid_fpath, sep='\t')
    test_set = pd.read_csv(test_fpath, sep='\t')

    return list(train_set.triple_id), list(valid_set.triple_id), list(test_set.triple_id)

def select_features(combined_feature_path, triples, p_features, criteria):

    feature_set = pd.read_pickle(combined_feature_path + 'combined_features.fset')
    feature_set = feature_set[feature_set['triple_id'].isin(triples)]

    if criteria == 'popularity':
        criteria = 'triple_id'

    summary = feature_set.pivot_table(['rwp', 'rwp_total', 'p_len', 'counts', 'triple_id'], index='path', aggfunc={'rwp':'mean','rwp_total':'mean', 'p_len':'mean', 'counts':'sum', 'triple_id':'count'}).reset_index()
    summary.sort_values(by=[criteria], ascending = False, inplace=True)

    return list(summary.head(p_features)['path'].values)

def fit_selector(feature_list):
    'This features takes as input a list of features, and returns a DictVectorizer object'

    dummy_dict = dict(zip(feature_list, [1] * len(feature_list)))

    selector = DictVectorizer(sparse = True)
    selector.fit([dummy_dict])

    return selector


def import_features(fpath, feature_paths, feature_types = {'PRA': True, 'DSP':False}, k = False):
    '''This function buils da dataset according to splits file (from benchmark) and features
    extracted
    '''

    pra = feature_types['PRA']
    dsp = feature_types['DSP']

    # to import features during fitting phase we build triple list from each dataset (train, test, valid)
    if type(fpath) == str:
        df = pd.read_csv(fpath, sep='\t')
        if k:
            df = df.head(k)
        triples_to_import = list(df['triple_id'])
        heads = list(df['e1'])
        tails = list(df['e2'])
    
    # during prediciton phase, the feature set is build from a dict:
    if type(fpath) == dict:
        if k:
            triples_to_import = fpath['triple_id'][:k]
            heads = fpath['e1'][:k]
            tails = fpath['e2'][:k]
        else:
            triples_to_import = fpath['triple_id']
            heads = fpath['e1']
            tails = fpath['e2']

    labels = []
    feature_dicts = []

    #Not allowing to run without PRA features:
    if dsp:
    
        for triple in triples_to_import:
            features = load_from_pkl(feature_paths + triple)
            features_pra = features['PRA']
            features_anyrel = build_anyrel_features(features_pra)
            features_pra = prune_features(features_pra, 3, 'PRA')
            
            
            # combined_features = {**features_pra, **features['DSP'], **features_anyrel}
            feature_dicts.append(combined_features)

    else:
        for triple in triples_to_import:
            features = pd.read_pickle(feature_paths + triple + '.fset')
            if features.shape[0] == 0:
                feature_dicts.append({})
            else:
                feature_dicts.append(dict(zip(features['path'], features['counts'])))
                feature_dicts.append(dict(zip(features['path'], [1] * len(features['path']))))


    feature_set = {'heads': heads,
                    'tails': tails,
                    'labels':labels,
                    'features': feature_dicts}

    return feature_set

def forward_selection(train_x, train_y, max_features):
    '''This functions receives a pandas Dataframe as features and a vector with response
    and performs a OLS regression with forward selection, returning the choosen model
    '''

    remaining = set(train_x.columns)
    if len(remaining) != len(train_x.columns):
        print('misaligned feature matrix!')
    selected = []

    #adding response to the dataframe
    train_x['y_score'] = train_y

    current_score, best_score = 0.0, 0.0

    while len(selected) <= max_features:

        current_step_scores = []

        for candidate in remaining:
            regression = '{} ~ {} + 1'.format('y_score', ' + '.join(selected + [candidate]))
            current_score = sm.OLS(regression, train_x).fit().rsqared_adj
            current_step_scores.append((current_score, candidate))
        current_step_scores.sort()
        best_new_score, best_candidate = current_step_scores.pop()
        print('Score {}'.format(best_new_score))
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    
    #fitting the best model again:
    regression = '{} ~ {} + 1'.format('y_score', ' + '.join(selected))
    model = smf.ols(formula, data).fit()

    return model   

def load_data(splits_path, features_path, target_relation, embd_model, rel_id, feature_types, selector):
    """Read embedding predicted data for the target relation from the split (i.e., from data
    whose features were extracted and whose labels are predictions from the embedding model).

    -------------------------------------
    WARNING: this function should be called whenever we change from one relation to another, so
    that it changes the `target_relation` and get the split data for the new relation.
    -------------------------------------

    Data is stored into the following lists or numpy ndarrays:

    - train_heads: contains the head for each training example
    - train_tails: contains the tail for each training example
    - train_y: contains the label (by the embedding model) for each training example
    - train_x: contains the extracted features (by PRA/SFE) for each training example

    - test_heads: contains the head for each testing example
    - test_tails: contains the tail for each testing example
    - test_y: contains the label (by the embedding model) for each testing example
    - test_x: contains the extracted features (by PRA/SFE) for each testing example

    If there is no test data, then we do not even train an explanation to begin with, and
    `False` is returned. If everything went alright, then `True` is returned.
    """
    train_fpath = splits_path + 'train.tsv'
    valid_fpath = splits_path + 'valid.tsv'
    test_fpath  = splits_path + 'test.tsv'

    # check if `test.npz` and `train.npz` are present
    if not os.path.exists(test_fpath) or os.stat(test_fpath).st_size == 0:
        print("There is no test file for relation `{}`, skipping.".format(target_relation))
        return False
    if not os.path.exists(train_fpath) or os.stat(train_fpath).st_size == 0:

        # raise IOError('`train.npz` not present for relation `{}`'.format(target_relation))
        print("There is no train data for relation `{}`, skipping.".format(target_relation))
        return False

    # read train data (always present - not entirely true for NELL)
    train = import_features(train_fpath, features_path, feature_types=feature_types, k=False)
    train_heads = train['heads']
    train_tails = train['tails']
    train_feat_dicts = train['features']
    train_y, train_y_score = embd_model.classify_triples(train_heads, train_tails, [rel_id] * len(train_tails))
    print('Feature Selector loaded with {} features'.format(len(selector.get_feature_names())))

    # read valid data (may not be present)
    if os.path.exists(valid_fpath) and os.stat(valid_fpath).st_size != 0:
        valid = import_features(valid_fpath, features_path, feature_types=feature_types, k=False)
        valid_heads = valid['heads']
        valid_tails = valid['tails']
        valid_feat_dicts = valid['features']
        valid_y, valid_y_score = embd_model.classify_triples(valid_heads, valid_tails, [rel_id] * len(valid_tails))
        train_x = selector.transform(train_feat_dicts)
        valid_x = selector.transform(valid_feat_dicts)
        # we merge validation with training data, because the GridSearchCV creates the valid split automatically
        train_heads = np.concatenate((train_heads, valid_heads))
        train_tails = np.concatenate((train_tails, valid_tails))
        train_y     = np.concatenate((train_y,     valid_y    ))
        train_y_score = np.concatenate((train_y_score, valid_y_score))
        train_x     = vstack((train_x, valid_x)) # concatenate the sparse matrices vertically
        assert(train_y.shape[0] == train_x.shape[0])
    else:
        train_x = selector.transform(train_feat_dicts)

    # read test data (always present)
    test = import_features(test_fpath, features_path, feature_types=feature_types, k=False)
    test_heads = test['heads']
    test_tails = test['tails']
    test_feat_dicts = test['features']
    test_y, test_y_score = embd_model.classify_triples(test_heads, test_tails, [rel_id] * len(test_tails))
    test_x = selector.transform(test_feat_dicts) 

    return train_x, train_y, train_y_score, test_x, test_feat_dicts, test_y, test_y_score

def load_lp_data(lp_path, triple, features_path, feature_types, rel_id, embd_model):
    """Read embedding predicted data for the target relation from the split (i.e., from data
    whose features were extracted and whose labels are predictions from the embedding model).

    
    """
    lp_fpath = lp_path + '{}.tsv'.format(triple)

    # read train data (always present - not entirely true for NELL)
    train = import_features(lp_fpath, features_path, feature_types=feature_types, k=False)
    train_heads = train['heads']
    train_tails = train['tails']
    train_feat_dicts = train['features']
    train_y, train_y_score = embd_model.classify_triples(train_heads, train_tails, [rel_id] * len(train_tails))
   
    return train_heads, train_tails, train_feat_dicts, train_y, train_y_score

def load_raw_feature_matrix(filepath):

    logger = logging.getLogger(__name__)
    logger.info('Parsing file `{}`'.format(filepath))

    with open(filepath, "rb") as input_file:
        feature_set = pickle.load(input_file)

    return feature_set['heads'], feature_set['tails'], feature_set['features']


def translate_feature_paths():
    pass

def train_global_logit(param_grid_logit, train_x, train_y):
    """Trains a logistic regression model globally for the current relation.
    """
    # check that there is at least one feature for the training set, otherwise it's not possible to fit the GS
    if train_x.shape[-1] == 0:
        print('Running PriorClassifier!')
        model = PriorClassifier()
        model.fit(train_x, train_y)
    else:
        # OBS: There is a bug when running `GridSearchCV` in vscode debugger
        gs = GridSearchCV(SGDClassifier(), param_grid_logit, n_jobs=multiprocessing.cpu_count(), refit=True, cv=8, verbose=10)
        gs.fit(train_x, train_y)
        model = gs.best_estimator_

    return model

def explain_model(model, feature_names, target_relation, top_n=10, output_path=None):
    """Explain the model using the coefficients (weights) that the linear/logistic regression
    associated to each feature. The explanations are stored in `self.explanation`, a pandas
    DataFrame with `feature` and `weight` columns, sorted from highest to lowest weight.

    If `output_path` is provided, then the method will export the DataFrame as tsv to a folder
    with the same name of the current model in `output_path`.
    """
    # feature_names = ['bias'] + feature_names

    explanation = pd.DataFrame({
        'feature': ['bias'] + feature_names,
        'weight': np.insert(model.coef_.reshape(-1), 0, model.intercept_) # extract coefficients (weights)
    }).sort_values(by="weight", ascending=False)

    # remove features whose weight is zero
    filtered = explanation[explanation['weight'] != 0]
    # get the top_n relevant features (for both positive and negative weights)
    top_n_relevant_features = pd.concat([filtered.iloc[0:top_n], filtered.iloc[-top_n:-1]])

    # save explanation if `output_path` provided
    if output_path:
        output_dir      = os.path.join(output_path)#, 'global_logit')
        output_filepath = os.path.join(output_dir,  '{}.tsv'.format(target_relation))
        ensure_dir(output_dir)
        explanation.to_csv(output_filepath, sep='\t', columns=['weight', 'feature'], index=False)
    
    return top_n_relevant_features

def get_dirs(dirpath):
    """Same as `os.listdir()` but ensures that only directories will be returned.
    """
    #fs = init_fs(True)
    dirs = []
    for f in os.listdir(dirpath):
        f_path = os.path.join(dirpath, f)
        if os.path.isdir(f_path):
            dirs.append(f)
    return dirs

def trim_relation_name(relation_name):
    """ just a simple function to remove forward slashes from relation names
        it trims the first forward slash and replaces the others
    """

    if relation_name[0] == '/':
        relation_name = relation_name[1:]
        
    return relation_name.replace('/', '-')
