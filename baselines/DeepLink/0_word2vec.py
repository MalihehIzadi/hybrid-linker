from gensim.models import word2vec, KeyedVectors, Word2Vec
import pandas as pd
import itertools
import os
import gc

corpus_df = pd.DataFrame()
diff_corpus_df = pd.DataFrame()


def add_summary_description_message_to_corpus(corpus_df, path, files): 
    x=0
    for file in files:
        if file.split('.')[1] == 'parquet' :
            df = pd.read_parquet(path+file)
        else:
            df = pd.read_pickle(path+file)
        x += (df.shape[0]*3)
        tmp = pd.concat([df['description_processed'], df['summary_processed'], df['message_processed']])
        corpus_df = pd.concat([corpus_df, tmp.to_frame()])
        del df
        gc.collect()
    print(x)
    return corpus_df


def add_diff_to_corpus(diff_corpus_df, path, files):
    c = 0
    for file in files:
        if file.split('.')[1] == 'parquet' :
            diff = pd.read_parquet(path+file)
        else:
            diff = pd.read_pickle(path+file)
            
        diff['processDiffCode'].loc[diff['processDiffCode'].isnull()] = diff['processDiffCode'].loc[diff['processDiffCode'].isnull()].apply(lambda x: [])
        diff.loc[:, 'tmp'] = diff['processDiffCode'].apply(lambda x: list(itertools.chain(*x))) # inorder to flatten the lists of processDiffCode column
        c += (diff.shape[0]*2)
        tmp = pd.concat([diff['changed_files'], diff['tmp']])
        diff_corpus_df = pd.concat([diff_corpus_df, tmp.to_frame()])
        del diff
        gc.collect()
    print(c)
    return diff_corpus_df


def preprocess_on_corpus(data_frame): 
    data_frame = data_frame.reset_index(drop=True)
    data_frame.rename(columns = {0:'data'}, inplace = True) 
    data_frame['data'].loc[data_frame['data'].isnull()] = data_frame['data'].loc[data_frame['data'].isnull()].apply(lambda x: [])
    data_frame = data_frame[~data_frame.data.str.len().eq(0)]
    data_frame.loc[:,'data'] = data_frame['data'].apply(lambda x: x.tolist() if not isinstance(x, list) else x)
    return data_frame


def files_in_directory(path):
    files = os.listdir(path)
    try:
        files.remove('.ipynb_checkpoints')
    except:
        print("ipynb_checkpoints doesn't exits")
    return files


def run_w2v(data_frame, file_name_to_save):
#     corpus_model = word2vec.Word2Vec(data_frame['data'], min_count=1, size= 100, workers=20, sg = 1, hs=1, negative=10, iter=50)
    corpus_model = word2vec.Word2Vec(data_frame['data'], size=100, sg=1, hs=1, workers=20, iter=50)

    corpus_model.wv.save_word2vec_format(file_name_to_save, binary=True)
    print(corpus_model)
    
    
def create_w2v_without_diff(corpus_df):
    files = files_in_directory('data')
    corpus_df = add_summary_description_message_to_corpus(corpus_df, '../../data/balanced_ultimate_false_link_processed/one_to_one_v2/', files)
    
    corpus_df = preprocess_on_corpus(corpus_df)
    
    run_w2v(corpus_df, 'word2vec_model_summary_description_message.bin')
    print('done')
    
    
def create_w2v_only_diff(diff_corpus_df):
    files = files_in_directory('data')
    diff_corpus_df = add_diff_to_corpus(diff_corpus_df, '../../data/balanced_ultimate_false_link_processed/one_to_one_v2/', files)
    
    diff_corpus_df = preprocess_on_corpus(diff_corpus_df)
    
    run_w2v(diff_corpus_df, 'word2vec_model_only_diff.bin')
    print('done')
    
    
create_w2v_without_diff(corpus_df)
create_w2v_only_diff(diff_corpus_df)
