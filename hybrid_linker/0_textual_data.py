import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
import itertools
import os
import gc
import warnings
warnings.filterwarnings('ignore')


def summary_description_corpus(issue_corpus_df, path, file):
    df = pd.read_parquet(path+file)
    df = df.loc[df['train_flag'] == 1].copy()
    print(df.shape)
    
    
    df['summary_processed'].loc[df['summary_processed'].isnull()] = df['summary_processed'].loc[df['summary_processed'].isnull()].apply(lambda x: [])
    df['description_processed'].loc[df['description_processed'].isnull()] = df['description_processed'].loc[df['description_processed'].isnull()].apply(lambda x: [])
    df['summary_processed_text'] = df['summary_processed'].apply(lambda x: ' '.join(x))
    df['description_processed_text'] = df['description_processed'].apply(lambda x: ' '.join(x))
    issue_corpus_df.loc[:, 'data'] = df['summary_processed_text'] + df['description_processed_text']

    del df
    gc.collect()
    print(issue_corpus_df.shape)
    return issue_corpus_df


def message_changedfile_corpus(commit_corpus_df, path, file):
    df = pd.read_parquet(path+file)
    df = df.loc[df['train_flag'] == 1].copy()
    print(df.shape)
    
    
    df['message_processed'].loc[df['message_processed'].isnull()] = df['message_processed'].loc[df['message_processed'].isnull()].apply(lambda x: [])
    df['changed_files'].loc[df['changed_files'].isnull()] = df['changed_files'].loc[df['changed_files'].isnull()].apply(lambda x: [])
    df['message_processed_text'] = df['message_processed'].apply(lambda x: ' '.join(x))
    df['changed_files_text'] = df['changed_files'].apply(lambda x: ' '.join(x))
    commit_corpus_df.loc[:, 'data'] = df['message_processed_text'] + df['changed_files_text']

    del df
    gc.collect()
    print(commit_corpus_df.shape)
    return commit_corpus_df


def diff_corpus(diff_corpus_df, path, file):
    df = pd.read_parquet(path+file)
    df = df.loc[df['train_flag'] == 1].copy()
    print(df.shape)
    
    
    df['processDiffCode'].loc[df['processDiffCode'].isnull()] = df['processDiffCode'].loc[df['processDiffCode'].isnull()].apply(lambda x: [])
    df.loc[:, 'tmp'] = df['processDiffCode'].apply(lambda x: list(itertools.chain(*x))) # inorder to flatten the lists of processDiffCode column
    df['processDiffCode_text'] = df['tmp'].apply(lambda x: ' '.join(x))
    diff_corpus_df.loc[:, 'data'] = df['processDiffCode_text']

    del df
    gc.collect()
    print(diff_corpus_df.shape)
    return diff_corpus_df


def preprocess_on_corpus(data_frame):
    data_frame = data_frame.reset_index(drop=True)
    data_frame['data'].fillna(value='', inplace=True)
    return data_frame


def files_in_directory(path):
    files = os.listdir(path)
    try:
        files.remove('.ipynb_checkpoints')
    except:
        print("ipynb_checkpoints doesn't exits")
    return files


def run_tfidf(data_frame, file_name_to_save):
   
    tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=10000) #ngram_range=(1,3)

    tfidf.fit(data_frame['data'])

    with open(file_name_to_save, 'wb') as file:
        pickle.dump(tfidf, file)    
        
        
def issue_tfidf():
    files = files_in_directory('data/trained_flagged_data/')
    for file in files:
        print(file)
        issue_corpus_df = pd.DataFrame(columns=['data'])
        issue_corpus_df = summary_description_corpus(issue_corpus_df, 'data/trained_flagged_data/', file)

        issue_corpus_df = preprocess_on_corpus(issue_corpus_df)
        
        run_tfidf(issue_corpus_df, 'data/textual_data/models/issue_model/'+str(file.split('.')[0])+'.pickle')

        del issue_corpus_df
        gc.collect()
        print('-----------------')
        
        
def commit_tfidf():
    files = files_in_directory('data/trained_flagged_data/')
    for file in files:
        print(file)
        commit_corpus_df = pd.DataFrame(columns=['data'])
        commit_corpus_df = message_changedfile_corpus(commit_corpus_df, 'data/trained_flagged_data/', file)

        commit_corpus_df = preprocess_on_corpus(commit_corpus_df)
        
        run_tfidf(commit_corpus_df, 'data/textual_data/models/commit_model/'+str(file.split('.')[0])+'.pickle')

        
        del commit_corpus_df
        gc.collect()
        print('-----------------')
        

def diff_tfidf():
    files = files_in_directory('data/trained_flagged_data/')
    for file in files:
        print(file)
        diff_corpus_df = pd.DataFrame(columns=['data'])
        diff_corpus_df = diff_corpus(diff_corpus_df, 'data/trained_flagged_data/',file)

        diff_corpus_df = preprocess_on_corpus(diff_corpus_df)

        run_tfidf(diff_corpus_df, 'data/textual_data/models/code_model/'+str(file.split('.')[0])+'.pickle')

        
        del diff_corpus_df
        gc.collect()
        print('-------------------')
        
        
issue_tfidf()
commit_tfidf()
diff_tfidf()

# ####### transforming train data #######

def transforming_train_data():
    files = os.listdir('data/trained_flagged_data/')
    for file in files:
        print(file)
        issue_load_path = 'data/textual_data/models/issue_model/'
        commit_load_path = 'data/textual_data/models/commit_model/'
        code_load_path = 'data/textual_data/models/code_model/'
        issue_save_path = 'data/textual_data/transformed_train/issue/'
        commit_save_path = 'data/textual_data/transformed_train/commit/'
        code_save_path = 'data/textual_data/transformed_train/code/'
        
        with open(issue_load_path+file.split('.')[0]+'.pickle', 'rb') as issue_model_file, open(commit_load_path+file.split('.')[0]+'.pickle', 'rb') as commit_model_file, open(code_load_path+file.split('.')[0]+'.pickle', 'rb') as code_model_file:
            issue_model = pickle.load(issue_model_file)
            commit_model = pickle.load(commit_model_file)
            code_model = pickle.load(code_model_file)

            df = pd.read_parquet('data/trained_flagged_data/'+file.split('.')[0]+'.parquet')
            df = df.loc[df['train_flag'] == 1]
            
            df['summary_processed'].loc[df['summary_processed'].isnull()] = df['summary_processed'].loc[df['summary_processed'].isnull()].apply(lambda x: [])
            df['description_processed'].loc[df['description_processed'].isnull()] = df['description_processed'].loc[df['description_processed'].isnull()].apply(lambda x: [])
            df['summary_processed_text'] = df['summary_processed'].apply(lambda x: ' '.join(x))
            df['description_processed_text'] = df['description_processed'].apply(lambda x: ' '.join(x))
            df['message_processed'].loc[df['message_processed'].isnull()] = df['message_processed'].loc[df['message_processed'].isnull()].apply(lambda x: [])
            df['changed_files'].loc[df['changed_files'].isnull()] = df['changed_files'].loc[df['changed_files'].isnull()].apply(lambda x: [])
            df['message_processed_text'] = df['message_processed'].apply(lambda x: ' '.join(x))
            df['changed_files_text'] = df['changed_files'].apply(lambda x: ' '.join(x))
            df['processDiffCode'].loc[df['processDiffCode'].isnull()] = df['processDiffCode'].loc[df['processDiffCode'].isnull()].apply(lambda x: [])
            df.loc[:, 'tmp'] = df['processDiffCode'].apply(lambda x: list(itertools.chain(*x))) # inorder to flatten the lists of processDiffCode column
            df['processDiffCode_text'] = df['tmp'].apply(lambda x: ' '.join(x))

            issue = issue_model.transform(df['summary_processed_text'] + df['description_processed_text'])
            commit = commit_model.transform(df['message_processed_text'] + df['changed_files_text'])
            code = code_model.transform(df['processDiffCode_text'])

            
            with open(issue_save_path+file.split('.')[0]+'.pickle', 'wb') as issue_file:
                pickle.dump(issue, issue_file)  
            with open(commit_save_path+file.split('.')[0]+'.pickle', 'wb') as commit_file:
                pickle.dump(commit, commit_file)  
            with open(code_save_path+file.split('.')[0]+'.pickle', 'wb') as code_file:
                pickle.dump(code, code_file)  

        del issue_model, commit_model, code_model, df, issue, commit, code
        gc.collect()
        print('-'*25)
        

transforming_train_data()

# ####### transforming test data #######

def transforming_test_data():
    files = os.listdir('data/trained_flagged_data/')
    for file in files:
        print(file)
        issue_load_path = 'data/textual_data/models/issue_model/'
        commit_load_path = 'data/textual_data/models/commit_model/'
        code_load_path = 'data/textual_data/models/code_model/'
        issue_save_path = 'data/textual_data/transformed_test/issue/'
        commit_save_path = 'data/textual_data/transformed_test/commit/'
        code_save_path = 'data/textual_data/transformed_test/code/'
        
        with open(issue_load_path+file.split('.')[0]+'.pickle', 'rb') as issue_model_file, open(commit_load_path+file.split('.')[0]+'.pickle', 'rb') as commit_model_file, open(code_load_path+file.split('.')[0]+'.pickle', 'rb') as code_model_file:
            issue_model = pickle.load(issue_model_file)
            commit_model = pickle.load(commit_model_file)
            code_model = pickle.load(code_model_file)

            df = pd.read_parquet('data/trained_flagged_data/'+file.split('.')[0]+'.parquet')
            df = df.loc[df['train_flag'] == 0]
            
            
            df['summary_processed'].loc[df['summary_processed'].isnull()] = df['summary_processed'].loc[df['summary_processed'].isnull()].apply(lambda x: [])
            df['description_processed'].loc[df['description_processed'].isnull()] = df['description_processed'].loc[df['description_processed'].isnull()].apply(lambda x: [])
            df['summary_processed_text'] = df['summary_processed'].apply(lambda x: ' '.join(x))
            df['description_processed_text'] = df['description_processed'].apply(lambda x: ' '.join(x))
            df['message_processed'].loc[df['message_processed'].isnull()] = df['message_processed'].loc[df['message_processed'].isnull()].apply(lambda x: [])
            df['changed_files'].loc[df['changed_files'].isnull()] = df['changed_files'].loc[df['changed_files'].isnull()].apply(lambda x: [])
            df['message_processed_text'] = df['message_processed'].apply(lambda x: ' '.join(x))
            df['changed_files_text'] = df['changed_files'].apply(lambda x: ' '.join(x))
            df['processDiffCode'].loc[df['processDiffCode'].isnull()] = df['processDiffCode'].loc[df['processDiffCode'].isnull()].apply(lambda x: [])
            df.loc[:, 'tmp'] = df['processDiffCode'].apply(lambda x: list(itertools.chain(*x))) # inorder to flatten the lists of processDiffCode column
            df['processDiffCode_text'] = df['tmp'].apply(lambda x: ' '.join(x))

            issue = issue_model.transform(df['summary_processed_text'] + df['description_processed_text'])
            commit = commit_model.transform(df['message_processed_text'] + df['changed_files_text'])
            code = code_model.transform(df['processDiffCode_text'])

            
            with open(issue_save_path+file.split('.')[0]+'.pickle', 'wb') as issue_file:
                pickle.dump(issue, issue_file)  
            with open(commit_save_path+file.split('.')[0]+'.pickle', 'wb') as commit_file:
                pickle.dump(commit, commit_file)  
            with open(code_save_path+file.split('.')[0]+'.pickle', 'wb') as code_file:
                pickle.dump(code, code_file)  

        del issue_model, commit_model, code_model, df, issue, commit, code
        gc.collect()
        print('-'*25)
        
        
transforming_test_data()