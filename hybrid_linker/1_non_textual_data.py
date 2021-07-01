import pandas as pd
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')


def files_in_directory(path):
    files = os.listdir(path)
    try:
        files.remove('.ipynb_checkpoints')
    except:
        print("ipynb_checkpoints doesn't exits")
    return files


used_columns = ["creator_key", "created_date","updated_date", "last_resolved_date", "author", "committer", "author_time_date", 
                          "commit_time_date","bug", "new feature", "task", "closed", "open", "resolved", 'label', 'train_flag']
cassandra_columns = ["creator_key", "created_date","updated_date", "last_resolved_date", "author", "committer", "author_time_date", 
                          "commit_time_date","bug", "new feature", "task", "open", "resolved", 'label', 'train_flag']
main_columns = ['creator_key', 'updated_date', 'hash', 'author', 'committer', 'author_time_date', 'bug', 'new feature', 'task', 'closed' 
                'open', 'resolved', 'label', 'train_flag']


def non_textual():
    files = files_in_directory('data/trained_flagged_data/')
    for file in files:
        print(file)
        df = pd.read_parquet('data/trained_flagged_data/'+file)
        
        if file.split('.')[0] == 'cassandra':
            df = df[cassandra_columns]
            df.insert(loc=9, column='closed', value=df.shape[0]*[0])
        else:
            df = df[used_columns]
        
        df_train = df.loc[df['train_flag'] == 1]
        df_test = df.loc[df['train_flag'] == 0]

        df_train.to_parquet('data/non_textual_data/train/'+file)
        df_test.to_parquet('data/non_textual_data/test/'+file)

        del df, df_train, df_test
        gc.collect()
        print('-------------------')
        
        
non_textual()