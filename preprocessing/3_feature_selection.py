import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import warnings
warnings.filterwarnings('ignore')

false_link_list = os.listdir('path to your false link') 
false_link_list

    
def check_hashed_data_correlation(true_link_df, false_link_df):
    same_true = 0
    same_false = 0
    for i in range(len(true_link_df)):
        if true_link_df['reporter_key'][i] == true_link_df['creator_key'][i]:
            same_true += 1
    print(f"percent of reporter/creator (data in issue) being same for true link: {(same_true/len(true_link_df))*100}")
    for i in range(len(false_link_df)):
        if false_link_df['reporter_key'][i] == false_link_df['creator_key'][i]:
            same_false += 1
    print(f"percent of reporter/creator (data in issue) being same for false link: {(same_false/len(false_link_df))*100}")
    same_true = 0
    same_false = 0
    for i in range(len(true_link_df)):
        if true_link_df['author'][i] == true_link_df['committer'][i]:
            same_true += 1
    print(f"percent of author/committer (data in commit) being same for true link: {(same_true/len(true_link_df))*100}")
    for i in range(len(false_link_df)):
        if false_link_df['author'][i] == false_link_df['committer'][i]:
            same_false += 1
    print(f"percent of author/committer (data in commit) being same for false link: {(same_false/len(false_link_df))*100}")
    
    
for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
#     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])
    check_hashed_data_correlation(true_link, false_link)
    print('-'*20)
    del true_link,false_link
    gc.collect()
    
    
def check_issuetype_status_component_correlation(true_link_df, false_link_df):
    true_link_df.loc[:, 'issuetype'] = true_link_df.issuetype.astype('category').cat.codes
    true_link_df.loc[:, 'status'] = true_link_df.status.astype('category').cat.codes
    true_link_df.loc[:, 'component'] = true_link_df.component.astype('category').cat.codes
    
    print(true_link_df.corr())
    
    false_link_df.loc[:, 'issuetype'] = false_link_df.issuetype.astype('category').cat.codes
    false_link_df.loc[:, 'status'] = false_link_df.status.astype('category').cat.codes
    false_link_df.loc[:, 'component'] = false_link_df.component.astype('category').cat.codes
    
    print(false_link_df.corr())
    
for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
#     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])
    check_issuetype_status_component_correlation(true_link[['issuetype', 'status', 'component']], false_link[['issuetype', 'status', 'component']])
    print('-'*20)
    del true_link,false_link
    gc.collect()
    
    
def check_histogram(true_link_df, false_link_df):
    plt.xticks(rotation=90)
    plt.title('true_issutype')
    plt.hist(true_link_df['issuetype'], align='mid')
    plt.show()
    
    plt.xticks(rotation=90)
    plt.title('true_status')
    plt.hist(true_link_df['status'], align='mid')
    plt.show()
    
    plt.xticks(rotation=90)
    plt.title('false_issutype')
    plt.hist(false_link_df['issuetype'], align='mid')
    plt.show()
    
    plt.xticks(rotation=90)
    plt.title('false_status')
    plt.hist(false_link_df['status'], align='mid')
    plt.show()

    
for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
#     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])
    check_histogram(true_link, false_link)
    print('-'*20)
    del true_link,false_link
    gc.collect()

def check_date_correlation(true_link_df, false_link_df):
    true_link_df.loc[:, 'created_date'] = pd.to_numeric(true_link_df.created_date)
    true_link_df.loc[:, 'updated_date'] = pd.to_numeric(true_link_df.updated_date)
    true_link_df.loc[:, 'last_resolved_date'] = pd.to_numeric(true_link_df.last_resolved_date)
    true_link_df.loc[:, 'author_time_date'] = pd.to_numeric(true_link_df.author_time_date)
    true_link_df.loc[:, 'commit_time_date'] = pd.to_numeric(true_link_df.commit_time_date)

    print(true_link_df.corr())
    
    false_link_df.loc[:, 'created_date'] = pd.to_numeric(false_link_df.created_date)
    false_link_df.loc[:, 'updated_date'] = pd.to_numeric(false_link_df.updated_date)
    false_link_df.loc[:, 'last_resolved_date'] = pd.to_numeric(false_link_df.last_resolved_date)
    false_link_df.loc[:, 'author_time_date'] = pd.to_numeric(false_link_df.author_time_date)
    false_link_df.loc[:, 'commit_time_date'] = pd.to_numeric(false_link_df.commit_time_date)
    
    print(false_link_df.corr())
    
    
for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
#     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])
    check_date_correlation(true_link[['created_date', 'updated_date', 'last_resolved_date', 'author_time_date', 'commit_time_date']], 
                           false_link[['created_date', 'updated_date', 'last_resolved_date', 'author_time_date', 'commit_time_date']])
    print('-'*20)
    del true_link,false_link
    gc.collect()
    
    
def check_issuetype_status_component_corr_with_classification_output(true_link_df, false_link_df):
    
    tmp = pd.concat([true_link_df, false_link_df], ignore_index=True)
    tmp.loc[:, 'out'] = tmp.out.astype('category').cat.codes
    tmp.loc[:, 'issuetype'] = tmp.issuetype.astype('category').cat.codes
    tmp.loc[:, 'status'] = tmp.status.astype('category').cat.codes
    tmp.loc[:, 'component'] = tmp.component.astype('category').cat.codes
    print(tmp.corr())

    
for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
#     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])
#     add classification column to data frames
    false_link.insert(24, 'out', '0')
    true_link.insert(23, 'out', '1')
    check_issuetype_status_component_corr_with_classification_output(true_link[['issuetype', 'status', 'component', 'out']], false_link[['issuetype', 'status', 'component', 'out']])
    print('-'*20)
    del true_link,false_link
    gc.collect()
    
    
def check_date_corr_with_classification_output(true_link_df, false_link_df):
    
    tmp = pd.concat([true_link_df, false_link_df], ignore_index=True)
    tmp.loc[:, 'created_date'] = pd.to_numeric(tmp.created_date)
    tmp.loc[:, 'updated_date'] = pd.to_numeric(tmp.updated_date)
    tmp.loc[:, 'last_resolved_date'] = pd.to_numeric(tmp.last_resolved_date)
    tmp.loc[:, 'author_time_date'] = pd.to_numeric(tmp.author_time_date)
    tmp.loc[:, 'out'] = tmp.out.astype('category').cat.codes
    
    print(tmp.corr())
    
for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
#     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])
#     add classification column to data frames
    false_link.insert(24, 'out', '0')
    true_link.insert(23, 'out', '1')
    check_date_corr_with_classification_output(true_link[['created_date', 'updated_date', 'last_resolved_date', 'author_time_date', 'out']], 
                                               false_link[['created_date', 'updated_date', 'last_resolved_date', 'author_time_date', 'out']])
    print('-'*20)
    del true_link,false_link
    gc.collect()

    
# #### doing the changes needed ####
def drop_reporter(true_link_df, false_link_df):
    true_link_df.drop(columns=['reporter_key'], inplace=True)
    false_link_df.drop(columns=['reporter_key'], inplace=True)
    return true_link_df, false_link_df

def drop_version(true_link_df, false_link_df):
    true_link_df.drop(columns=['version'], inplace=True)
    false_link_df.drop(columns=['version'], inplace=True)
    return true_link_df, false_link_df

def reduce_status_to_open_close_resolved(true_link_df, false_link_df):
    different_status = ['in progress', 'reopened', 'triage needed', 'patch available', 'awaiting feedback', 'ready to commit',
                        'changes suggested', 'review in progress']
    for stat in different_status:
        true_link_df.status.replace(stat ,value='open', inplace=True) 
        false_link_df.status.replace(stat ,value='open', inplace=True) 
        
    return true_link_df, false_link_df

def reduce_issuetype_to_task_newFeature_bug(true_link_df, false_link_df):
    to_newfeature = ['request', 'question', 'proposal', 'brainstorming']
    for stat in to_newfeature:
        true_link_df.issuetype.replace(stat ,value='new feature', inplace=True) 
        false_link_df.issuetype.replace(stat ,value='new feature', inplace=True) 
        
    to_task = ['dependency upgrade', 'documentation', 'sub-task', 'umbrella', 'test', 'epic', 'story', 'technical task']    
    for stat in to_task:
        true_link_df.issuetype.replace(stat ,value='task', inplace=True) 
        false_link_df.issuetype.replace(stat ,value='task', inplace=True) 
    
    return true_link_df, false_link_df


def one_hot_encoding(true_link_df, false_link_df, col_name):
#     for true link
    encoded_columns = pd.get_dummies(true_link_df[col_name])
    true_link_df = true_link_df.join(encoded_columns).drop(col_name, axis=1)
#     for false link
    encoded_columns = pd.get_dummies(false_link_df[col_name])
    false_link_df = false_link_df.join(encoded_columns).drop(col_name, axis=1)
    
    return true_link_df, false_link_df

def one_hot_encoding_for_issuetype_status(true_link_df, false_link_df):
    true_link_df, false_link_df = one_hot_encoding(true_link_df, false_link_df, 'issuetype')
    true_link_df, false_link_df = one_hot_encoding(true_link_df, false_link_df, 'status')

    return true_link_df, false_link_df

for project in false_link_list:
    if project.split('.')[1] == 'parquet':
        false_link = pd.read_parquet('path to false link'+project)
    else:
        false_link = pd.read_pickle('path to false link'+project)
    #     true links are all parquet
    true_link = pd.read_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print(project.split('.')[0])

    true_link, false_link = drop_reporter(true_link, false_link)
    true_link, false_link = drop_version(true_link, false_link)
    true_link, false_link = reduce_status_to_open_close_resolved(true_link, false_link)
    true_link, false_link = reduce_issuetype_to_task_newFeature_bug(true_link, false_link)
    true_link, false_link = one_hot_encoding_for_issuetype_status(true_link, false_link)

    if project.split('.')[1] == 'parquet':
        false_link.to_parquet('path to false link'+project.split('.')[0]+'.parquet')
    else:
        false_link.to_pickle('path to false link'+project.split('.')[0]+'.pickle')

    #     true links are all parquet
    true_link.to_parquet('path to true link'+project.split('.')[0]+'.parquet')
    print('-'*20)
    del true_link,false_link
    gc.collect()



