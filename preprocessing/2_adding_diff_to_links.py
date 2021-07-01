import pandas as pd
import gc
import os


def load_diff_file(diff_processed_name):
    if diff_processed_name.split('.')[1] == 'parquet':
        diff_file = pd.read_parquet('path to load diff code file')
    else:
        diff_file = pd.read_pickle('path to load diff code file')
    return diff_file

def create_ultimate_true_link(true_link_name, diff_processed_name, ultimate_true_link_name, save_type):
    true_file = pd.read_parquet('path to load true link without diff code')
    
    diff_file = load_diff_file(diff_processed_name)
    diff_file = diff_file[['source', 'repo', 'hash', 'changed_files', 'processDiffCode']]
    
    ultimate_true_link = pd.merge(left=true_file, right=diff_file, how='left', left_on=['source', 'repo', 'hash'], right_on=['source', 'repo', 'hash'])
    
    print('number of diffs in true links: ', ultimate_true_link[ultimate_true_link['changed_files'].notnull() | ultimate_true_link['processDiffCode'].notnull()].shape)
    
    if save_type == 'parquet':
        ultimate_true_link.to_parquet('path to save true link with diff code')
    else:
        ultimate_true_link.to_pickle('path to save true link with diff code')
        
def create_ultimate_false_link(false_link_name, diff_processed_name, ultimate_false_link_name, save_type):
    false_file = pd.read_parquet('path to load false link without diff code')
    
    diff_file = load_diff_file(diff_processed_name)
    diff_file = diff_file[['source', 'repo', 'hash', 'changed_files', 'processDiffCode']]
    
    ultimate_false_link = pd.merge(left=false_file, right=diff_file, how='left', left_on=['source', 'repo', 'hash'], right_on=['source', 'repo', 'hash'])
    del false_file, diff_file
    gc.collect()
    print('number of diffs in false links: ', ultimate_false_link[ultimate_false_link['changed_files'].notnull() | ultimate_false_link['processDiffCode'].notnull()].shape)
    
    if ultimate_false_link_name == 'ambari':
        ultimate_false_link = ultimate_false_link.sample(n = 50*35589) # 50 times bigger than its true link
        
    if save_type == 'parquet':
        ultimate_false_link.to_parquet('path to save false link with diff code')
    else:
        ultimate_false_link.to_pickle('path to save false link with diff code')
        
def create_link():
    create_ultimate_true_link('netbeans_true_link.parquet', 'netbeans.pickle', 'netbeans', 'parquet')
    create_ultimate_false_link('netbeans_false_link.parquet', 'netbeans.pickle', 'netbeans', 'parquet')
    print('netbeans done')
    create_ultimate_true_link('calcite_true_link.parquet', 'calcite.pickle', 'calcite', 'parquet')
    create_ultimate_false_link('calcite_false_link.parquet', 'calcite.pickle', 'calcite', 'parquet')
    print('calcite done')
    create_ultimate_true_link('beam_true_link.parquet', 'beam.pickle', 'beam', 'parquet')
    create_ultimate_false_link('beam_false_link.parquet', 'beam.pickle', 'beam', 'pickle')
    print('beam done')
    create_ultimate_true_link('flink_true_link.parquet', 'flink.pickle', 'flink', 'parquet')
    create_ultimate_false_link('flink_false_link.parquet', 'flink.pickle', 'flink', 'pickle')
    print('flink done')
    create_ultimate_true_link('airflow_true_link.parquet', 'airflow.parquet', 'airflow', 'parquet')
    create_ultimate_false_link('airflow_false_link.parquet', 'airflow.parquet', 'airflow', 'parquet')
    print('airflow done')
    create_ultimate_true_link('cassandra_true_link.parquet', 'cassandra.parquet', 'cassandra', 'parquet')
    create_ultimate_false_link('cassandra_false_link.parquet', 'cassandra.parquet', 'cassandra', 'parquet')
    print('cassandra done')
    create_ultimate_true_link('freemarker_true_link.parquet', 'freemarker.parquet', 'freemarker', 'parquet')
    create_ultimate_false_link('freemarker_false_link.parquet', 'freemarker.parquet', 'freemarker', 'parquet')
    print('freemarker done')
    create_ultimate_true_link('groovy_true_link.parquet', 'groovy.parquet', 'groovy', 'parquet')
    create_ultimate_false_link('groovy_false_link.parquet', 'groovy.parquet', 'groovy', 'parquet')
    print('groovy done')
    create_ultimate_true_link('ambari_true_link.parquet', 'ambari.parquet', 'ambari', 'parquet')
    create_ultimate_false_link('ambari_false_link.parquet', 'ambari.parquet', 'ambari', 'pickle')
    print('ambari done')
    create_ultimate_true_link('arrow_true_link.parquet', 'arrow.parquet', 'arrow', 'parquet')
    create_ultimate_false_link('arrow_false_link.parquet', 'arrow.parquet', 'arrow', 'pickle')
    print('arrow done')
    create_ultimate_true_link('isis_true_link.parquet', 'isis.parquet', 'isis', 'parquet')
    create_ultimate_false_link('isis_false_link.parquet', 'isis.parquet', 'isis', 'parquet')
    print('isis done')
    create_ultimate_true_link('ignite_true_link.parquet', 'ignite.parquet', 'ignite', 'parquet')
    create_ultimate_false_link('ignite_false_link.parquet', 'ignite.parquet', 'ignite', 'parquet')
    print('ignite done')
    
create_link()
