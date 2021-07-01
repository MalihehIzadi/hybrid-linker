# ##### run this script for each project to produce the true link for that project #####

import pandas as pd
import numpy as np

dummy_commit = pd.read_parquet('path to read commit')
dummy_commit

# deleting all the null issue_ids
dummy_commit.reset_index(drop=True, inplace=True)

print(np.where(pd.isnull(dummy_commit['issue_id']))[0])
print(len(np.where(pd.isnull(dummy_commit['issue_id']))[0]))
dummy_commit = dummy_commit.drop(np.where(pd.isnull(dummy_commit['issue_id']))[0])
dummy_commit.shape

dummy_commit.issue_id = dummy_commit.issue_id.astype(int)
dummy_commit.issue_id = dummy_commit.issue_id.astype(str)

# working on issue
dummy_issue = pd.read_parquet('path to read issue')
dummy_issue

print(np.where(pd.isnull(dummy_issue['issue_id'])))
print(len(np.where(pd.isnull(dummy_issue['issue_id']))[0]))
# 
dummy_issue.issue_id = dummy_issue.issue_id.astype(str)

# Building True Links
x = list(dummy_commit.columns)
x.remove('source')
x.remove('issue_id')
cols = list(dummy_issue.columns) + x

True_link = pd.DataFrame(columns=cols)

unique_issue_id_in_commits = dummy_commit.issue_id.unique()
print(len(unique_issue_id_in_commits))

for i in range(len(unique_issue_id_in_commits)):
    selected_commit = dummy_commit.loc[dummy_commit['issue_id'] == unique_issue_id_in_commits[i]]
    selected_issue = dummy_issue.loc[dummy_issue['issue_id'] == unique_issue_id_in_commits[i]]
    resulted_true_link = pd.merge(left=selected_issue, right=selected_commit, how='left', left_on=['source', 'issue_id'], right_on=['source', 'issue_id'])
    True_link = True_link.append(resulted_true_link, ignore_index=True)
print(True_link.shape)
True_link.to_parquet('path to save true link')