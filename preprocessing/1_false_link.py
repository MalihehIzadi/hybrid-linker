# ##### run this script for each project to produce the false link for that project #####

import pandas as pd
import numpy as np
from datetime import timedelta

# working on the commit
dummy_commit = pd.read_parquet('path to read commit')
dummy_commit = dummy_commit.rename(columns={'issue_id': 'commit_issue_id'})

# get rows with null issue_ids
dummy_commit.reset_index(drop=True, inplace=True)
null_issue_id = list(np.where(pd.isnull(dummy_commit['commit_issue_id']))[0])
print(null_issue_id)
print(null_issue_id)

dummy_commit = dummy_commit.drop(np.where(pd.isnull(dummy_commit['commit_issue_id']))[0])
print(dummy_commit.shape)

dummy_commit.commit_issue_id = dummy_commit.commit_issue_id.astype(int)
dummy_commit.commit_issue_id = dummy_commit.commit_issue_id.astype(str)

#  working on the issue
dummy_issue = pd.read_parquet('path to read issue')
dummy_issue.issue_id = dummy_issue.issue_id.astype(str)

# creating false link
x = list(dummy_commit.columns)
x.remove('source')
cols = list(dummy_issue.columns) + x

False_link = pd.DataFrame(columns=cols)

unique_issue_id_in_commits = dummy_commit.commit_issue_id.unique()
print(len(unique_issue_id_in_commits))

for i in range(len(unique_issue_id_in_commits)):
    selected_commit = dummy_commit.loc[dummy_commit['commit_issue_id'] == unique_issue_id_in_commits[i]]
    selected_issue = dummy_issue.loc[(dummy_issue['issue_id'] != unique_issue_id_in_commits[i])]
    for j in range(len(selected_commit)):
        tmp = selected_issue.loc[((timedelta(days = -7) <= selected_issue['created_date'] - selected_commit.iloc[j]['author_time_date']) &
                                   (selected_issue['created_date'] - selected_commit.iloc[j]['author_time_date'] <= timedelta(days = 7))) |
                                  ((timedelta(days = -7) <= selected_issue['created_date'] - selected_commit.iloc[j]['commit_time_date']) &
                                   (selected_issue['created_date'] - selected_commit.iloc[j]['commit_time_date'] <= timedelta(days = 7))) |
                                  ((timedelta(days = -7) <= selected_issue['updated_date'] - selected_commit.iloc[j]['author_time_date']) &
                                   (selected_issue['updated_date'] - selected_commit.iloc[j]['author_time_date'] <= timedelta(days = 7))) |
                                  ((timedelta(days = -7) <= selected_issue['updated_date'] - selected_commit.iloc[j]['commit_time_date']) &
                                   (selected_issue['updated_date'] - selected_commit.iloc[j]['commit_time_date'] <= timedelta(days = 7))) |
                                  ((timedelta(days = -7) <= selected_issue['last_resolved_date'] - selected_commit.iloc[j]['author_time_date']) &
                                   (selected_issue['last_resolved_date'] - selected_commit.iloc[j]['author_time_date'] <= timedelta(days = 7))) |
                                  ((timedelta(days = -7) <= selected_issue['last_resolved_date'] - selected_commit.iloc[j]['commit_time_date']) &
                                   (selected_issue['last_resolved_date'] - selected_commit.iloc[j]['commit_time_date'] <= timedelta(days = 7)))
                                 ]
        if tmp.shape[0] != 0:
            y = selected_commit.iloc[j]
            x = tmp
            resulted_false_link = pd.merge(left=pd.DataFrame(x), right=pd.DataFrame(y).transpose(), how='left', left_on='source', right_on='source')
            False_link = False_link.append(resulted_false_link, ignore_index=True)
print(False_link.shape)
False_link.to_parquet('path to save true link')