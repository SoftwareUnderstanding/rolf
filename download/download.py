import os
import json
import pandas as pd

df = pd.read_csv('../dataset/train_all.csv', sep=';')
df_repos = df['Repo']
df_repos.drop_duplicates(inplace=True)
print(df_repos.size)
#Size: 10674

for repo in df_repos:
    name ='_'.join(repo.split('/')[-2:])
    os.system("somef describe -r {} -o tmp.json -p -t 0.8".format(name))
    os.system("mv {}.json data/".format(name))