import os
import json
import pandas as pd

df = pd.read_csv('../../data/train_all.csv', sep=';')
df_repos = df['Repo']
df_repos.drop_duplicates(inplace=True)
#Size: 10674

os.system("mkdir data")
for repo in df_repos:
    name ='_'.join(repo.split('/')[-2:])
    os.system("somef describe -r {} -o {}.json -p -t 0.8".format(repo, name))
    os.system("mv {}.json data/".format(name))
    os.system("mkdir data")
    
