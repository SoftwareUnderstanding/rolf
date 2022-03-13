import os
import json
import pandas as pd
import requests

df = pd.read_csv('../../documentation/dataset/awesome_links.csv', sep=';')
df_repos = df['Repo']
df_repos.drop_duplicates(inplace=True)

for repo in df_repos:
    name ='_'.join(repo.split('/')[-2:])
    readme = 'https://raw.githubusercontent.com/' + '/'.join(repo.split('/')[-2:]) + '/master/README.md'
    print(readme)
    r = requests.get(readme)
    f = open("data/{}.txt".format(name), "w")
    f.write(r.text)
    f.close
    #os.system("somef describe -r {} -o {}.json -p -t 0.8".format(repo, name))
    #os.system("mv {}.json data/".format(name))
    
