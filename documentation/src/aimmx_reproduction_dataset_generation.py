import json
import pandas as pd



#Preprocess data from Papers with code
data = {'Text':[], 'Label':[], 'Repo':[]}
methods = json.load(open("dataset/methods.json"))
links = json.load(open("dataset/links.json"))
urls = []
labels = ['Computer Vision', 'Natural Language Processing', 'Reinforcement Learning']
papers = []


for link in links:
    repo_url = link['repo_url']
    paper = link['paper_url'].split('/')[4]
    for method in methods:
        count = 0
        if method['paper'] == paper and len(method['collections']) >= 1:
            count += 1
            area = method['collections'][0]['area']
        if count == 1 and area in labels:
            data['Repo'].append(repo_url)
            data['Label'].append(method['collections'][0]['area'])
            text = '"'
            for key, value in method.items():
                if key != 'collections':
                    text += str(value) + ' '
            text = text.replace('\n', ' ')
            text = text.replace (',', ' ')
            text += "\""
            data['Text'].append(str(text))

df_train = pd.DataFrame(data)
df_train.to_csv("dataset/train_3.csv", sep=';', index=False)