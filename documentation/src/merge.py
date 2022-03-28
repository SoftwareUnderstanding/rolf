import pandas as pd

df4 = pd.read_csv('../data/readme_train.csv', sep=';')
df5 = pd.read_csv('../data/abstracts.csv', sep=';')

merged = pd.concat([df4, df5])
print(merged.shape)
merged.to_csv('../data/merged_abstracts_readme_train.csv', sep=';', index=False)