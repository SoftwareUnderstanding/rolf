import pandas as pd
import Preprocessor

df = pd.read_csv('../data/abstracts.csv', sep=';')
pr = Preprocessor.Preprocessor(df)

pr.run()

print(df.head())
print(df['clean_Text'][0])
print()
print(df['clean_Text'][1])
print()
print(df['clean_Text'][2])
print()
print(df['clean_Text'][3])