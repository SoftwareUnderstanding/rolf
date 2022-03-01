import pandas as pd
from collections import Counter
from Preprocessor import Preprocessor
import matplotlib.pyplot as plt

# Create filteres dataset
#df = pd.read_csv('data/readme.csv', sep=';')

#def filter_dataframe(df, cat):
#    for ind, row in df.iterrows(): 
#        if cat != str(row['Label']): df.drop([ind], inplace = True)

#df.drop(['Repo'], axis=1, inplace=True)
#filter_dataframe(df, 'Reinforcement Learning')

#df.to_csv('data/readme_reinforcement_learning.csv', sep=';', index=False)

#exit(0)

# Count word occurences


def get_keys(filename):
    df = pd.read_csv(filename, sep=';')
    Preprocessor(df).run()
    dict1={}
    for ind, row in df.iterrows():
        #print(row['Text'])
        for eachStr in row['Text'].split():
            if eachStr in dict1.keys():
                count = dict1[eachStr]
                count = count + 1
                dict1[eachStr.lower()] = count
            else: dict1[eachStr.lower()] = 1
    remekys = []
    #print(dict1)
    for key in dict1:
        if dict1[key] < len(df['Label']) or len(key) <= 2:
            remekys.append(key)
    for key in remekys:
        del dict1[key]
    #return dict1
    return list(dict1.keys()), list(dict1.values())




#filenames = ['readme_audio.csv', 'readme_computer_vision.csv', 
#             'readme_general.csv', 'readme_graphs.csv', 
#            'readme_natural_language_processing.csv', 'readme_reinforcement_learning.csv'm
#            'readme_sequential.csv']


keys_cv, values_cv = get_keys('data/readme_computer_vision.csv')
keys_au, values_au = get_keys('data/readme_audio.csv')
keys_gen, values_gen = get_keys('data/readme_general.csv')
keys_gr, values_gr = get_keys('data/readme_graphs.csv')
keys_nlp, values_nlp = get_keys('data/readme_natural_language_processing.csv')
keys_rl, values_rl = get_keys('data/readme_reinforcement_learning.csv')
keys_se, values_se = get_keys('data/readme_sequential.csv')

set_cv = set(keys_cv)
set_au = set(keys_au)
set_gen = set(keys_gen)
set_gr = set(keys_gr)
set_nlp = set(keys_nlp)
set_rl = set(keys_rl)
set_se = set(keys_se)

sets = []
sets.append(set_cv)
sets.append(set_au)
sets.append(set_gen)
sets.append(set_gr)
sets.append(set_nlp)
sets.append(set_rl)
sets.append(set_se)

intersect = set.intersection(*sets)
print('Intersection: ', intersect)
uni = set.union(set_au, set_cv, set_gen, set_gr, set_nlp, set_rl, set_se)
print('Union: ', uni)

print('Union - intersect: ', uni.difference(intersect))


print('Audio: ', set_au)
only_au = set_au.difference(set.union(set_cv, set_gen, set_gr, set_nlp, set_rl, set_se))
print('Only audio: ', only_au)
print('CV: ', set_cv)
only_cv = set_cv.difference(set.union(set_au, set_gen, set_gr, set_nlp, set_rl, set_se))
print('Only cv: ', only_cv)
print('General: ', set_gen)
only_gen = set_gen.difference(set.union(set_au, set_cv, set_gr, set_nlp, set_rl, set_se))
print('Only general: ', only_gen)
print('Graphs: ', set_gr)
only_gr = set_gr.difference(set.union(set_au, set_cv, set_gen, set_nlp, set_rl, set_se))
print('Only graphs: ', only_gr)
print('Nlp: ', set_nlp)
only_nlp = set_nlp.difference(set.union(set_au, set_cv, set_gen, set_gr, set_rl, set_se))
print('Only nlp: ', only_nlp)
print('Rl: ', set_rl)
only_rl = set_rl.difference(set.union(set_au, set_cv, set_gen, set_gr, set_nlp, set_se))
print('Only rl: ', only_rl)
print('Sequential: ', set_se)
only_se = set_se.difference(set.union(set_au, set_cv, set_gen, set_gr, set_nlp, set_rl))
print('Only sequential: ', only_se)









#keys_cv, values_cv = get_keys('data/readme_computer_vision.csv')
#plt.bar(keys_cv, values_cv, color='g')
#plt.title('Most frequent words in Computer Vision readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/cv.png')
#plt.show()

#keys_au, values_au = get_keys('data/readme_audio.csv')
#plt.bar(keys_au, values_au, color='g')
#plt.title('Most frequent words in Audio readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/audio.png')
#plt.show()

#keys_gen, values_gen = get_keys('data/readme_general.csv')
#plt.bar(keys_gen, values_gen, color='g')
#plt.title('Most frequent words in General readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/general.png')
#plt.show()

#keys_gr, values_gr = get_keys('data/readme_graphs.csv')
#plt.bar(keys_gr, values_gr, color='g')
#plt.title('Most frequent words in Graphs readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/graphs.png')
#plt.show()

#keys_nlp, values_nlp = get_keys('data/readme_natural_language_processing.csv')
#plt.bar(keys_nlp, values_nlp, color='g')
#plt.title('Most frequent words in Natural Language Processing readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/nlp.png')
#plt.show()

#keys_rl, values_rl = get_keys('data/readme_reinforcement_learning.csv')
#plt.bar(keys_rl, values_rl, color='g')
#plt.title('Most frequent words in Reinforcement Leraning readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/rl.png')
#plt.show()

#keys_se, values_se = get_keys('data/readme_sequential.csv')
#plt.bar(keys_se, values_se, color='g')
#plt.title('Most frequent words in Sequential readmes')
#plt.xticks(rotation='vertical')
#plt.savefig('results/sequential.png')
#plt.show()

