import pandas as pd
import pickle
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class DataframeContainer:
    def __init__(self, name, filanemCsv):
        self.name = name
        self.df_X = pd.DataFrame([sys.argv[1]], columns=['Text'])
        self.df_X.reset_index(drop=True, inplace=True)

    def predict(self):
        self.y_pred = self.clf.predict(self.df_X)
        return self.y_pred

    def load_pickle(self):
        filename = 'pickles/3/' + self.name + '.sav'
        self.clf = pickle.load(open(filename, 'rb'))


names_list = ["Audio", "Computer Vision", "Graphs", "General", "Natural Language Processing", "Reinforcement Learning", "Sequential"]
output = []
dataframecontainers_list = [DataframeContainer(name, 'dataset/somef_data.csv') for name in names_list]
for container in dataframecontainers_list:
    container.load_pickle()
    output.append((container.predict()[0], container.name))

for o in output:
    print(o[1], o[0])