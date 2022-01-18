import pandas as pd
import pickle
import sys

class DataframeContainer:
    def __init__(self, name):
        self.name = name

    def predict(self):
        data = {'Text': [args]}
        df = pd.DataFrame(data)
        self.y_pred = self.clf.predict(df)
        return self.y_pred[0]

    def load_pickle(self):
        filename = 'pickles/' + sys.argv[1] + '/' + self.name + ".sav"
        self.clf = pickle.load(open(filename, 'rb'))

args = ' '.join(sys.argv[2:])
print(args)

names_list = ["Audio", "Computer Vision", "Graphs", "Natural Language Processing", "Reinforcement Learning", "Sequential"]
dataframecontainers_list = [DataframeContainer(name) for name in names_list]
for container in dataframecontainers_list:
    container.load_pickle()

predictions = []
for container in dataframecontainers_list:
    predictions.append(container.predict())

print(predictions)