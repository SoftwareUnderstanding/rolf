import pandas as pd
import pickle
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessor import Preprocessor
import os
import argparse
import requests
from pathlib import Path



parser = argparse.ArgumentParser(description='Try classification')
parser.add_argument('--models_dir', dest='models_dir', help='The path to the folder containing the models', required=True)
parser.add_argument('--readme_url', dest='readme_url', help='GitHub URL to the raw readme')
parser.add_argument('--text_file', dest='text_file', help='Path to the txt file that contains the text')
args = parser.parse_args()

text = ''

if args.readme_url:
    r = requests.get(args.readme_url)
    r.raise_for_status()
    text = r.text
elif args.text_file:
    with open(args.text_file) as f:
        text = f.read().replace('\n', ' ')
else:
    raise Exception('No text is given')


text = pd.DataFrame([text], columns=['Text'])
Preprocessor(text).run()
dir = Path(args.models_dir)
models = os.listdir(dir)
predictions = []
for model in models:
    clf = pickle.load(open(dir/model, 'rb'))
    pred = clf.predict(text)
    if pred != 'Other':
        predictions.append(pred[0])

print('Predictions:')
if not predictions:
    print('Other')
else:
    print(predictions)
