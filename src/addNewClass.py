import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--newdata', dest ='newdata', help='Define the destination of the new class in csv format.', default=False)
parser.add_argument('--data', dest ='data', help='Define the destination of the existing classes in csv format.', default=False)


args = parser.parse_args()

newdata_path = Path(args.newdata)
if not newdata_path.is_file():
	raise FileNotFoundError('File does not exist: ' + args['newdata'])

data_path = Path(args.data)
if not data_path.is_file():
	raise FileNotFoundError('File does not exist: ' + args['data'])

df_new = pd.read_csv(newdata_path, sep=';')
df = pd.read_csv(data_path, sep=';')
df_concat = pd.concat([df, df_new])
df_concat.to_csv(data_path, sep=';', index = False)

print('Success!')
