import csv
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import argparse
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)

def countWordOccurences(filename: str, limit: int, output_folder: str):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    categories = defaultdict(lambda : Counter())
    sample_count = Counter()
    
    reader = csv.DictReader(open(filename), delimiter=';')
    for row in reader:
        categories[row['Label']].update(row['Text'].split(' '))
        sample_count[row['Label']] += 1

    for category, counter in categories.items():
        most_commons = counter.most_common(limit)
        keys = [key[0] for key in most_commons]
        values = [key[1] for key in most_commons]

        plt.figure(figsize=(20,15))
        plt.bar(keys, values, color='#00c7c3')
        plt.title(f'Most frequent words in {category} readmes', size=18, weight='bold')
        plt.xlabel('Words', size=16, weight='bold')
        plt.ylabel('Frequency', size=16, weight='bold')
        plt.axhline(y = sample_count[category], color = '#ff556b', linestyle = '-', label='Number of samples')
        plt.xticks(rotation=40, size=16)
        plt.yticks(size=14)
        plt.legend(prop={'size': 14})
        plt.savefig(output_path / f'{category.lower().replace(" ", "_")}_raw.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python src/count_word_occurences.py", description='Count word occurences.')
    parser.add_argument('filename', help='name of the file with the preprocessed data')
    parser.add_argument('--limit', type=int, default=20, required=False, help='Set the limit of the common words. (default 20)')
    parser.add_argument('--outFolder', type=str, default='results/pics/', required=False, help='Give the folder for the output. (default "results/")')

    args = parser.parse_args()
    countWordOccurences(args.filename, args.limit, args.outFolder)
