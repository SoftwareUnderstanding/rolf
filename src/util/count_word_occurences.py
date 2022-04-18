import csv
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import argparse
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)


def countWordOccurences(filename: str, limit: int, output_folder: str, cat: str):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    categories = defaultdict(lambda : Counter())
    sample_count = Counter()
    reader = csv.DictReader(open(filename), delimiter=';')
    print(cat)
    for row in reader:
        if cat != 'all':
            if row['Label'] == cat:
                categories['Label'].update(row['Text'].split(' '))
                sample_count['Label'] += 1
        else:
            categories['Label'].update(row['Text'].split(' '))
            sample_count['Label'] += 1
    for category, counter in categories.items():
        most_commons = counter.most_common(limit)
        keys = [key[0] for key in most_commons]
        values = [key[1] for key in most_commons]

        plt.figure(figsize=(20,15))
        plt.bar(keys, values, color='#00c7c3')
        plt.title(f'Most frequent words in preprocessed readmes for '+cat+' category', size=18, weight='bold')
        plt.xlabel('Words', size=16, weight='bold')
        plt.ylabel('Frequency', size=16, weight='bold')
        plt.axhline(y = sample_count[category], color = '#ff556b', linestyle = '-', label='Number of samples')
        plt.xticks(rotation=40, size=16)
        plt.yticks(size=14)
        plt.legend(prop={'size': 14})
        plt.savefig(output_path / f'{cat.lower().replace(" ", "_")}.svg')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python src/util/count_word_occurences.py", description='Count word occurences.')
    parser.add_argument('filename', help='name of the file with the preprocessed data')
    parser.add_argument('--limit', type=int, default=20, required=False, help='Set the limit of the common words. (default 20)')
    parser.add_argument('--outFolder', type=str, default='results/pics/', required=False, help='Give the folder for the output. (default "results/")')
    parser.add_argument('--category', type=str, default='all', required=False, help='Define which category to do the analysis for')

    args = parser.parse_args()
    if args.category:
        countWordOccurences(args.filename, args.limit, args.outFolder, args.category)
    else:
        countWordOccurences(args.filename, args.limit, args.outFolder)

    