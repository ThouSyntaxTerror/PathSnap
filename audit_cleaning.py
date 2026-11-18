"""Summarize what the shared GPS cleaning step changes in the student data."""
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
from pathsnap import clean_track

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
args = parser.parse_args()
total = Counter()
biggest = []
for path in args.data.rglob('*.track'):
    clean, counts = clean_track(np.loadtxt(path, ndmin=2))
    total.update(counts)
    biggest.append((counts['outlier_rows'], path.parent.name))
print(dict(total))
print('Most outlier removals:', sorted(biggest, reverse=True)[:10])
