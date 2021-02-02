from tunimi import IloTunimi
import sys
from collections import Counter

if __name__ == '__main__':
	counter = Counter()
	tokenizer = IloTunimi()
	for xs in sys.stdin:
		xs = tokenizer(xs, as_str=True)
		xs = [x for x in xs if not x.startswith('<')]
		counter.update(xs)

	for x, f in counter.most_common():
		print(x + '\t' + str(f))

