from tunimi import IloTunimi
import sys
from collections import Counter

if __name__ == '__main__':
	counter = Counter()
	tokenizer = IloTunimi()
	for xs in sys.stdin:
		xs = tokenizer(xs, as_str=True)
		if '<unk>' in xs:
			continue
		i = 0
		lst = []
		tmp = []
		while i < len(xs):
			if xs[i] == 'li':
				tmp = ['li']
			elif xs[i] in {'<sep>', 'la'}:
				tmp = []
			elif tmp != []:
				tmp.append(xs[i])
				if tmp[-1] == 'e':
					lst.append(' '.join(tmp))
					tmp = []
			i += 1
		counter.update(lst)

	for x, f in counter.most_common():
		print(x + '\t' + str(f))

