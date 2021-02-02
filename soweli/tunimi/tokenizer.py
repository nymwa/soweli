import re
from .vocabulary import TokiPonaVocabulary

class IloTunimi:
	def __init__(self, comma_split_special_tokens = None):
		self.vocab = TokiPonaVocabulary(comma_split_special_tokens = comma_split_special_tokens)
		self.prp_pattern = re.compile(r'^([AIUEO]|[KSNPML][aiueo]|[TJ][aueo]|W[aie])n?(([ksnpml][aiueo]|[tj][aueo]|w[aie])n?)*$')

	def convert(self, x):
		if x in {'.', '!', '?', ':'}:
			return self.vocab.sep_id

		x = re.sub('[^0-9A-Za-z]', '', x)

		if x == '':
			return None

		if x in self.vocab.dictionary:
			return self.vocab.dictionary[x]
		elif x.isdecimal():
			return self.vocab.num_id
		elif self.prp_pattern.match(x) and ('nm' not in x) and ('nn' not in x):
			return self.vocab.prp_id
		else:
			return self.vocab.unk_id

	def choose(self, x, y, no_sep, no_unk, no_num, no_prp):
		z = self.vocab[y]
		if (no_sep and z == '<sep>') or (no_unk and z == '<unk>') or (no_num and z == '<num>') or (no_prp and z == '<prp>'):
			return x
		else:
			return y

	def __call__(self, xs, as_str=False, join=False, no_sep=False, no_unk=False, no_num=False, no_prp=False):
		xs = xs.strip()
		xs = re.sub(r'([.!?:])', ' \\1 ', xs)
		xs = xs.split()

		xs = [(x, self.convert(x)) for x in xs]
		xs = [(x, y) for x, y in xs if y is not None]

		if as_str:
			xs = [self.choose(x, y, no_sep, no_unk, no_num, no_prp) for x, y in xs]
		else:
			xs = [y for x, y in xs]
		return xs

