import sys
from .tokenizer import IloTunimi 
from argparse import ArgumentParser

def main():
	parser = ArgumentParser()
	parser.add_argument('--as-int', action = 'store_true')
	parser.add_argument('--no-sep', action = 'store_true')
	parser.add_argument('--no-unk', action = 'store_true')
	parser.add_argument('--no-num', action = 'store_true')
	parser.add_argument('--no-prp', action = 'store_true')
	parser.add_argument('--no-join', action = 'store_true')
	args = parser.parse_args()

	tokenizer = IloTunimi()
	for xs in sys.stdin:
		print(tokenizer(xs, as_str=not args.as_int, no_sep=args.no_sep, no_unk=args.no_unk, no_num=args.no_num, no_prp=args.no_prp, join=not args.no_join))

