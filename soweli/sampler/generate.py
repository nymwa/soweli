from .model import PonartWrapper
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('-i', type=int, default=30)
    parser.add_argument('-t', type=float, default=1.0)
    parser.add_argument('-p', type=float, default=0.5)
    parser.add_argument('-l', type=int, default=None)

    args = parser.parse_args()

    model = PonartWrapper(args.checkpoint_path)
    model.cuda()

    for _ in range(args.n):
        seq = model.sample_seq(num_iter=args.i, temp=args.t, p=args.p, seq_len=args.l)
        model.show_seq(seq)

