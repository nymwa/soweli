import random as rd
import numpy as np
import torch
from soweli.ponart.util import make_tokenizer
from soweli.ponart.ponart import Ponart

class PonartWrapper:
    def __init__(self, checkpoint_path = None):
        self.tokenizer = make_tokenizer()
        self.ponart = Ponart(len(self.tokenizer.vocab))
        if checkpoint_path:
            self.ponart.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
        self.ponart.eval()

    def cuda(self):
        self.ponart = self.ponart.cuda()

    def mask_one(self, seq, n):
        seq = [w if i != n else self.tokenizer.vocab.msk_id for i, w in enumerate(seq)]
        return seq

    def seq_to_tensor(self, seq):
        x = [self.tokenizer.vocab.cls_id] + seq
        x = torch.tensor([x]).T
        x = x.to(self.ponart.fc.weight.device)
        return x

    def calc_logit(self, ten, n, temp):
        with torch.no_grad():
            logit = self.ponart(ten)[n + 1, 0]
        return logit / temp

    def top_p_sampling(self, logit, p):
        logit[[
            self.tokenizer.vocab.pad_id,
            self.tokenizer.vocab.cls_id,
            self.tokenizer.vocab.msk_id,
            self.tokenizer.vocab.unk_id]] = -float('Inf')
        values, indices = torch.sort(torch.softmax(logit, dim=-1))
        is_removed = torch.cumsum(values, dim=-1) < (1 - p)
        logit[indices[is_removed]] = -float('Inf')
        probs = torch.softmax(logit, dim=-1)
        next_token = np.random.choice(range(len(self.tokenizer.vocab)), p=probs.cpu().numpy())
        return next_token

    def sample_pos(self, seq):
        mask_pos_list = [i for i, w in enumerate(seq) if w == self.tokenizer.vocab.msk_id]
        if mask_pos_list:
            n = rd.choice(mask_pos_list)
        else:
            n = rd.randrange(len(seq))
        return n

    def initial_seq(self, seq_len=None):
        if seq_len is None:
            seq_len = rd.randrange(2, 16)
        seq = [self.tokenizer.vocab.msk_id for _ in range(seq_len)]
        return seq

    def renew_seq(self, seq, temp, p):
        n = self.sample_pos(seq)
        seq = self.mask_one(seq, n)
        x = self.seq_to_tensor(seq)
        logit = self.calc_logit(x, n, temp)
        seq[n] = self.top_p_sampling(logit, p)
        return seq

    def sample_seq(self, num_iter = 100, temp = 2.0, p = 0.3, seq_len = None):
        seq = self.initial_seq(seq_len)
        for _ in range(num_iter):
            seq = self.renew_seq(seq, temp, p)
        return seq

    def show_seq(self, seq):
        print(' '.join([self.tokenizer.vocab[w] for w in seq]))

    def calc_log_prob(self, seq, n):
        masked_seq = self.mask_one(seq, n)
        x = self.seq_to_tensor(masked_seq)
        log_prob = torch.log_softmax(self.ponart(x)[n + 1, 0], dim=-1)[seq[n]]
        return log_prob.item()

    def calc_score(self, seq):
        log_probs = [self.calc_log_prob(seq, n) for n in range(len(seq))]
        mean_log_prob = np.mean(log_probs)
        mean_prob = np.exp(mean_log_prob)
        return mean_prob

