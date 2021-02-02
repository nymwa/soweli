import random as rd
import numpy as np
import torch
from soweli.ponart.util import make_tokenizer
from soweli.kipisi.model import KipisiClassifier

class KipisiWrappen:
    def __init__(self, checkpoint_path = None):
        self.tokenizer = make_tokenizer()
        self.kipisi = KipisiClassifier(len(self.tokenizer.vocab))
        if checkpoint_path:
            self.kipisi.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
        self.kipisi.eval()

    def cuda(self):
        self.kipisi = self.kipisi.cuda()

    def seq_to_tensor(self, seq):
        x = [self.tokenizer.vocab.cls_id] + seq
        x = torch.tensor([x]).T
        x = x.to(self.ponart.fc.weight.device)
        return x

    def infer(self, seq):
        x = seq_to_tensor(seq)
        with torch.no_grad():
            x = model(x)
        x = x.argmax(dim = -1).squeeze(-1).cpu().numpy()[1:]
        return x

