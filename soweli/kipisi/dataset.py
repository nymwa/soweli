import torch
from soweli.ponart.batch import Batch
from torch.nn.utils.rnn import pad_sequence as pad

class Dataset(list, torch.utils.data.Dataset):
    def __init__(self, sents, vocab):
        super().__init__(sents)
        self.vocab = vocab
        self.pad = self.vocab.pad_id

    def collate(self, batch):
        ei = [torch.tensor(sent) for sent in batch]
        eo = [torch.tensor([x if x is not None else -100 for x in sent.target]) for sent in batch]
        el = torch.tensor([len(sent) for sent in batch])
        ei = pad(ei, padding_value = self.pad)
        eo = pad(eo, padding_value = -100)
        return Batch(ei, eo, el)

