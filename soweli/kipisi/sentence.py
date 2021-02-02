class Sentence(list):
    sep_list = ['.', '!', '?', ':']
    sep_dict = {x:n for n, x in enumerate(sep_list)}
    tokenizer = None

    def __init__(self, x):
        sent = self.tokenizer(x)
        src_sent = self.tokenizer(x, as_str = True, no_sep = True)
        target = []
        for x, y in zip(sent, src_sent):
            if x == self.tokenizer.vocab.sep_id:
                elem = self.sep_dict[y]
            else:
                elem = None
            target.append(elem)
        super().__init__(sent)
        self.target = target

    def joined(self):
        lst = []
        for x, y in zip(self, self.target):
            if y is None:
                elem = self.tokenizer.vocab[x]
            else:
                elem = self.sep_list[y]
            lst.append(elem)
        return ' '.join(lst)

