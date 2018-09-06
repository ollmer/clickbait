import numpy as np
import sentencepiece as spm
import torch as t
from tqdm import tqdm


class MySentences:
    def __init__(self, fname):
        self.corpus = []
        with open(fname) as f:
            for i, line in enumerate(tqdm(f)):
                if len(line) < 20:
                    continue
                self.corpus.append(line.rstrip('\n'))

    def __len__(self):
        return len(self.corpus)

    def __iter__(self):
        for line in tqdm(self.corpus):
            yield line


class Dataloader:
    def __init__(self, data_path=''):
        """
        :param data_path: path to data
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.targets = ['train', 'test']

        self.data = {
            target: MySentences('data/opensub_%s.txt' % target).corpus
            for target in self.targets
        }
        print('data loaded', [len(d) for d in self.data.values()])

        self.sp = spm.SentencePieceProcessor()
        bpefile = '{}bpe/opensub10k_bpe.model'.format(data_path)
        self.sp.Load(bpefile)
        print('loaded bpe', bpefile)

        '''
        Actually, max length is lower than this value
        '''
        self.max_len = max([len(line) for tar in self.targets for line in self.data[tar]])
        print('max len found', self.max_len)

    def next_batch(self, batch_size, target, device):
        data = self.data[target]

        input = [data[np.random.randint(len(data))] for _ in range(batch_size)]

        input = [[1] + self.sp.EncodeAsIds(line) + [2] for line in input]

        target = self.padd_sequences([line[1:] for line in input])
        input = self.padd_sequences([line[:-1] for line in input])

        return tuple([t.tensor(i, dtype=t.long, device=device) for i in [input, target]])

    @staticmethod
    def padd_sequences(lines):
        lengths = [len(line) for line in lines]
        max_length = max(lengths)

        return np.array([line + [0] * (max_length - lengths[i]) for i, line in enumerate(lines)])
