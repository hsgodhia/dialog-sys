import os, torch, numpy as np, pdb, pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler


class MovieTriples(Dataset):
    def __init__(self, data_type):
        # we use a common dict for all test, train and validation
        _dict_file = '/home/harshal/code/research/untitled/data/MovieTriples_Dataset/Training.dict.pkl'

        if data_type == 'train':
            _file = '/home/harshal/code/research/untitled/data/MovieTriples_Dataset/Training.triples.pkl'
        elif data_type == 'valid':
            _file = '/home/harshal/code/research/untitled/data/MovieTriples_Dataset/Validation.triples.pkl'
        elif data_type == 'test':
            _file = '/home/harshal/code/research/untitled/data/MovieTriples_Dataset/Test.triples.pkl'

        with open(_file, 'rb') as fp:
            self.data = pickle.load(fp)

        with open(_dict_file, 'rb') as fp2:
            self.dict_data = pickle.load(fp2)
        # dictionary data is like ('</s>', 2, 588827, 785135)
        # so i believe that the first is the ids are assigned by frequency
        # thinking to use a counter collection out here maybe
        self.inv_dict = {}
        self.dict = {}
        for x in self.dict_data:
            tok, f, _, _ = x
            self.dict[tok] = f
            self.inv_dict[f] = tok

        self.sentences = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])

# encode each sentence utterance into a single vector
class BaseEncoder(nn.Module):
    def __init__(self):
        self.rnn = nn.GRU(hidden_size=1000, num_layers=1, bidirectional=False, batch_first=True)

# encode the hidden states of a number of utterances
class SessionEncoder(nn.Module):
    def __init__(self):
        self.rnn = nn.GRU(hidden_size=1500, num_layers=1, bidirectional=False, batch_first=True)

# decode the hidden state
class Decoder(nn.Module):
    def __init__(self):
        pass


def main():
    BATCH_SIZE, train_dataset = 1, MovieTriples(data_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    for i_batch, sample_batch in enumerate(train_dataloader):
        print(sample_batch)
        break
        # 1 * 33 dimensional input

main()
