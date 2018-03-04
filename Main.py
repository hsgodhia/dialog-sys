import os, torch, numpy as np, pdb, pickle, copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler


def cmp_dialog(d1, d2):
    if len(d1) < len(d2):
        return -1
    elif len(d2) > len(d1):
        return 1
    else:
        return 0


class DialogTurn():
    def __init__(self, item):
        cur_list, i = [], 0
        for d in item:
            cur_list.append(d)
            if d == 2:
                if i == 0:
                    self.u1 = copy.copy(cur_list)
                    cur_list[:] = []
                elif i == 1:
                    self.u2 = copy.copy(cur_list)
                    cur_list[:] = []
                else:
                    self.u3 = copy.copy(cur_list)
                    cur_list[:] = []
                i += 1

    def __len__(self):
        return len(self.u1) + len(self.u2) + len(self.u3)

    def __repr__(self):
        return str(self.u1 + self.u2 + self.u3)


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
        self.utterance_data = []

        with open(_file, 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                self.utterance_data.append(DialogTurn(d))
        self.utterance_data.sort(cmp=cmp_dialog)
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

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        return self.utterance_data[idx]


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
        self.rnn = nn.GRU(hidden_size=1000, num_layers=1, bidirectional=False, batch_first=True)


def custom_collate_fn(batch):
    u1_batch, u2_batch, u3_batch, cur_batch = [], [], [], []
    l_u1, l_u2, l_u3 = 0, 0, 0
    for d in batch:
        if len(d.u1) != l_u1:
            if len(cur_batch) > 0:
                u1_batch.append(torch.stack(cur_batch, 0))
                cur_batch[:] = []

        l_u1 = len(d.u1)
        cur_batch.append(torch.LongTensor(d.u1))
    if len(cur_batch) > 0:
        u1_batch.append(torch.stack(cur_batch, 0))
    cur_batch[:] = []

    for d in batch:
        if len(d.u2) != l_u2:
            if len(cur_batch) > 0:
                u2_batch.append(torch.stack(cur_batch, 0))
                cur_batch[:] = []

        l_u2 = len(d.u2)
        cur_batch.append(torch.LongTensor(d.u2))

    if len(cur_batch) > 0:
        u2_batch.append(torch.stack(cur_batch, 0))
    cur_batch[:] = []

    for d in batch:
        if len(d.u3) != l_u3:
            if len(cur_batch) > 0:
                u3_batch.append(torch.stack(cur_batch, 0))
                cur_batch[:] = []

        l_u3 = len(d.u3)
        cur_batch.append(torch.LongTensor(d.u3))

    if len(cur_batch) > 0:
        u3_batch.append(torch.stack(cur_batch, 0))

    return u1_batch, u2_batch, u3_batch


def main():
    BATCH_SIZE, train_dataset = 20, MovieTriples(data_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    for i_batch, sample_batch in enumerate(train_dataloader):
        print(sample_batch[0])
        break
        # 1 * 33 dimensional input

main()
