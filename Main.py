import torch, pickle, copy, argparse, time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


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

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        return self.utterance_data[idx]


# encode each sentence utterance into a single vector
class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, num_lyr, bidi):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.direction = 2 if bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(dropout=0.2, bias=False, input_size=emb_size, hidden_size=hid_size,
                          num_layers=num_lyr, bidirectional=bidi, batch_first=True)

    def forward(self, inp_batches):
        # here input is a list of batches that are of the same size so no padding needed
        output = []
        for x in inp_batches:
            h_0 = Variable(torch.zeros(self.direction * self.num_lyr, x.size(0), self.hid_size))
            if use_cuda:
                x = x.cuda()
                h_0 = h_0.cuda()

            x_emb = self.embed(x)
            _, x_hid = self.rnn(x_emb, h_0)
            # move the batch to the front of the tensor
            x_hid = x_hid.view(x.size(0), -1, self.hid_size)

            output.append(x_hid)
        output = torch.cat(output, 0)
        return output


# encode the hidden states of a number of utterances
class SessionEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, num_lyr, bidi):
        super(SessionEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.direction = 2 if bidi else 1
        self.rnn = nn.GRU(dropout=0.2, hidden_size=hid_size, input_size=inp_size,
                          num_layers=num_lyr, bidirectional=bidi, batch_first=True)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, x.size(0), self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()

        # output, h_n for output batch is already dim 0
        _, o = self.rnn(x, h_0)
        # move the batch to the front of the tensor
        o = o.view(x.size(0), -1, self.hid_size)
        return o


# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, ses_hid_size, hid_size, num_lyr=1, bidi=False, teacher=True):
        super(Decoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.embed = nn.Embedding(vocab_size, emb_size) # currently the output embedding doesn't share weight
        self.direction = 2 if bidi else 1
        self.lin1 = nn.Linear(ses_hid_size, hid_size)
        self.tanh = nn.Tanh()
        self.rnn = nn.GRU(dropout=0.2, hidden_size=hid_size, input_size=emb_size,
                          num_layers=num_lyr, bidirectional=False, batch_first=True)
        self.lin2 = nn.Linear(hid_size, vocab_size)
        self.log_soft = nn.LogSoftmax(dim=2)
        self.loss_cri = nn.NLLLoss()
        self.teacher_forcing = teacher

    def forward(self, ses_encoding, inp_batches=None):
        # this indicates inference mode
        if inp_batches is None:
            # todo assume batch size 1 for this method
            # we use this for inference when the only input is the ses_encoding
            ses_encoding = self.tanh(self.lin1(ses_encoding))
            siz = ses_encoding.size(0)
            # in each iteration the generated sentence's length increases by 1 so loop terminates
            gen_len, sent, hid_n = 0, [], ses_encoding
            # the start token has index 1

            tok = Variable(torch.ones(siz, 1).long())
            sent.append(tok.data[0, 0])
            if use_cuda:
                tok = tok.cuda()

            while True:
                if gen_len > 50 or tok.data[0, 0] == 2:
                    break
                tok_vec = self.embed(tok)
                hid_n, _ = self.rnn(tok_vec, hid_n)
                op = self.lin2(hid_n)
                op = self.log_soft(op)
                op = op.squeeze(1)
                tok_val, tok = torch.max(op, dim=1, keepdim=True)
                sent.append(tok.data[0, 0])
                gen_len += 1

            return sent

        else:
            # would have to do NLL loss here itself
            ses_encoding = self.tanh(self.lin1(ses_encoding))

            c_siz, loss = 0, 0
            for x in inp_batches:
                if use_cuda:
                    x = x.cuda()
                siz, seq_len = x.size(0), x.size(1)
                x_emb = self.embed(x)

                sub_ses_encoding = ses_encoding[c_siz: (c_siz + siz), :, :]
                sub_ses_encoding = sub_ses_encoding.view(self.num_lyr*self.direction, siz, self.hid_size)

                if not self.teacher_forcing:
                    # start of sentence is the first tok
                    tok = x_emb[:, 0, :]
                    tok = tok.unsqueeze(1)
                    hid_n = sub_ses_encoding

                    for i in range(seq_len):
                        hid_o, hid_n = self.rnn(tok, hid_n)
                        # hid_o (seq_len, batch, hidden_size * num_directions) batch first affects this
                        # hid_n (num_layers * num_directions, batch, hidden_size)  batch first doesn't affect
                        # h_0 (num_layers * num_directions, batch, hidden_size) batch first doesn't affect
                        op = self.lin2(hid_o)
                        op = self.log_soft(op)
                        op = op.squeeze(1)
                        if i+1 < seq_len:
                            loss += self.loss_cri(op, x[:, i+1])
                            _, tok = torch.max(op, dim=1, keepdim=True)
                            tok = self.embed(tok)
                else:
                    # I'm directly doing teacher forcing here by feeding the true sequence via embedding layer
                    dec_ts, dec_o = self.rnn(x_emb, sub_ses_encoding)
                    # dec_ts is of size (batch, seq_len, hidden_size * num_directions)

                    # got a input is not contiguous error above
                    dec_ts = self.lin2(dec_ts)
                    dec_ts = self.log_soft(dec_ts)

                    # here the dimension is N*SEQ_LEN*VOCAB_SIZE
                    for i in range(seq_len):
                        loss += self.loss_cri(dec_ts[:, i, :], x[:, i])

                c_siz += siz

            return loss

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val


def custom_collate_fn(batch):
    u1_batch, u2_batch, u3_batch, cur_batch = [], [], [], []
    l_u1, l_u2, l_u3 = 0, 0, 0
    for d in batch:
        if len(d.u1) != l_u1:
            if len(cur_batch) > 0:
                u1_batch.append(Variable(torch.stack(cur_batch, 0)))
                cur_batch[:] = []

        l_u1 = len(d.u1)
        cur_batch.append(torch.LongTensor(d.u1))

    if len(cur_batch) > 0:
        u1_batch.append(Variable(torch.stack(cur_batch, 0)))
    cur_batch[:] = []

    for d in batch:
        if len(d.u2) != l_u2:
            if len(cur_batch) > 0:
                u2_batch.append(Variable(torch.stack(cur_batch, 0)))
                cur_batch[:] = []

        l_u2 = len(d.u2)
        cur_batch.append(torch.LongTensor(d.u2))

    if len(cur_batch) > 0:
        u2_batch.append(Variable(torch.stack(cur_batch, 0)))
    cur_batch[:] = []

    for d in batch:
        if len(d.u3) != l_u3:
            if len(cur_batch) > 0:
                u3_batch.append(Variable(torch.stack(cur_batch, 0)))
                cur_batch[:] = []

        l_u3 = len(d.u3)
        cur_batch.append(torch.LongTensor(d.u3))

    if len(cur_batch) > 0:
        u3_batch.append(Variable(torch.stack(cur_batch, 0)))

    return u1_batch, u2_batch, u3_batch


def calc_valid_loss(base_enc, ses_enc, dec):
    base_enc.eval()
    ses_enc.eval()
    dec.eval()
    dec.set_teacher_forcing(False)

    bt_siz, valid_dataset = 32, MovieTriples(data_type='valid')[:32]
    valid_dataloader = DataLoader(valid_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)

    valid_loss = 0
    for i_batch, sample_batch in enumerate(valid_dataloader):
        u1, u2, u3 = sample_batch[0], sample_batch[1], sample_batch[2]
        o1, o2 = base_enc(u1), base_enc(u2)
        qu_seq = torch.cat((o1, o2), 1)
        final_session_o = ses_enc(qu_seq)
        loss = dec(final_session_o, u3)
        valid_loss += loss.data[0]

    return valid_loss/(1 + i_batch)


def train(options, base_enc, ses_enc, dec):
    base_enc.train()
    ses_enc.train()
    dec.train()

    all_params = list(base_enc.parameters()) + list(ses_enc.parameters()) + list(dec.parameters())
    # init parameters
    for name, param in base_enc.named_parameters():
        if name.startswith('rnn') and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)

    bt_siz, train_dataset = 32, MovieTriples(data_type='train')[:32]
    train_dataloader = DataLoader(train_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)
    optimizer = optim.Adam(all_params)

    for i in range(options.e):
        tr_loss = 0
        strt = time.time()
        for i_batch, sample_batch in enumerate(train_dataloader):
            u1, u2, u3 = sample_batch[0], sample_batch[1], sample_batch[2]
            o1, o2 = base_enc(u1), base_enc(u2)
            qu_seq = torch.cat((o1, o2), 1)

            # if we need to decode the intermediate queries we may need the hidden states
            final_session_o = ses_enc(qu_seq)

            loss = dec(final_session_o, u3)
            tr_loss += loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('done', i_batch)

        vl_loss = calc_valid_loss(base_enc, ses_enc, dec)
        print("Valid loss", vl_loss)
        print("Training loss", tr_loss/(1 + i_batch))
        print("epoch took", (time.time() - strt)/3600.0)
        if i_batch%2 == 0:
            torch.save(base_enc.state_dict(), 'enc_mdl.pth')
            torch.save(ses_enc.state_dict(), 'ses_mdl.pth')
            torch.save(dec.state_dict(), 'dec_mdl.pth')
            torch.save(optimizer.state_dict(), 'opti_st.pth')


def main():
    # we use a common dict for all test, train and validation
    _dict_file = '/home/harshal/code/research/untitled/data/MovieTriples_Dataset/Training.dict.pkl'
    with open(_dict_file, 'rb') as fp2:
        dict_data = pickle.load(fp2)
    # dictionary data is like ('</s>', 2, 588827, 785135)
    # so i believe that the first is the ids are assigned by frequency
    # thinking to use a counter collection out here maybe
    inv_dict = {}
    dict = {}
    for x in dict_data:
        tok, f, _, _ = x
        dict[tok] = f
        inv_dict[f] = tok

    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-e', dest='e', type=int, default=10, help='number of epochs')
    options = parser.parse_args()

    base_enc = BaseEncoder(10003, 300, 1000, 1, False)
    ses_enc = SessionEncoder(1500, 1000, 1, False)
    dec = Decoder(10003, 300, 1500, 1000, 1, False, False)
    if use_cuda:
        base_enc.cuda()
        ses_enc.cuda()
        dec.cuda()

    train(options, base_enc, ses_enc, dec)
    # inference_beam(base_enc, ses_enc, dec, inv_dict)


def tensor_to_sent(x, inv_dict):
    sent = []
    for i in x:
        sent.append(inv_dict[i])

    return " ".join(sent)


# sample a sentence from the test set by using beam search
# todo currently does greedy modify to do beam
def inference_beam(base_enc, ses_enc, dec, inv_dict, width=5):
    saved_state = torch.load("enc_mdl.pth")
    base_enc.load_state_dict(saved_state)

    saved_state = torch.load("ses_mdl.pth")
    ses_enc.load_state_dict(saved_state)

    saved_state = torch.load("dec_mdl.pth")
    dec.load_state_dict(saved_state)

    base_enc.eval()
    ses_enc.eval()
    dec.eval()

    # todo not sure how to do inference for batch size > 1
    bt_siz, test_dataset = 1, MovieTriples(data_type='test')[:10]
    test_dataloader = DataLoader(test_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)

    for i_batch, sample_batch in enumerate(test_dataloader):
        u1, u2 = sample_batch[0], sample_batch[1]
        o1, o2 = base_enc(u1), base_enc(u2)
        qu_seq = torch.cat((o1, o2), 1)

        # if we need to decode the intermediate queries we may need the hidden states
        final_session_o = ses_enc(qu_seq)

        sent = dec(final_session_o)
        print(tensor_to_sent(sent, inv_dict))


main()
