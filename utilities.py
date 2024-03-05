import re
import torch
from typing import Sequence
from functools import partial
import random
import torch
import numpy as np
import random
import re

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torcheval.metrics import R2Score

EMBEDDING_DIM = 8
LSTM_HIDDEN = 16
LSTM_LAYER = 1

alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}
int2dna = { i: a for a, i in zip(alphabet, range(5))}
intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

REGEX_PATTERN = '(?=(CG))'

def process_text_regex(text):
    return len(re.findall(REGEX_PATTERN, text))


class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.embedding = torch.nn.Embedding(num_embeddings=len(alphabet), embedding_dim=EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(
            input_size=EMBEDDING_DIM, hidden_size=LSTM_HIDDEN, 
            num_layers=LSTM_LAYER, batch_first=True, bidirectional=True,
            dropout= 0.3 if LSTM_LAYER > 1 else 0
        )
        self.classifier = torch.nn.Linear(2*LSTM_HIDDEN, 1)
        
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        
        embedded = self.dropout(self.embedding(x))
        # embedded = [batch, seq_len, embedding_dim]
        
        _, (hn, _) = self.lstm(embedded)
        # hn, cn = [num_layers*directions, batch, hidden_dim]

        stacked = torch.cat([hn[-1, :, :], hn[-2, :, :]], dim=1).squeeze(0) # taking the foward and backward h from the last layer
        logits = self.classifier(self.dropout(F.relu(stacked)))
        
        return logits
    

def load_model():
    model = CpGPredictor()
    # print(model)
    model.load_state_dict(torch.load('model_1.pt', map_location=torch.device('cpu')))
    return model


def get_prediction(model, dna_seq):
    model.eval()
    print(dna_seq)
    int_seq = list(dnaseq_to_intseq(dna_seq))
    input_tensor = torch.LongTensor([int_seq])
    print(input_tensor)
    with torch.no_grad():
        out = model(input_tensor)
    print(out)
    num_output_cgs = round(out.detach().tolist()[0])
    print(num_output_cgs)
    return num_output_cgs

