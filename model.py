import torch
import torch.nn as nn

import transformers

class Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self, num_class):
        super(Bert_Classification_Model, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.out = nn.Linear(768, num_class)

    def forward(self, ids, mask, token_type_ids):

        output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)

        return self.out(output['pooler_output'])

    
class RNNOverBERT(nn.Module):

    def __init__(self, bertFineTuned, num_class, hidden_dim=100, yers=1, bidirectional=False):
        super(RNNOverBERT, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.bertFineTuned = bertFineTuned
        
        self.lstm = nn.LSTM(768, hidden_dim, num_layers=1, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_dim, num_class)

    def forward(self, ids, mask, token_type_ids, lengt):

        output = self.bertFineTuned(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = output['pooler_output'].split_with_sizes(lengt)

        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        batch_emb_pad = nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=-91, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input, )  # (h_t, h_c))

        h_t = h_t.view(-1, self.hidden_dim)

        return self.out(h_t)
