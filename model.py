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
