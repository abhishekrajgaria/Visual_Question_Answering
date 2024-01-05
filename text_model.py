import torch
import torch.nn as nn

from config import Config

config = Config()


class TextModel(nn.Module):
    def __init__(self, bert_model):
        super(TextModel, self).__init__()
        self.bert_model = bert_model
        self.cls_dim = config.bert_cls_emb_dim
        self.W = nn.Linear(self.cls_dim, config.num_answer_class, bias=True)

    def forward(self, input):
        input_ids, token_ids, attention_mask = input
        bert_output = self.bert_model(
            input_ids=input_ids, token_type_ids=token_ids, attention_mask=attention_mask
        )
        bert_output = bert_output.pooler_output
        # print(bert_output.shape)
        return self.W(bert_output)


if __name__ == "__main__":
    model = TextModel()
