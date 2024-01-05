import torch
import torch.nn as nn

from config import Config

config = Config()


class ClassifierViltModel(nn.Module):
    def __init__(self, model):
        super(ClassifierViltModel, self).__init__()
        self.vilt_model = model
        self.vlt_emb_dim = config.vlt_emb_dim

        # for param in self.vilt_model.parameters():
        #     param.requires_grad = False

        self.W = nn.Linear(self.vlt_emb_dim, config.num_answer_class, bias=True)

    def forward(self, input):
        # print(input)
        vilt_output = self.vilt_model(**input)
        vilt_output = vilt_output.pooler_output
        # print(vilt_output.shape)
        return self.W(vilt_output)


if __name__ == "__main__":
    model = ClassifierViltModel()
