import torch
import torch.nn as nn

from config import Config

config = Config()


class ImageTextModel(nn.Module):
    def __init__(self, bert_model, resnet_model):
        super(ImageTextModel, self).__init__()
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.cls_dim = config.bert_cls_emb_dim

        # for param in self.resnet_model.parameters():
        #     param.requires_grad = False

        # self.layers = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=5, padding=2),  # Initial conv layer
        #     nn.ReLU(),  # Activation
        #     nn.MaxPool2d(2),  # Downsampling
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Deeper conv layer
        #     nn.ReLU(),  # Activation
        #     # nn.MaxPool2d(2),  # Further downsampling
        #     # nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Final conv layer
        #     # nn.ReLU(),  # Activation
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        # )

        self.W = nn.Linear(2 * self.cls_dim, config.num_answer_class, bias=True)
        self.ResW = nn.Linear(config.res_pooler_dim, self.cls_dim, bias=True)

    def forward(self, input):
        input_ids, token_ids, attention_mask, input_images = input
        bert_output = self.bert_model(
            input_ids=input_ids, token_type_ids=token_ids, attention_mask=attention_mask
        )
        bert_output = bert_output.pooler_output

        # print("hello")
        resnet_output = self.resnet_model(pixel_values=input_images)
        resnet_output = resnet_output.pooler_output.view(-1, config.res_pooler_dim)

        # print(bert_output.shape)
        # print(resnet_output.shape)
        # cnn_output = self.layers(input_images).view(512, 128)

        # print(cnn_output.shape)

        resnet_output = self.ResW(resnet_output)
        output = torch.cat((resnet_output, bert_output), dim=1)
        return self.W(output)


if __name__ == "__main__":
    model = ImageTextModel()
