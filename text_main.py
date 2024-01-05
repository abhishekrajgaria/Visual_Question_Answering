import torch
import time
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from transformers import BertModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from text_model import TextModel
from config import Config
from dataset_util import *

config = Config()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available")

best_perf_dict = {
    "acc_metric": -1,
    "bert_model_param": None,
    "text_model_param": None,
    "optim_param": None,
    "epoch": 0,
    "learning_rate": 0,
    "output_filename": "",
}

parser = argparse.ArgumentParser()

args = parser.parse_args()


def loss_plot_train_val(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("text_loss_plot.png")
    plt.show()


def acc_plot_train_val(train_acc, val_acc):
    plt.figure()
    plt.plot(train_acc, label="Training Acc.", color="blue")
    plt.plot(val_acc, label="Validation Acc.", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Acc.")
    plt.title("Training and Validation Acc. Over Epochs")
    plt.legend()
    plt.savefig("text_acc_plot.png")
    plt.show()


class Trainer:
    def __init__(self, epochs, batch_size, train_dataset, val_dataset, weights):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.weights = weights

    def train(self, learning_rate):
        loss_function = nn.CrossEntropyLoss(weight=self.weights).to(device)
        bert_model = BertModel.from_pretrained(config.bert_model_path)
        text_model = TextModel(bert_model).to(device)

        optimizer = optim.Adam(params=text_model.parameters(), lr=learning_rate)

        train_dataLoader = torch.utils.data.DataLoader(
            self.train_dataset, self.batch_size, shuffle=True
        )
        val_dataLoader = torch.utils.data.DataLoader(
            self.val_dataset, self.batch_size, shuffle=True
        )
        e_train_loss = []
        e_val_loss = []
        e_train_acc = []
        e_val_acc = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            train_loss = []
            train_pred = []
            train_gold = []

            for batch in tqdm(train_dataLoader):
                text_model.train()
                optimizer.zero_grad()
                (
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    tokenized_label,
                    _,
                    tokenized_answer_id,
                ) = batch

                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)

                model_input = (input_ids, token_type_ids, attention_mask)
                output = text_model(model_input)
                loss = loss_function(output, tokenized_label.to(device))

                loss.backward()

                optimizer.step()
                train_loss.append(loss.cpu().item())

                preds = torch.argmax(output, dim=1)
                train_pred.extend(preds.cpu().tolist())
                train_gold.extend(tokenized_label.tolist())
                torch.cuda.empty_cache()

            e_train_loss.append(np.mean(train_loss))
            print(f"Average training batch loss: {np.mean(train_loss)}")
            train_acc = accuracy_score(train_gold, train_pred)
            e_train_acc.append(train_acc)
            print(f"Training accuracy score: {train_acc}")

            val_loss = []
            val_pred = []
            val_gold = []

            for batch in val_dataLoader:
                (
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    tokenized_label,
                    _,
                    tokenized_answer_id,
                ) = batch

                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                text_model.eval()

                with torch.no_grad():
                    model_input = (input_ids, token_type_ids, attention_mask)

                    output = text_model(model_input)
                    loss = loss_function(output, tokenized_label.to(device))

                    val_loss.append(loss.cpu().item())
                    preds = torch.argmax(output, dim=1)
                    val_pred.extend(preds.cpu().tolist())
                    val_gold.extend(tokenized_label.tolist())

            e_val_loss.append(np.mean(val_loss))
            print(f"Average validation batch loss: {np.mean(val_loss)}")
            val_acc = accuracy_score(val_gold, val_pred)
            e_val_acc.append(val_acc)
            print(f"Validation accuracy score: {val_acc}")

            output_filename = (
                f"{config.project_storage_path}/models/text/{learning_rate}_{epoch}"
            )

            if (
                best_perf_dict["acc_metric"] == -1
                or val_acc > best_perf_dict["acc_metric"]
            ):
                best_perf_dict["acc_metric"] = val_acc
                best_perf_dict["epoch"] = epoch
                best_perf_dict["learning_rate"] = learning_rate
                best_perf_dict["output_filename"] = output_filename

            torch.save(
                {
                    "bert_model_param": bert_model.state_dict(),
                    "text_model_param": text_model.state_dict(),
                    "optim_param": optimizer.state_dict(),
                    "acc_metric": val_acc,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                },
                output_filename,
            )
        loss_plot_train_val(e_train_loss, e_val_loss)
        acc_plot_train_val(e_train_acc, e_val_acc)


def loadModel(model_instance, model_path):
    checkpoint = torch.load(model_path)
    # print(checkpoint)
    model_instance.load_state_dict(checkpoint["text_model_param"])
    print(
        f"""Val_Acc of loaded model: {checkpoint["acc_metric"]} at epoch {checkpoint["epoch"]} with learning rate {checkpoint["learning_rate"]}"""
    )
    return model_instance


def evalute_test():
    test_dataset = load_pickle_data(config.test_dataset_path_small)
    test_dataLoader = torch.utils.data.DataLoader(test_dataset, 1024, shuffle=True)
    bert_model = BertModel.from_pretrained(config.bert_model_path)
    text_model = TextModel(bert_model).to(device)

    print("output_file --", best_perf_dict["output_filename"])

    best_text_model = loadModel(text_model, best_perf_dict["output_filename"])

    test_pred = []
    test_gold = []
    test_ids = []
    best_text_model = best_text_model.to(device)

    for batch in test_dataLoader:
        (
            input_ids,
            token_type_ids,
            attention_mask,
            tokenized_label,
            _,
            tokenized_answer_id,
        ) = batch

        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        best_text_model.eval()

        with torch.no_grad():
            model_input = (input_ids, token_type_ids, attention_mask)

            output = best_text_model(model_input)

            preds = torch.argmax(output, dim=1)
            test_pred.extend(preds.cpu().tolist())
            test_gold.extend(tokenized_label.tolist())
            test_ids.extend(tokenized_answer_id.tolist())

    test_acc = accuracy_score(test_gold, test_pred)
    print(f"Test accuracy score: {test_acc}")

    prediction = (test_pred, test_gold, test_ids)
    save_pickle_data(prediction, config.text_prediction_data)


if __name__ == "__main__":
    torch.manual_seed(42)
    learning_rates = [0.001]

    train_dataset = load_pickle_data(config.train_dataset_path_small)
    # train_image_dict = get_image_dict("train")

    val_dataset = load_pickle_data(config.val_dataset_path_small)
    # val_image_dict = get_image_dict("val")

    weights = get_class_weights()

    for learning_rate in learning_rates:
        print(f"learning_rate {learning_rate}")
        trainer = Trainer(
            epochs=20,
            batch_size=1024,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            weights=weights,
        )
        trainer.train(learning_rate)

    print(
        f"""\nBest Val performance of {best_perf_dict["acc_metric"]} at epoch {best_perf_dict["epoch"]} for learning_rate {best_perf_dict["learning_rate"]}"""
    )

    evalute_test()
