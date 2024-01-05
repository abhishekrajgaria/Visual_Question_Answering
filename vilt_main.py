import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from transformers import ViltProcessor, ViltModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from vilt_model import ClassifierViltModel
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
    "resnet_model_param": None,
    "vilt_model_param": None,
    "optim_param": None,
    "epoch": 0,
    "learning_rate": 0,
    "output_filename": "",
}

parser = argparse.ArgumentParser()

args = parser.parse_args()

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
train_ques_dict = load_pickle_data(config.train_ques_dict_path)
val_ques_dict = load_pickle_data(config.val_ques_dict_path)
test_ques_dict = load_pickle_data(config.test_ques_dict_path)


def loss_plot_train_val(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("vilt_loss_plot.png")
    plt.show()


def acc_plot_train_val(train_acc, val_acc):
    plt.figure()
    plt.plot(train_acc, label="Training Acc.", color="blue")
    plt.plot(val_acc, label="Validation Acc.", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Acc.")
    plt.title("Training and Validation Acc. Over Epochs")
    plt.legend()
    plt.savefig("vilt_acc_plot.png")
    plt.show()


def get_process_data(tensored_image_ids, image_path_prefix, tensored_answer_ids, split):
    dict = {}
    if split == "train":
        dict = train_ques_dict
    elif split == "val":
        dict = val_ques_dict
    else:
        dict = test_ques_dict
    images = []
    quess = []
    for ele in tensored_image_ids.tolist():
        image_id = str(ele)
        image_id_len = len(image_id)
        image_id_prfxd = "0" * (12 - image_id_len) + image_id

        image_path = image_path_prefix + str(image_id_prfxd) + ".jpg"
        img = cv2.imread(image_path)
        # rimg = cv2.resize(img, (config.img_dim, config.img_dim))
        images.append(img)

    for ele in tensored_answer_ids.tolist():
        quess.append(dict[ele])

    processed_data = processor(
        images,
        quess,
        # max_length=config.max_ques_len_char,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return processed_data


class Trainer:
    def __init__(self, epochs, batch_size, train_dataset, val_dataset, weights):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.weights = weights

    def train(self, learning_rate):
        loss_function = nn.CrossEntropyLoss(weight=self.weights).to(device)
        model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        vilt_model = ClassifierViltModel(model).to(device)

        optimizer = optim.Adam(params=vilt_model.parameters(), lr=learning_rate)

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
                vilt_model.train()
                optimizer.zero_grad()
                (
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    tokenized_label,
                    tokenized_image_id,
                    tokenized_answer_id,
                ) = batch

                vilt_inputs = get_process_data(
                    tokenized_image_id,
                    config.train_image_path,
                    tokenized_answer_id,
                    "train",
                ).to(device)

                output = vilt_model(vilt_inputs)
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
                    tokenized_image_id,
                    tokenized_answer_id,
                ) = batch
                vilt_model.eval()

                with torch.no_grad():
                    vilt_inputs = get_process_data(
                        tokenized_image_id,
                        config.val_image_path,
                        tokenized_answer_id,
                        "val",
                    ).to(device)
                    output = vilt_model(vilt_inputs)

                    loss = loss_function(output, tokenized_label.to(device))

                    val_loss.append(loss.cpu().item())
                    preds = torch.argmax(output, dim=1)
                    val_pred.extend(preds.cpu().tolist())
                    val_gold.extend(tokenized_label.tolist())
                    torch.cuda.empty_cache()

            e_val_loss.append(np.mean(val_loss))
            print(f"Average validation batch loss: {np.mean(val_loss)}")
            val_acc = accuracy_score(val_gold, val_pred)
            e_val_acc.append(val_acc)
            print(f"Validation accuracy score: {val_acc}")

            output_filename = (
                f"{config.project_storage_path}/models/vilt/{learning_rate}_{epoch}"
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
                    "vilt_model_param": vilt_model.state_dict(),
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
    model_instance.load_state_dict(checkpoint["vilt_model_param"])
    print(
        f"""Val_Acc of loaded model: {checkpoint["acc_metric"]} at epoch {checkpoint["epoch"]} with learning rate {checkpoint["learning_rate"]}"""
    )
    return model_instance


def evalute_test():
    test_dataset = load_pickle_data(config.test_dataset_path_small)
    test_dataLoader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=True)

    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    vilt_model = ClassifierViltModel(model).to(device)

    print("output_file --", best_perf_dict["output_filename"])

    best_vilt_model = loadModel(vilt_model, best_perf_dict["output_filename"])

    test_pred = []
    test_gold = []
    test_ids = []
    best_vilt_model = best_vilt_model.to(device)

    for batch in test_dataLoader:
        (
            input_ids,
            token_type_ids,
            attention_mask,
            tokenized_label,
            tokenized_image_id,
            tokenized_answer_id,
        ) = batch
        vilt_model.eval()

        with torch.no_grad():
            vilt_inputs = get_process_data(
                tokenized_image_id,
                config.val_image_path,
                tokenized_answer_id,
                "test",
            ).to(device)
            output = vilt_model(vilt_inputs)

            preds = torch.argmax(output, dim=1)
            test_pred.extend(preds.cpu().tolist())
            test_gold.extend(tokenized_label.tolist())
            test_ids.extend(tokenized_answer_id.tolist())
            torch.cuda.empty_cache()

    test_acc = accuracy_score(test_gold, test_pred)
    print(f"Test accuracy score: {test_acc}")

    prediction = (test_pred, test_gold, test_ids)
    save_pickle_data(prediction, config.vilt_prediction_data)


if __name__ == "__main__":
    torch.manual_seed(42)
    learning_rates = [0.0001]

    train_dataset = load_pickle_data(config.train_dataset_path_small)
    # train_image_dict = get_image_dict("train")

    val_dataset = load_pickle_data(config.val_dataset_path_small)
    # val_image_dict = get_image_dict("val")

    weights = get_class_weights()

    for learning_rate in learning_rates:
        print(f"learning_rate {learning_rate}")
        trainer = Trainer(
            epochs=5,
            batch_size=128,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            weights=weights,
        )
        trainer.train(learning_rate)

    print(
        f"""\nBest Val performance of {best_perf_dict["acc_metric"]} at epoch {best_perf_dict["epoch"]} for learning_rate {best_perf_dict["learning_rate"]}"""
    )

    evalute_test()
