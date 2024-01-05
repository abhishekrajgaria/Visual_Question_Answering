import torch
import pickle
from config import Config
from extract_data_and_image import get_top_answer
from transformers import BertTokenizer


config = Config()


def save_pickle_data(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_pickle_data(filename):
    with open(filename, "rb") as file:
        loaded_data = pickle.load(file)
        print("loaded_data type", type(loaded_data))
        return loaded_data


def get_class_weights():
    class_dict = load_pickle_data(config.top_answer_class_id_filename)
    dict = {}
    for label in class_dict:
        # print(type(label))
        dict[class_dict[label]] = 0
    print(len(dict))
    data = load_pickle_data(config.train_dataframe_path)["label"].tolist()
    total_cnt = 0
    # print(dict.keys())
    for label in data:
        dict[label] += 1
        total_cnt += 1

    weights = [0 for i in range(len(class_dict))]
    for label in dict.keys():
        weights[label] = 1 - (dict[label] / total_cnt)

    # print(weights)
    return torch.tensor(weights)


def get_tokenized_dataset(data, tokenizer):
    # print(type(data))
    tokenized_ques = tokenizer(
        data,
        max_length=config.max_ques_len_char,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_tensor = tokenized_ques["input_ids"]
    print("input_tensor", input_tensor.shape)
    return tokenized_ques


def get_tokenized_ques(df):
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
    tokenized_ques = get_tokenized_dataset(df["question"].tolist(), tokenizer)
    return tokenized_ques


def get_image_dict(split):
    image_dict = {}
    if split == "train":
        image_dict = load_pickle_data(config.train_image_dict_path)
    elif split == "val":
        image_dict = load_pickle_data(config.val_image_dict_path)
    else:
        image_dict = load_pickle_data(config.test_dataframe_path)

    return image_dict


def convert_column_to_tensor(column):
    column_list = column.tolist()
    return torch.tensor(column_list)


def get_dataset(split):
    dataframe = []
    if split == "train":
        dataframe = load_pickle_data(config.train_dataframe_path)
        dataframe = dataframe.groupby("answer_type").apply(
            lambda x: x.sample(frac=0.15)
        )
        ques_dict = dict(zip(dataframe["answer_id"], dataframe["question"]))
        save_pickle_data(ques_dict, config.train_ques_dict_path)

    elif split == "val":
        dataframe = load_pickle_data(config.val_dataframe_path)
        dataframe = dataframe.groupby("answer_type").apply(lambda x: x.sample(frac=0.1))
        ques_dict = dict(zip(dataframe["answer_id"], dataframe["question"]))
        save_pickle_data(ques_dict, config.val_ques_dict_path)

    else:
        dataframe = load_pickle_data(config.test_dataframe_path)
        dataframe = dataframe.groupby("answer_type").apply(lambda x: x.sample(frac=0.2))
        ques_dict = dict(zip(dataframe["answer_id"], dataframe["question"]))
        save_pickle_data(ques_dict, config.test_ques_dict_path)

    print("split", split)
    print(dataframe.shape)
    print(dataframe.head())

    tokenized_question = get_tokenized_ques(dataframe)
    # print("tokenized_question", tokenized_question.shape)
    tokenized_label = convert_column_to_tensor(dataframe["label"])
    print("tokenized_label", tokenized_label.shape)
    tokenized_image_id = convert_column_to_tensor(dataframe["image_id"])
    print("tokenized_image_id", tokenized_image_id.shape)
    tokenized_answer_id = convert_column_to_tensor(dataframe["answer_id"])
    print("tokenized_answer_id", tokenized_answer_id.shape)

    dataset = torch.utils.data.TensorDataset(
        tokenized_question["input_ids"],
        tokenized_question["token_type_ids"],
        tokenized_question["attention_mask"],
        tokenized_label,
        tokenized_image_id,
        tokenized_answer_id,
    )

    return dataset


if __name__ == "__main__":
    train_dataset = get_dataset("train")
    save_pickle_data(train_dataset, config.train_dataset_path_small)
    val_dataset = get_dataset("val")
    save_pickle_data(val_dataset, config.val_dataset_path_small)
    test_dataset = get_dataset("test")
    save_pickle_data(test_dataset, config.test_dataset_path_small)
    # weights = get_class_weights()
    # save_pickle_data(weights, config.class_weights_path)
