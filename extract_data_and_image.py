import re
import cv2
import json
import torch
import pickle
import pandas as pd
from config import Config
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from transformers import BertTokenizer
from transformers import AutoImageProcessor


config = Config()


def save_pickle_data(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_pickle_data(filename):
    with open(filename, "rb") as file:
        loaded_data = pickle.load(file)
        print("loaded_data type", type(loaded_data))
        return loaded_data


def gen_vocab():
    vocab = set()
    possible_answers = get_top_answer()
    for answer in possible_answers:
        answer_list = answer.split()
        for ele in answer_list:
            vocab.add(ele)
    print(vocab)
    save_pickle_data(vocab, config.vocab_filename)


def get_answer_cnt_dict():
    answer_cnt_dict = {}
    train_dataframe = get_train_df()

    for _, row in train_dataframe.iterrows():
        answer = row["correct_answer"]

        if answer not in answer_cnt_dict.keys():
            answer_cnt_dict[answer] = 0
        answer_cnt_dict[answer] += 1

    save_pickle_data(answer_cnt_dict, config.answer_cnt_dict_filename)


def get_top_answer():
    answer_cnt_dict = load_pickle_data(config.answer_cnt_dict_filename)

    top_answer_cnt_dict = dict(
        sorted(answer_cnt_dict.items(), key=lambda x: x[1], reverse=True)[
            : config.num_answer_class
        ]
    )

    # print(top_answer_cnt_dict.keys())
    top_answer_class_id = {}
    cnt = 0
    for key in top_answer_cnt_dict.keys():
        top_answer_class_id[key] = cnt
        cnt += 1
    save_pickle_data(top_answer_class_id, config.top_answer_class_id_filename)
    return top_answer_cnt_dict.keys()


def gen_image_dict(df, split):
    data = df
    image_path_prefix = ""

    if split == "train":
        image_path_prefix = config.train_image_path

    elif split == "val":
        image_path_prefix = config.val_image_path

    # image_data = []
    image_processor = AutoImageProcessor.from_pretrained(config.resnet_model_path)

    image_dict = {}
    image_ids = set()
    for image_id in data["image_id"]:
        image_id = str(image_id)
        image_ids.add(image_id)

    print(len(image_ids))

    for image_id in image_ids:
        image_dict[image_id] = None

    for image_id in data["image_id"]:
        image_id = str(image_id)
        image_id_len = len(image_id)
        image_id_prfxd = "0" * (12 - image_id_len) + image_id
        if image_dict[image_id] == None:
            image_filename = image_path_prefix + str(image_id_prfxd) + ".jpg"
            tensored_image = load_image(image_filename, image_processor)
            image_dict[image_id] = tensored_image

    print(len(image_dict))
    flag = 0
    for key in image_dict.keys():
        if image_dict[key] == None:
            flag = 1
            break

    if flag == 1:
        print("wrongly done")
    else:
        print("correctly done")

    # save_pickle_data(image_dict, save_pickle_filname)
    return image_dict


def load_image(image_path, image_processor):
    img = cv2.imread(image_path)
    rimg = cv2.resize(img, (config.img_dim, config.img_dim))
    tensored_img = image_processor([rimg], return_tensors="pt")["pixel_values"]
    return tensored_img


def get_index_to_remove(df):
    index_to_remove = []
    top_answers = get_top_answer()
    for index, row in df.iterrows():
        answer = row["correct_answer"]
        if answer not in top_answers:
            index_to_remove.append(index)

    return index_to_remove


def get_dataframe(ques_data, anno_data, fraction):
    ques_df = pd.DataFrame(ques_data["questions"])
    anno_df = pd.DataFrame(anno_data["annotations"])

    anno_df["answer_id"] = anno_df.index
    anno_df = anno_df.rename(columns={"multiple_choice_answer": "correct_answer"})
    df = pd.merge(anno_df, ques_df, on="question_id", how="left")
    df = df.rename(columns={"image_id_x": "image_id"})
    print("hello")
    return df[
        [
            "answer_id",
            "correct_answer",
            "answer_type",
            "question_id",
            "question",
            "question_type",
            "image_id",
        ]
    ]


def add_label_column(df):
    labels = []
    top_answer_dict = load_pickle_data(config.top_answer_class_id_filename)
    for _, row in df.iterrows():
        labels.append(top_answer_dict[row["correct_answer"]])
    # print(len(labels))
    df["label"] = labels


def get_train_df():
    train_ques_data = json.load(open(config.train_question_path))
    train_anno_data = json.load(open(config.train_annotation_path))
    train_dataframe = get_dataframe(
        train_ques_data, train_anno_data, config.train_fraction
    )
    index_to_remove = get_index_to_remove(train_dataframe)
    train_dataframe.drop(index_to_remove, axis=0, inplace=True)
    add_label_column(train_dataframe)
    print("train_columns", train_dataframe.columns)
    print("train_dataframe_shape", train_dataframe.shape)
    return train_dataframe


def get_val_df(fraction):
    val_ques_data = json.load(open(config.val_question_path))
    val_anno_data = json.load(open(config.val_annotation_path))
    val_dataframe = get_dataframe(val_ques_data, val_anno_data, fraction)
    index_to_remove = get_index_to_remove(val_dataframe)
    val_dataframe.drop(index_to_remove, axis=0, inplace=True)
    add_label_column(val_dataframe)
    print("val_columns", val_dataframe.columns)
    print("val_dataframe_shape", val_dataframe.shape)
    return val_dataframe


def load_df_img_in_pickle():
    train_dataframe = get_train_df()
    val_dataframe = get_val_df(config.val_fraction)
    test_dataframe = get_val_df(config.test_fraction)
    # train_image_dict = gen_image_dict(train_dataframe, "train")
    # val_image_dict = gen_image_dict(val_dataframe, "val")
    # test_image_dict = gen_image_dict(test_dataframe, "val")

    save_pickle_data(train_dataframe, config.train_dataframe_path)
    save_pickle_data(val_dataframe, config.val_dataframe_path)
    save_pickle_data(test_dataframe, config.test_dataframe_path)

    # save_pickle_data(train_image_dict, config.train_image_dict_path)
    # save_pickle_data(val_image_dict, config.val_image_dict_path)
    # save_pickle_data(test_image_dict, config.test_image_dict_path)


def get_dataset(split):
    pass


if __name__ == "__main__":
    # get_text_data()
    # gen_vocab_char()
    # load_vocab()
    # get_anno_vocab2idx()
    # get_anno_idx2vocab()
    # print(get_anno_vocab_weights())
    # get_tensored_id("train")
    # gen_image_dict("train")
    # get_answer_cnt_dict()
    # get_top_answer()
    # get_train_df()
    load_df_img_in_pickle()
