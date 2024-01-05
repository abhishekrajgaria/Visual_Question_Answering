import pickle
import pandas as pd
from config import Config
from sklearn.metrics import accuracy_score, f1_score

config = Config()


def load_pickle_data(filename):
    with open(filename, "rb") as file:
        loaded_data = pickle.load(file)
        print("loaded_data type", type(loaded_data))
        return loaded_data


def get_test_data():
    test_dataframe = load_pickle_data(config.test_dataframe_path)
    print("test_dataframe column", test_dataframe.columns)
    return test_dataframe


def get_test_result(prediction_path):
    test_result = load_pickle_data(prediction_path)
    return test_result


def form_desired_df(model_type):
    test_df = get_test_data()
    prediction_path = ""
    if model_type == "text":
        prediction_path = config.text_prediction_data
    elif model_type == "image_text":
        prediction_path = config.image_text_prediction_data
    else:
        prediction_path = config.vilt_prediction_data

    result = get_test_result(prediction_path)

    pred, label, answer_id = result

    dict = {"prediction": pred, "gold": label, "answer_id": answer_id}

    # compute accuracy
    print("Acc. ", accuracy_score(dict["gold"], dict["prediction"]))

    # compute f1_score
    print("F-1 macro", f1_score(dict["gold"], dict["prediction"], average="macro"))

    df = pd.DataFrame(dict)

    req_df = pd.merge(df, test_df, on="answer_id", how="left")

    final_df = req_df[
        [
            "prediction",
            "gold",
            "answer_id",
            "answer_type",
            "question_type",
            "correct_answer",
            "question",
        ]
    ]
    return final_df


def get_metrics_answer_type(model_type):
    df = form_desired_df(model_type)
    answer_type = df["answer_type"].unique()
    dict = {}
    for ans_type in answer_type:
        condition = df["answer_type"] == ans_type
        cond_result = df.loc[condition, ["prediction", "gold"]]
        dict[ans_type] = accuracy_score(
            cond_result["gold"].tolist(), cond_result["prediction"].tolist()
        )
    print(dict)
    print(df["answer_type"].value_counts(normalize=True))


def get_metric_question_type(model_type):
    df = form_desired_df(model_type)
    df["question_type"] = df["question_type"].apply(lambda x: x.split()[0])
    question_type = df["question_type"].unique()
    dict = {}
    for ques_type in question_type:
        condition = df["question_type"] == ques_type
        cond_result = df.loc[condition, ["prediction", "gold"]]
        dict[ques_type] = accuracy_score(
            cond_result["gold"].tolist(), cond_result["prediction"].tolist()
        )
    print(dict)
    print(df["question_type"].value_counts())


if __name__ == "__main__":
    # get_metrics_answer_type("text")
    # get_metric_question_type("text")

    # get_metrics_answer_type("image_text")
    # get_metric_question_type("image_text")

    get_metrics_answer_type("vilt")
    get_metric_question_type("vilt")
