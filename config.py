from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


class Config:
    # Holds model hyperparams and data information
    num_answer_class = 1000
    max_sent_len_char = 80
    max_ques_len_char = 100

    img_dim = 240

    vlt_emb_dim = 768

    remove_spl_char = False

    train_fraction = 0.8
    val_fraction = 0.2

    test_fraction = 1

    resnet_model_path = "microsoft/resnet-18"
    bert_model_path = "prajjwal1/bert-tiny"

    bert_cls_emb_dim = 128
    res_pooler_dim = 512

    project_storage_path = "/scratch/general/vast/u1471428/cs6957/project"

    train_ques_dict_path = project_storage_path + "/data/pkld/train_ques_dict.pickle"
    val_ques_dict_path = project_storage_path + "/data/pkld/val_ques_dict.pickle"
    test_ques_dict_path = project_storage_path + "/data/pkld/test_ques_dict.pickle"

    vilt_prediction_data = project_storage_path + "/data/pkld/vilt_prediction.pickle"

    text_prediction_data = project_storage_path + "/data/pkld/text_prediction.pickle"
    image_text_prediction_data = (
        project_storage_path + "/data/pkld/image_text_prediction.pickle"
    )

    class_weights_path = project_storage_path + "/data/pkld/class_weight.pickle"

    train_dataset_path = project_storage_path + "/data/pkld/train_dataset.pickle"
    val_dataset_path = project_storage_path + "/data/pkld/val_dataset.pickle"
    test_dataset_path = project_storage_path + "/data/pkld/test_dataset.pickle"

    train_dataset_path_small = (
        project_storage_path + "/data/pkld/train_dataset_small.pickle"
    )
    val_dataset_path_small = (
        project_storage_path + "/data/pkld/val_dataset_small.pickle"
    )
    test_dataset_path_small = (
        project_storage_path + "/data/pkld/test_dataset_small.pickle"
    )

    train_dataframe_path = project_storage_path + "/data/pkld/train_dataframe.pickle"
    val_dataframe_path = project_storage_path + "/data/pkld/val_dataframe.pickle"
    test_dataframe_path = project_storage_path + "/data/pkld/test_dataframe.pickle"

    train_image_dict_path = (
        project_storage_path + "/data/pkld/train_image_dict_path.pickle"
    )

    val_image_dict_path = project_storage_path + "/data/pkld/val_image_dict_path.pickle"
    test_image_dict_path = (
        project_storage_path + "/data/pkld/test_image_dict_path.pickle"
    )

    top_answer_class_id_filename = (
        project_storage_path + "/data/pkld/top_answer_class_id.pickle"
    )

    vocab_filename = project_storage_path + "/data/pkld/vocab.pickle"
    vocab_char_filename = project_storage_path + "/data/pkld/vocab_char.pickle"

    answer_cnt_dict_filename = (
        project_storage_path + "/data/pkld/answer_cnt_dict.pickle"
    )

    train_question_path = (
        project_storage_path + "/data/brd/v2_OpenEnded_mscoco_train2014_questions.json"
    )
    train_annotation_path = (
        project_storage_path + "/data/brd/v2_mscoco_train2014_annotations.json"
    )
    train_image_path = project_storage_path + "/data/brd/train2014/COCO_train2014_"

    val_question_path = (
        project_storage_path + "/data/brd/v2_OpenEnded_mscoco_val2014_questions.json"
    )
    val_annotation_path = (
        project_storage_path + "/data/brd/v2_mscoco_val2014_annotations.json"
    )
    val_image_path = project_storage_path + "/data/brd/val2014/COCO_val2014_"
