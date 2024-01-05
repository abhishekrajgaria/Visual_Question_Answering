# Visual_Question_Answering
Visual Question Answering on VQA v2 dataset


## Introduction
Asking a neural network a textual question that requires Visual stimuli to get to an answer is interesting.
As humans, over the years we easily get accustomed to identify visual objects or infer from visual scenes
based on text but training a neural network to do the same is a challenging task. This particular problem
has many applications but the most notable one is helping visually impaired people to describe/infer from
surroundings based on their query. The problem involves multi-modality learning which requires effective
representation of textual data to successfully infer from image data.

Relevance to class - The problem requires effective representation of the textual Question data and to
also accommodate the image information, this includes both Natural Language Processing and Deep
Learning understanding.

Personal Motivation - The problem is interesting to me as it deals with multi-modal data and which is like
adding a new layer of sense to the model with the help of data. In class we have only dealt with textual
data, I also wanted to explore the image data and experience the challenges that come with multiple
modals of data.

## Literature Survey

Visual Question Answering is among the tasks which gained popularity around the time when Deep
Learning techniques were started giving state-of-art for various Vision and NLP based tasks. In the past
years, there has been tremendous work performed on the intersection of the Visual and NLP field not only
in terms of designing new tasks and coming with new architecture but also carefully designing and
collecting dataset for such multimodal tasks. In the past it has been observed that models tend to learn
biases or make decisions from the textual information and do not adequately infer information from the
image counterpart of the sample.

In this section, we will discuss briefly about the techniques used in the models. We will talk about the
dataset in the Dataset section.

Convolutional Neural Networks (CNN) revolutionized the field of Deep Learning on Image data, showing a
great ability to effectively learn from the raw image in hierarchical manner while learning various features
in between. In recent years, research has been working with very deep neural networks (and it seems to
work well), among one such deep neural network is ResNet, it was one of ground-breaking architecture in
the domain. Resnet addressed the challenge of training very deep neural networks by introducing a novel
residual learning framework. Idea is very much similar to the Highway networks (we saw in the class), it
connects higher level blocks to lower level blocks by skipping a few layers in between (addresses the
issue of vanishing gradient in the deep networks).

Transformers, introduced by Vaswani et. al. in 2017, has shifted the paradigm in the field of Deep
Learning, Natural Language Processing and other Artificial Intelligence fields as a whole. It moved the
field from sequential processing by introducing a self-attention mechanism to generate contextualized
embedding in parallel. The Bidirectional Encoder Representation from Transformers (BERT), set
benchmark for various NLP tasks, as the name suggests it is an encoder model. It introduced the
bidirectional context understanding in the Transformer architecture.

The versatility of the Transformer model extends beyond the natural language processing to computer
vision and more. One such architecture is the Vision-and-Language Transformer. It addresses the
problem of high cost for extracting visual embeddings encountered with deep Convolutional networks as
compared to that of Multi-Modality interaction step. It generates convolutional free embedding of image
input by dividing the image into blocks then flattening it to finally pass it to the transformer, it also
preserves the location of image blocks by passing the positional embeddings.

There are many interesting architectures that have been innovated in this field, but I would like to talk
about one more “Show, Ask, Attend and Answer” - the idea is simple and mostly we all humans apply
the same approach for answering any Visual Question Answering Problem. The architecture first
generates a textual embedding representing the question and based on this embedding apply Attention
on the image to gather the relevant information.

## Dataset
VQA dataset produced by Dhruv Batra and Devi Parikh et. al. has been a standard dataset for Visual
Question Answering tasks. The first version of dataset VQA v1 was an unbalanced dataset which had a
lot of bias on the Textual part of the input (Question-bias), improving over it, they released VQA v2 which
is balanced and introduced examples which require solid information gathering from the image. As a part
of the project I have also inference the performance of models based on Question-type, Answer-type and
learning via only textual mode. Original Dataset Contained 82,273 training images, 443,757 training
questions, and 4,437,570 training answers (claim is 10 answers per question, but many of them just
repeat). 40,504 validation images, 214,354 validation questions, 2,143,540 validation answers. Testing
data is only available for the Image and Question not for answers.
In the original paper, they have addressed the VQA problem as a Classification task, as many of the
answers are single words (86% of total answers). In the project, I have filtered the dataset to
accommodate top 1000 answers, as the rest only have single digit samples (keeping the single word
answer ratio same). Moreover, In order to reduce the size of the dataset, I have sampled training samples
from training dataset, validation and test samples from the validation dataset based on the answer type
ratio to original dataset size.

## Methodology
In this section, I will go over the models implemented. Three models have been implemented: Text model,
Image + Text model and ViLT model. Hugging Face Library has been used, all models are fine-tuned on
the dataset and accompanied by a classification layer (fully connected network) to generate 1000 logits (#
of classes).

Text Model - The objective of the model was to learn how much is Question bias is present in the new
VQA version 2 dataset. It is also one of the baseline models. Pre-Trained BERT model used
“prajjwal1/bert-tiny”, question data is tokenized using the same model tokenizer, CLS vector dimension
= 128. Model is trained for 20 epochs with lr = [0.001, 0.0001] and the best validation model is saved
(accuracy as the metric).

Image + Text Model - The objective of the was to combine both the textual and image embedding
together and to observe the contribution of the image for the task. For generating textual embedding the
above BERT model is used. For generating image embedding I have used “microsoft/resnet-18” trained
for image classification task, image is processed using the image processor for the same mode. Final
layer output contains 512 features for 1x1 block which are transformed to 128 vectors using a FC layer.
Finally concatenated the two embedding forming a 256 size vector and passed to the classification layer.
Model is trained for 10 epochs for lr=[0.001, 0.0001].

ViLT Model - The objective was to see the effect of a specific model designed for a visual and language
task which does not use convolutional networks. I have used “dandelin/vilt-b32-mlm” trained on
masked-language-modeling given an image and text. Final layer output is a 768 size vector and is passed
to the classification layer. Model is trained for 5 epochs with lr = [0.0001].
Loss - weighted crossEntropy loss function is used based on the labels as the data is highly imbalanced,
Adam - optimizer is used.

## Results and Inference
● ViLT model performed the best, which shows the superiority of the transformer over the CNN (it is
little exaggerating to say this as ViLT is much more complex then our Image+Text model).
● Image + Text model performed the worst, the model tends to overfit the training dataset as
evident from the Train & Val plot for the both Acc. and Loss. It could be due to the ResNet
requiring larger training size.
● In VQA v2, there is still a lot of textual bias, one evident would be that “Yes/No” type, this
answer_type class has higher accuracy then rest, because it is observed Questions started with
word “is”, “are”, “has”, “was” more tend to have binary answer which the model is exploiting.
● ViLT and Text model would be good for comparison (As both are Transformers based models),
ViLT performs better, which indicates the contribution of the image counterpart in the dataset.

## Conclusion
● Ideas Explored - In this project, I explored working with multi-modal dataset, How to infer your
model which are indicative of model performance. Techniques to reduce the GPU usage.
● Learnings - Convolutional Networks, Transformers on Images, Read the Show, Ask, Attend and
Tell paper (application of attention concept)
● Future work - I would like to implement an attention mechanism on the ResNet output based on
the Bert output. Also would like to make the model work on larger data.

## References
1. https://arxiv.org/pdf/2102.03334.pdf
2. https://arxiv.org/pdf/1704.03162.pdf
3. https://visualqa.org/index.html
