import pandas as pd
from sklearn.model_selection import train_test_split
import os
from ft5.args import T5Args
from ft5.evaluation import sentence_label_evaluation

from ft5.t5_model import T5Model
import math
import statistics
import torch

FOLDS = 10
SEED = 777
macro_f1_scores = []
weighted_f1_scores = []

for i in range(FOLDS):

    hateval = pd.read_csv("examples/HATEVAL/hateval2019_en_train.csv")
    hateval_test = pd.read_csv("examples/HATEVAL/hateval2019_en_test.csv")

    hateval = hateval[["text", "HS"]]
    hateval_test = hateval_test[["text", "HS"]]

    hateval['HS'] = hateval['HS'].map({1: "HAT", 0: "NOH"})
    hateval_test['HS'] = hateval_test['HS'].map({1: "HAT", 0: "NOH"})


    hateval["prefix"] = "hateval"
    hateval = hateval.rename(columns={'text': 'input_text', 'HS': 'target_text'})
    hateval = hateval[["prefix", "input_text", "target_text"]]

    train_df, eval_df = train_test_split(hateval, test_size=0.2, random_state=SEED * i)

    model_args = T5Args()
    model_args.num_train_epochs = 5
    model_args.no_save = False
    model_args.fp16 = False
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 16
    model_args.max_length = 3
    model_args.max_seq_length = 256
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = int(
       math.floor(len(train_df) / (model_args.train_batch_size * 3) / 100.0)) * 100
    model_args.evaluate_during_training_verbose = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.use_multiprocessed_decoding = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.manual_seed = SEED*i

    model_type = "t5"
    model_name = "t5-base"
    threshold = 0.05
    # model_name = os.path.join("ft5_" + str(threshold), "outputs", "best_model")
    model_name_prefix = "hateval_" + model_name

    model_args.output_dir = os.path.join(model_name_prefix, "outputs")
    model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
    model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

    model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=0)

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    result = model.eval_model(eval_df)

    test_list = []

    for index, row in hateval_test.iterrows():
       test_list.append("hateval: " + row['text'])

    model = T5Model(model_type, model_args.best_model_dir, args=model_args, use_cuda=torch.cuda.is_available(),
                    cuda_device=0)

    preds = model.predict(test_list)
    macro_f1, weighted_f1 = sentence_label_evaluation(preds, hateval_test["HS"].tolist())
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)


print("Weighted F1 scores ", weighted_f1_scores)
print("Mean weighted F1 scores", statistics.mean(weighted_f1_scores))
print("STD weighted F1 scores", statistics.stdev(weighted_f1_scores))

print("Macro F1 scores ", macro_f1_scores)
print("Mean macro F1 scores", statistics.mean(macro_f1_scores))
print("STD macro F1 scores", statistics.stdev(macro_f1_scores))




