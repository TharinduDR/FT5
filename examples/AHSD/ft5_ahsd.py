import argparse
import math
import os
import statistics

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ft5.args import T5Args
from ft5.evaluation import sentence_label_evaluation
from ft5.t5_model import T5Model


parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="t5-base")
parser.add_argument('--model_type', required=False, help='model type', default="t5")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

model_type = arguments.model_type
model_name = arguments.model_name
cuda_device = int(arguments.cuda_device)

FOLDS = 5
SEED = 777
macro_f1_scores = []
weighted_f1_scores = []

thresholds = [0.05, 0.1, 0.15]
offensive_thresholds = [0.8, 0.7, 0.6, 0.5]


for i in range(FOLDS):

    ahsd = pd.read_csv("examples/AHSD/labeled_data.csv")
    ahsd_train, ahsd_test = train_test_split(ahsd, test_size=0.2, random_state=777)


    ahsd_train["prefix"] = "ahsd"
    ahsd_train['target_text'] = ahsd_train['class'].map({0: "HAT", 1: "OFF", 2: "NOT"})

    ahsd_test['target_text'] = ahsd_test['class'].map({0: "HAT", 1: "OFF", 2: "NOT"})

    ahsd_train = ahsd_train.rename(columns={'tweet': 'input_text'})
    ahsd_train = ahsd_train[["prefix", "input_text", "target_text"]]
    train_df, eval_df = train_test_split(ahsd_train, test_size=0.2, random_state=SEED * i)

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
    model_args.manual_seed = SEED * i


    model_name_prefix = "ahsd_" + model_name

    model_args.output_dir = os.path.join(model_name_prefix, "outputs")
    model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
    model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

    model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    result = model.eval_model(eval_df)

    test_list = []

    for index, row in ahsd_test.iterrows():
        test_list.append("ahsd: " + row['tweet'])

    model = T5Model(model_type, model_args.best_model_dir, args=model_args, use_cuda=torch.cuda.is_available(),
                            cuda_device=cuda_device)

    preds = model.predict(test_list)
    macro_f1, weighted_f1 = sentence_label_evaluation(preds, ahsd_test['target_text'].tolist())
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)



print("Weighted F1 scores ", weighted_f1_scores)
print("Mean weighted F1 scores", statistics.mean(weighted_f1_scores))
print("STD weighted F1 scores", statistics.stdev(weighted_f1_scores))

print("Macro F1 scores ", macro_f1_scores)
print("Mean macro F1 scores", statistics.mean(macro_f1_scores))
print("STD macro F1 scores", statistics.stdev(macro_f1_scores))
