import pandas as pd
from sklearn.model_selection import train_test_split

from ft5.args import T5Args
from ft5.evaluation import print_information
from ft5.t5_model import T5Model

import torch

olid = pd.read_csv("examples/OLID/olid_train.csv", sep="\t")
olid_test = pd.read_csv("examples/OLID/olid_test.csv", sep="\t")


olid["prefix"] = "binary"
olid = olid.rename(columns={'Text': 'input_text', 'Class': 'target_text'})
olid = olid[["prefix", "input_text", "target_text"]]


train_df, eval_df = train_test_split(olid, test_size=0.2)

model_args = T5Args()
model_args.num_train_epochs = 3
model_args.no_save = False
model_args.fp16 = False
model_args.learning_rate = 1e-5
model_args.max_length = 3
model_args.max_seq_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 400
model_args.evaluate_during_training_verbose = True
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False

model = T5Model("t5", "t5-base", args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=0)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

test_list = []

for index, row in olid_test.iterrows():
   test_list.append("binary: " + row['Text'])

preds = model.predict(test_list)

print_information(preds, olid_test["Class"].tolist())
