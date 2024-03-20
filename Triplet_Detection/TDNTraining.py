# This is tuned hyper-parameters
from __future__ import absolute_import, division, print_function
import sys
import logging.config
from configparser import ConfigParser
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from parser import DFG_solidity

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange

import os
import random
import numpy as np
#import cPickle as pickle
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
from keras.callbacks import CSVLogger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import torch.nn as nn
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = False  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
from model import TripletNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

description = 'Triplet_Model'
alpha = 0.1
batch_size_value = 128
emb_size = 64
number_epoch = 30
cpu_cont = 16
logging.config.fileConfig("logging.cfg")
logger = logging.getLogger("root")
# Training the Triplet Model
from model import Model
from MyDetect import TextDataset, RuntimeContext

import torch.optim as optim

import torch
import torch.nn.functional as F

from model import TripletNetwork

alpha_value = float(alpha)
print(description)
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

args = RuntimeContext()

    # Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    # Set seed
set_seed(args)
config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
config.num_labels = 1
tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
# Evaluate the prediction
num_classes = 2
encoder_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
model = Model(encoder_model)
model.load_state_dict(torch.load("trained_model.pth"))

model.fc = torch.nn.Linear(in_features=64, out_features=num_classes)
# model.eval()

# Set up the test data loader
N_training_dataset = TextDataset(tokenizer, args, file_path='dataset/valid.txt')
N_training_sampler = SequentialSampler(N_training_dataset)
N_training_dataloader = DataLoader(N_training_dataset, sampler=N_training_sampler, batch_size=args.eval_batch_size, num_workers=2)

# Initialize lists to store predictions and labels
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
  model.train()
  for batch in N_training_dataloader:
    inputs_ids_1, position_idx_1, attn_mask_1, label1, _, _, _, _, _, _, _, _ = batch # For testing, positive and negative IDs are not used
    optimizer.zero_grad()
    outputs = model(inputs_ids_1, position_idx_1, attn_mask_1)
    print("the size of output:", outputs.shape)
    print("the outputs are: ", outputs)
    loss = criterion(outputs, label1)
    loss.backward()
    optimizer.step()
  print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")

torch.save(model.state_dict(), 'trained_classifier.pth')





