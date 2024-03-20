import os
import random
import numpy as np
import cPickle as pickle
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
from keras.callbacks import CSVLogger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import torch.nn as nn
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
from model import TripletNetwork
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

alpha = 0.1
batch_size_value = 128
emb_size = 64
number_epoch = 30
cpu_cont = 16
logging.config.fileConfig("logging.cfg")
logger = logging.getLogger("root")
# Training the Triplet Model
from model import Model
from MyModel import TextDataset, RuntimeContext

import torch.optim as optim

import torch
import torch.nn.functional as F

from model import TripletNetwork
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
#anchor_features, positive_features, negative_features = TripletNetwork(Model)
# Customized loss
# def cosine_triplet_loss(anchor_features, positive_features, negative_features):
#     pos_sim = torch.nn.functional.cosine_similarity(anchor_features, positive_features)
#     neg_sim = torch.nn.functional.cosine_similarity(anchor_features, negative_features)
#     loss = torch.mean(torch.relu(neg_sim - pos_sim + 1))
#     return loss

# model = RobertaForSequenceClassification.from_pretrained(
#         args.model_name_or_path, config=config)
# model = Model(model, config, tokenizer, args)
# anchor_features, positive_features, negative_features = TripletNetwork(model)
#
# pos_sim = Dot(axes=-1, normalize=True)([anchor_features,positive_features])
# neg_sim = Dot(axes=-1, normalize=True)([anchor_features,negative_features])
#
# # customized loss
# loss = Lambda(cosine_triplet_loss,
#               output_shape=(1,))(
#              [pos_sim,neg_sim])
#
# model_triplet = Model(
#     inputs=[anchor_features, positive_features, negative_features],
#     outputs=loss)
# print(model_triplet.summary())
#
# opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#
# model_triplet.compile(loss=identity_loss, optimizer=opt)
# csv_logger = CSVLogger('log/Training_Log_%s.csv'%description, append=True, separator=';')
# train_dataset = TextDataset(tokenizer, args, file_path=args.Anchor_train_data_file)
# train_sampler = RandomSampler(train_dataset)
# train_dataloader = DataLoader(
#     train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
# for epoch in range(args.epochs):
#     bar = tqdm(train_dataloader, total=len(train_dataloader))
#     for step, batch in enumerate(bar):
#          (inputs_ids_1, position_idx_1, attn_mask_1, label1,
#          inputs_ids_2, position_idx_2, attn_mask_2, label2,
#          inputs_ids_3, position_idx_3, attn_mask_3, label3) = [x.to(args.device) for x in batch]
#          anchor_features, positive_features, negative_features = model(inputs_ids_1, inputs_ids_2, inputs_ids_2,
#                                                                           position_idx_1, position_idx_2,
#                                                                           position_idx_3,
#                                                                           attn_mask_1, attn_mask_2, attn_mask_3)
#         model_triplet.fit()
args = RuntimeContext()

    # Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu, )
    # Set seed
set_seed(args)
config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
config.num_labels = 1
tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

def cosine_triplet_loss(y_pred, margin=0.2):
    pos_sim, neg_sim = y_pred
    loss = torch.relu(margin + neg_sim - pos_sim).mean()
    return loss

encoder_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
model = Model(encoder_model)
triplet_network = TripletNetwork(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_dataset= TextDataset(tokenizer, args, file_path=args.Anchor_train_data_file)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

epochs = args.epochs

for epoch in range(epochs):
    for batch in train_dataloader:
        inputs_ids_1, position_idx_1, attn_mask_1, label1, inputs_ids_2, position_idx_2, attn_mask_2, label2, inputs_ids_3, position_idx_3, attn_mask_3, label3 = batch
        optimizer.zero_grad()
        anchor_features, positive_features, negative_features = triplet_network(inputs_ids_1, position_idx_1, attn_mask_1,
                                                                                 label1, inputs_ids_2, position_idx_2, attn_mask_2,
                                                                                 label2, inputs_ids_3, position_idx_3, attn_mask_3, label3)
        pos_sim = nn.functional.cosine_similarity(anchor_features, positive_features)
        neg_sim = nn.functional.cosine_similarity(anchor_features, negative_features)
        loss = cosine_triplet_loss((pos_sim, neg_sim))

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")


