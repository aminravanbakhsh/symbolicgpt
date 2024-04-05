# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
),

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

print(parent_dir)
print(parent_parent_dir)

from dimension_analyst import Dimensional_Analyst
from formula import Formula
from unit import Unit

# load libraries
import glob
import json
import math
import pickle
import random
import numpy as np
#from tqdm import tqdm
from numpy import * # to override the math functions

import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig
from scipy.optimize import minimize, least_squares

from helpers import set_seed, sample_from_model
from helpers import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc

import pdb

# set the random seed
set_seed(42)

# Get the parent directory (symbolicgpt)
project_dir = os.path.dirname(os.path.dirname(current_dir))

# Add the project directory to sys.path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from helpers import set_seed, sample_from_model


def test_001():

        TEST_INDEX = 1  
        print("\n-----------------------------------------------")
        print("Start test_{}:".format(TEST_INDEX))
        print("-----------------------------------------------")

        device='CPU'
        scratch=True # if you want to ignore the cache and start for scratch
        numEpochs = 40 # number of epochs to train the GPT+PT model
        embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
        numPoints = [20,250] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
        numVars = 9 # the dimenstion of input points x, if you don't know then use the maximum
        numYs = 1 # the dimension of output points y = f(x), if you don't know then use the maximum
        blockSize = 32 # spatial extent of the model for its context
        testBlockSize = 800
        batchSize = 128 # batch size of training data
        target = 'Skeleton' #'Skeleton' #'EQ'
        const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
        decimals = 8 # decimals of the points only if target is Skeleton
        trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton


        # dataDir = 'D:/Datasets/Symbolic Dataset/Datasets/FirstDataGenerator/'  #'./datasets/'
        # dataFolder = '1-9Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_20-250'
        # dataTestFolder = '1-9Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_20-250/Test_Benchmarks'

        dataDir = "/home/amin/vscodes/symbolicgpt/untracked_folder/datasets"
        dataFolder = "sample_dataset"
        dataTestFolder = "sample_dataset/Test"

        dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize'.format(numVars, numPoints[0], numPoints[1], embeddingSize)
        titleTemplate = "{} equations of {} variables - Benchmark"
        addr = './SavedModels/' # where to save model
        method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
        # EMB_CAT: Concat point embedding to GPT token+pos embedding
        # EMB_SUM: Add point embedding to GPT tokens+pos embedding
        # OUT_CAT: Concat the output of the self-attention and point embedding
        # OUT_SUM: Add the output of the self-attention and point embedding
        # EMB_CON: Conditional Embedding, add the point embedding as the first token
        variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
        # NOT_VAR: Do nothing, will not pass any information from the number of variables in the equation to the GPT
        # LEA_EMB: Learnable embedding for the variables, added to the pointNET embedding
        # STR_VAR: Add the number of variables to the first token
        addVars = True if variableEmbedding == 'STR_VAR' else False
        maxNumFiles = 100 # maximum number of file to load in memory for training the neural network
        bestLoss = None # if there is any model to load as pre-trained one
        fName = '{}_SymbolicGPT_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                                'GPT_PT_{}_{}'.format(method, target), 
                                                'Padding',
                                                variableEmbedding)
        

        LOADEING_DIR = "/home/amin/vscodes/symbolicgpt/untracked_folder/models"

        ckptPath = '{}/{}.pt'.format(LOADEING_DIR,fName.split('.txt')[0])
        print("ckptPath:", ckptPath)

        try: 
                os.mkdir(addr)
        except:
                print('Folder already exists!')

        # load the train dataset
        train_file = 'train_dataset_{}.pb'.format(fName)
        if os.path.isfile(train_file) and not scratch:
                # just load the train set
                with open(train_file, 'rb') as f:
                        train_dataset,trainText,chars = pickle.load(f)
        else:
                # process training files from scratch
                path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)

                # pdb.set_trace()
                files = glob.glob(path)[:maxNumFiles]
                text = processDataFiles(files)
                chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
                text = text.split('\n') # convert the raw text to a set of examples
                trainText = text[:-1] if len(text[-1]) == 0 else text
                random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
                train_dataset = CharDataset(text, blockSize, chars, numVars=numVars, 
                                numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                                const_range=const_range, xRange=trainRange, decimals=decimals, augment=False) 
                # with open(train_file, 'wb') as f:
                #     pickle.dump([train_dataset,trainText,chars], f)

        # # load the val dataset
        # path = '{}/{}/Val/*.json'.format(dataDir,dataFolder)
        # files = glob.glob(path)
        # textVal = processDataFiles([files[0]])
        # textVal = textVal.split('\n') # convert the raw text to a set of examples
        # val_dataset = CharDataset(textVal, blockSize, chars, numVars=numVars, 
        #                 numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
        #                 const_range=const_range, xRange=trainRange, decimals=decimals)

        # load the test data
        path = f'{dataDir}/{dataTestFolder}/*.json'
        print(f'test path is {path}')
        files = glob.glob(path)
        textTest = processDataFiles(files)
        textTest = textTest.split('\n') # convert the raw text to a set of examples
        # test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
        test_dataset = CharDataset(textTest, testBlockSize, chars, numVars=numVars, 
                        numYs=numYs, numPoints=numPoints, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals)
        
        CHARS = ['\n', ' ', '"', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ':', '<', '>', 'C', 'E', 'Q', 'S', 'T', 'X', 'Y', '[', ']', '_', 'a', 'b', 'c', 'e', 'g', 'i', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's', 't', 'x', '{', '}']
        
        vocab_size = len(CHARS)

        # create the model
        pconf = PointNetConfig(embeddingSize=embeddingSize, 
                        numberofPoints=numPoints[1]-1, 
                        numberofVars=numVars, 
                        numberofYs=numYs,
                        method=method,
                        variableEmbedding=variableEmbedding)
        
        mconf = GPTConfig(
                        vocab_size, 
                        blockSize, 
                        n_layer=8,
                        n_head=8,
                        n_embd=embeddingSize, 
                        padding_idx=train_dataset.paddingID)

        model = GPT(mconf, pconf)

        # load the best model
        print('The following model {} has been loaded!'.format(ckptPath))
        model.load_state_dict(torch.load(ckptPath))
        # model = model.eval()

        ## Test the model
        # alright, let's sample some character-level symbolic GPT 
        loader = torch.utils.data.DataLoader(
                                        test_dataset, 
                                        shuffle=False, 
                                        pin_memory=True,
                                        batch_size=1,
                                        num_workers=0)

        pdb.set_trace()

        print("\n-----------------------------------------------")
        print("End test_{}:".format(TEST_INDEX))
        print("-----------------------------------------------")
