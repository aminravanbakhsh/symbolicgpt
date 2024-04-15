import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

import json

import torch
import torch.nn as nn
from torch.nn import functional as F

import random
import glob
import pdb

from helpers import set_seed, sample_from_model
from helpers import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc

from scipy.optimize import minimize, least_squares
from models import GPT, GPTConfig, PointNetConfig


class Pipeline:

    ################################################################################################
    #                              #            Pipeline           #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################

    @classmethod
    def run_model(cls, model, data):
        pass


    ################################################################################################
    #                              #            Model              #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################

    @classmethod
    def instantiate_model(cls, num_vars=1) -> GPT:
        args = None
        if num_vars == 1:
            args = cls.ARGS_1

        numVars             = args["numVars"]
        numYs               = args["numYs"]
        numPoints           = args["numPoints"] 
        embeddingSize       = args["embeddingSize"]
        method              = args["method"]
        variableEmbedding   = args["variableEmbedding"]
        block_size          = args["blockSize"]
        vocab_size          = args["vocab_size"]
        paddingID           = args["paddingID"]

        pdb.set_trace()

        # create the model
        pconf = PointNetConfig(
                                embeddingSize        = embeddingSize, 
                                numberofPoints       = numPoints[1]-1, 
                                numberofVars         = numVars, 
                                numberofYs           = numYs,
                                method               = method,
                                variableEmbedding    = variableEmbedding)

        mconf = GPTConfig(
                            vocab_size, 
                            block_size,
                            n_layer         = 8, 
                            n_head          = 8,   
                            n_embd          = embeddingSize, 
                            padding_idx     = paddingID
                        )

        model = GPT(mconf, pconf)

        pdb.set_trace()

        return model

    @classmethod
    def train_model(cls, num_vars=5):
        pass


    @classmethod
    def load_model(cls, num_vars = 5):

        """
            XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
            XYE_2Var_200-201Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
            XYE_3Var_500-501Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
            XYE_5Var_10-200Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
            XYE_9Var_20-250Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
        """

        params = {}
        model = None

        return model, params

    ################################################################################################
    #                              #            Data               #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################

    @classmethod
    def load_data(cls, path):
        data = None
        return data    

    @classmethod
    def load_train_data(cls, args_index = 1):

        args = None
        if args_index == 1:
            args = cls.ARGS_1

        data_dir        = args["data_dir"]
        blockSize       = args["blockSize"]
        numVars         = args["numVars"]
        numYs           = args["numYs"]
        numPoints       = args["numPoints"] 
        target          = args["target"]
        addVars         = True if args["variableEmbedding"] == 'STR_VAR' else False
        const_range     = args["const_range"]
        trainRange      = args["trainRange"]
        decimals        = args["decimals"]

        maxNumFiles = 10

        path = "{}/Train/*.json".format(data_dir)
        files = glob.glob(path)[:maxNumFiles]
        text = processDataFiles(files)
        chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
        text = text.split('\n') # convert the raw text to a set of examples
        trainText = text[:-1] if len(text[-1]) == 0 else text
        
        random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
        
        train_dataset = CharDataset(text, 
                                    blockSize, 
                                    chars, 
                                    numVars     = numVars, 
                                    numYs       = numYs,
                                    numPoints   = numPoints,
                                    target      = target,
                                    addVars     = addVars,
                                    const_range = const_range, 
                                    xRange      = trainRange, 
                                    decimals    = decimals, 
                                    augment     = False
                                    )

        pdb.set_trace()

        return train_dataset
                

    @classmethod
    def sample_from_data(cls, 
                         data, 
                         num = -1, 
                         index_list = None, 
                         shuffe = False):

        sampled_data = None
        return sampled_data

    @classmethod
    def transform_data(cls, data):
        transformed_data = None
        return transformed_data
    
    @classmethod
    def transform_equation(cls, equation):
        transformed_equation = None
        return equation
    

    
    ################################################################################################
    #                              #          Parameters           #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################

    ARGS_1 = {
            "data_dir"            : "/home/amin/vscodes/symbolicgpt/untracked_folder/datasets/1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points",
            "model_dir_path"      : "/home/amin/vscodes/symbolicgpt/untracked_folder/models",
            "model_path"          : "XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
            "numEpochs"           : 20, # number of epochs to train the GPT+PT model
            "embeddingSize"       : 512, # the hidden dimension of the representation of both GPT and PT
            "numPoints"           : [30,31], # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
            "numVars"             : 1, # the dimenstion of input points x, if you don't know then use the maximum
            "numYs"               : 1, # the dimension of output points y = f(x), if you don't know then use the maximum
            "blockSize"           : 200, # spatial extent of the model for its context
            "testBlockSize"       : 400,
            "batchSize"           : 128, # batch size of training data
            "target"              : 'Skeleton', #'Skeleton' #'EQ'
            "const_range"         : [-2.1, 2.1], # constant range to generate during training only if target is Skeleton
            "decimals"            : 8, # decimals of the points only if target is Skeleton
            "trainRange"          : [-3.0,3.0], # support range to generate during training only if target is Skeleton
            "dataDir"             : './datasets/',
            # "dataInfo"            : 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize),
            "titleTemplate"       : "{} equations of {} variables - Benchmark",
            "target"              : 'Skeleton', #'Skeleton' #'EQ'
            "dataFolder"          : '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points',
            "addr"                : './SavedModels/', # where to save model
            "method"              : 'EMB_SUM', # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
            "variableEmbedding"   : 'NOT_VAR', # NOT_VAR/LEA_EMB/STR_VAR
            "vocab_size"          : 49,
            "paddingID"           : 34,
            "size"                : 498795,
            "itos"                : {   
                                        0: '\n',
                                        1: ' ', 
                                        2: '"', 
                                        3: '(', 
                                        4: ')', 
                                        5: '*', 
                                        6: '+',
                                        7: ',', 
                                        8: '-',
                                        9: '.', 
                                        10: '/', 
                                        11: '0',
                                        12: '1',
                                        13: '2',
                                        14: '3', 
                                        15: '4', 
                                        16: '5', 
                                        17: '6', 
                                        18: '7', 
                                        19: '8', 
                                        20: '9', 
                                        21: ':', 
                                        22: ':', 
                                        23: '<', 
                                        24: '>', 
                                        25: 'C', 
                                        26: 'E', 
                                        27: 'Q', 
                                        28: 'S', 
                                        29: 'T', 
                                        30: 'X', 
                                        31: 'Y', 
                                        32: '[', 
                                        33: ']', 
                                        34: '_', 
                                        35: 'c', 
                                        36: 'e', 
                                        37: 'g', 
                                        38: 'i', 
                                        39: 'k', 
                                        40: 'l', 
                                        41: 'n', 
                                        42: 'o', 
                                        43: 'p', 
                                        44: 's', 
                                        45: 't', 
                                        46: 'x', 
                                        47: '{', 
                                        48: '}'}
    }