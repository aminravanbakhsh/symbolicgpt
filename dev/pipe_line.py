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

        if      num_vars == 1:
            dir_path    = "/home/amin/vscodes/symbolicgpt/untracked_folder/models"
            model_path  = "XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt"


            numEpochs           = 20 # number of epochs to train the GPT+PT model
            embeddingSize       = 512 # the hidden dimension of the representation of both GPT and PT
            numPoints           = [30,31] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
            numVars             = 1 # the dimenstion of input points x, if you don't know then use the maximum
            numYs               = 1 # the dimension of output points y = f(x), if you don't know then use the maximum
            blockSize           = 200 # spatial extent of the model for its context
            testBlockSize       = 400
            batchSize           = 128 # batch size of training data
            target              = 'Skeleton' #'Skeleton' #'EQ'
            const_range         = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
            decimals            = 8 # decimals of the points only if target is Skeleton
            trainRange          = [-3.0,3.0] # support range to generate during training only if target is Skeleton
            dataDir             = './datasets/'
            dataInfo            = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
            titleTemplate       = "{} equations of {} variables - Benchmark"
            target              = 'Skeleton' #'Skeleton' #'EQ'
            dataFolder          = '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points'
            addr                = './SavedModels/' # where to save model
            method              = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
            variableEmbedding   = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR






        elif    num_vars == 2:
            dir_path    = "/home/amin/vscodes/symbolicgpt/untracked_folder/models"
            model_path  = "XYE_2Var_200-201Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt"


        elif    num_vars == 3:
            dir_path    = "/home/amin/vscodes/symbolicgpt/untracked_folder/models"
            model_path  = "XYE_3Var_500-501Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt"


        elif    num_vars == 5:
            dir_path    = "/home/amin/vscodes/symbolicgpt/untracked_folder/models"
            model_path  = "XYE_5Var_10-200Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt"



        elif    num_vars == 9:
            dir_path    = "/home/amin/vscodes/symbolicgpt/untracked_folder/models"
            model_path  = "XYE_9Var_20-250Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt"


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
    
