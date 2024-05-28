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

from numpy import * # to override the math functions
from helpers import set_seed, sample_from_model
from helpers import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc

from scipy.optimize import minimize, least_squares
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig


class Pipeline:

    ################################################################################################
    #                              #            Pipeline           #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################

    @classmethod
    def eval_model(cls, dataset, args_index = 1):

        ## Test the model
        # alright, let's sample some character-level symbolic GPT

        args = None
        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9


        blockSize           = args["blockSize"]
        # target              = args["target"]
        itos                = args["itos"]
        variableEmbedding   = args["variableEmbedding"]
        paddingToken        = args["paddingToken"]

        model = cls.load_model(args_index)

        loader = torch.utils.data.DataLoader(
                                                dataset, 
                                                shuffle         = False, 
                                                pin_memory      = True,
                                                batch_size      = 1,
                                                num_workers     = 0
                                            )

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        try:
            for i, batch in enumerate(loader):

                inputs, outputs, points, variables = batch

                # print("-----------------------------------------------")
                # print(inputs)
                # print(points)
                # print("-----------------------------------------------")


                inputs      = inputs[:,0:1].to(device)
                points      = points.to(device)
                variables   = variables.to(device)

                t = json.loads(dataset.data[i])
                outputsHat_raw  = sample_from_model(
                                model           = model, 
                                x               = inputs, 
                                steps           = blockSize, 
                                points          = points,
                                variables       = variables,
                                temperature     = 1.0, 
                                sample          = True, 
                                top_k           = 0.0,
                                top_p           = 0.7,
                                params          = args
                                )

                outputsHat = outputsHat_raw[0]

                # filter out predicted
                target      = ''.join([itos[int(i)] for i in outputs[0]])
                predicted   = ''.join([itos[int(i)] for i in outputsHat])

                if variableEmbedding == 'STR_VAR':
                    target      = target.split(':')[-1]
                    predicted   = predicted.split(':')[-1]

                target      = target.strip(paddingToken).split('>')
                target      = target[0] #if len(target[0])>=1 else target[1]
                target      = target.strip('<').strip(">")
                predicted   = predicted.strip(paddingToken).split('>')
                predicted   = predicted[0] #if len(predicted[0])>=1 else predicted[1]
                predicted   = predicted.strip('<').strip(">")

                print("-----------------------------------------------")
                print('Target:    {}\nSkeleton:  {}'.format(target, predicted))
                print("-----------------------------------------------")

                # train a regressor to find the constants (too slow)
                c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
                
                # c[-1] = 0 # initialize the constant as zero
                b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables

                try:
                    if len(c) != 0:
                        # This is the bottleneck in our algorithm
                        # for easier comparison, we are using minimize package  
                        cHat = minimize(lossFunc, c, #bounds=b,
                                    args=(predicted, t['X'], t['Y'])) 

                        predicted = predicted.replace('C','{}').format(*cHat.x)

                except ValueError:
                    raise 'Err: Wrong Equation {}'.format(predicted)
                except Exception as e:
                    raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)
                
                print("-----------------------------------------------")
                print('Skeleton+LS:{}'.format(predicted))
                print("-----------------------------------------------")

                #Calculating Error

                # Ys = [] #t['YT']
                Ys = t["YT"]

                Yhats = []
                for xs in t['XT']:
                    
                    Yhat = 100

                    try:
                        
                        eqTmp = predicted + '' # copy eq
                        eqTmp = eqTmp.replace(' ','')
                        eqTmp = eqTmp.replace('\n','')

                        for i,x in enumerate(xs):
                            # replace xi with the value in the eq
                            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                            if ',' in eqTmp:
                                assert 'There is a , in the equation!'

                        Yhat = eval(eqTmp)
                        # Yhat = 0 if np.isnan(Yhat) else Yhat
                        # Yhat = 100 if np.isinf(Yhat) else Yhat

                    except:

                        print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                        Yhat = 100
                        # pdb.set_trace()

                    Yhats.append(Yhat)

                err = relativeErr(Ys,Yhats, info=True)

                # if type(err) is np.complex128 or np.complex:
                if type(err) is np.complex128:
                    err = abs(err.real)

                return err

                # resultDict[fName]['SymbolicGPT'].append(err)
            
        except KeyboardInterrupt:
            print('KeyboardInterrupt')


    ################################################################################################
    #                              #            Model              #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################

    @classmethod
    def instantiate_model(cls, args_index=1) -> GPT:
        args = None
        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9
        

        numVars             = args["numVars"]
        numYs               = args["numYs"]
        numPoints           = args["numPoints"] 
        embeddingSize       = args["embeddingSize"]
        method              = args["method"]
        variableEmbedding   = args["variableEmbedding"]
        block_size          = args["blockSize"]
        vocab_size          = args["vocab_size"]
        paddingID           = args["paddingID"]

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

        return model

    @classmethod
    def train_model(cls, args_index, model, train_dataset, val_dataset, device="mps"):

        args = None
        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9


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
        batchSize       = args["batchSize"]
        ckpt_path_dir   = args["ckpt_path_dir"]
        fName           = args["fName"]

        bestLoss        = None # if there is any model to load as pre-trained one
        numEpochs       = 20 # number of epochs to train the GPT+PT model
        ckptPath        = "{}/{}".format(ckpt_path_dir, fName)
        
        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(
                                max_epochs      = numEpochs, 
                                batch_size      = batchSize, 
                                learning_rate   = 6e-4,
                                lr_decay        = True, 
                                warmup_tokens   = 512*20, 
                                final_tokens    = 2*len(train_dataset) * blockSize,
                                num_workers     = 0,
                                ckpt_path       = ckptPath,
                            )

        trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss, device=device)

        # # load the best model before training
        # print('The following model {} has been loaded!'.format(ckptPath))
        # model.load_state_dict(torch.load(ckptPath))
        # model = model.eval().to(trainer.device)

        try:
            trainer.train()
        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        # load the best model
        print('The following model {} has been loaded!'.format(ckptPath))
        model.load_state_dict(torch.load(ckptPath))

        return model

    @classmethod
    def load_model(cls, args_index = 1):

        args = None

        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9


        model_dir_path  = args["model_dir_path"]
        model_path      = args["model_path"]
        path    = "{}/{}".format(model_dir_path, model_path)
        model   = cls.instantiate_model(args_index)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_state = torch.load(path, map_location=device)
        model.load_state_dict(model_state)
        model.to(device)

        return model


    ################################################################################################
    #                              #            Data               #                               #
    ################################################################################################
    #                              ####    ####    ####    ####    #                               # 
    ################################################################################################


    @classmethod
    def load_train_data(cls, args_index = 1, points_num = -1):

        args = None
        
        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9
    

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
        maxNumFiles     = 10

        path            = "{}/Train/*.json".format(data_dir)
        files           = glob.glob(path)[:maxNumFiles]
        text            = processDataFiles(files)
        chars           = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data       
        text            = text.split('\n') # convert the raw text to a set of examples
        trainText       = text[:-1] if len(text[-1]) == 0 else text

        random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
        
        
        if not points_num == -1:
            trainText = trainText[:points_num]
        
        train_dataset = CharDataset(trainText, 
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
        
        return train_dataset
    
    @classmethod
    def load_val_data(cls, args_index = 1, points_num = -1):

        args = None

        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9

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
        chars           = list(args["itos"].values())

        path    = "{}/Val/*.json".format(data_dir)

        files = glob.glob(path)
        textVal = processDataFiles([files[0]])
        textVal = textVal.split('\n') # convert the raw text to a set of examples
        
        if not points_num == -1:
            textVal = textVal[:points_num]    

        val_dataset = CharDataset(
                                    textVal, 
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
                                )
        return val_dataset
    
    @classmethod
    def load_test_data(cls, args_index = 1, points_num = -1):

        args = None

        if args_index == 1:
            args = cls.ARGS_1
        elif args_index == 2:
            args = cls.ARGS_2
        elif args_index == 3:
            args = cls.ARGS_3
        elif args_index == 5:
            args = cls.ARGS_5
        elif args_index == 9:
            args = cls.ARGS_9

        data_dir        = args["data_dir"]
        blockSize       = args["blockSize"]
        numVars         = args["numVars"]
        numYs           = args["numYs"]
        numPoints       = args["numPoints"] 
        target          = args["target"] # "EQ"
        addVars         = True if args["variableEmbedding"] == 'STR_VAR' else False
        const_range     = args["const_range"]
        trainRange      = args["trainRange"]
        decimals        = args["decimals"]
        chars           = list(args["itos"].values())

        path            = "{}/Test/*.json".format(data_dir)

        files           = glob.glob(path)
        textTest        = processDataFiles([files[0]])
        textTest        = textTest.split('\n') # convert the raw text to a set of examples

        if not points_num == -1:
            textTest = textTest[:points_num]

        test_dataset    = CharDataset(
                                    textTest, 
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
                                )
        
        return test_dataset
        
                
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
            "data_dir"              : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/Datasets/1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points",
            "model_dir_path"        : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/Models",
            "model_path"            : "XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
            "fName"                 : "XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
            "ckpt_path_dir"         : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/trained_models/var_1",
            
            "numEpochs"             : 20, # number of epochs to train the GPT+PT model
            "embeddingSize"         : 512, # the hidden dimension of the representation of both GPT and PT
            "numPoints"             : [30,31], # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
            "numVars"               : 1, # the dimenstion of input points x, if you don't know then use the maximum
            "numYs"                 : 1, # the dimension of output points y = f(x), if you don't know then use the maximum
            "blockSize"             : 64, # spatial extent of the model for its context
            "testBlockSize"         : 400,
            "batchSize"             : 128, # batch size of training data
            "target"                : 'Skeleton', #'Skeleton' #'EQ'
            "const_range"           : [-2.1, 2.1], # constant range to generate during training only if target is Skeleton
            "decimals"              : 8, # decimals of the points only if target is Skeleton
            "trainRange"            : [-3.0,3.0], # support range to generate during training only if target is Skeleton
            "dataDir"               : './datasets/',
            # "dataInfo"              : 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize),
            "titleTemplate"         : "{} equations of {} variables - Benchmark",
            "dataFolder"            : '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points',
            "addr"                  : './SavedModels/', # where to save model
            "method"                : 'EMB_SUM', # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
            "variableEmbedding"     : 'NOT_VAR', # NOT_VAR/LEA_EMB/STR_VAR
            "vocab_size"            : 49,
            "paddingID"             : 34,
            "size"                  : 498795,
            "paddingToken"          : '_',
            "itos"                  : {   
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

    ARGS_2 = {
        "data_dir"              : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/Datasets/2Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_200Points",
        "model_dir_path"        : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/Models",
        "model_path"            : "XYE_2Var_200-201Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
        "fName"                 : "XYE_2Var_200-201Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
        "ckpt_path_dir"         : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/trained_models/var_2",
        "numEpochs"             : 20,
        "embeddingSize"         : 512,
        "numPoints"             : [200,201],
        "numVars"               : 2,
        "numYs"                 : 1,
        "blockSize"             : 64,
        "testBlockSize"         : 400,
        "batchSize"             : 128,
        "target"                : 'Skeleton',
        "const_range"           : [-2.1, 2.1],
        "decimals"              : 8,
        "trainRange"            : [-3.0,3.0],
        "dataDir"               : './datasets/',
        # "dataInfo"             ,
        # "titleTemplate"         : "{} equations of {} variables - Benchmark",
        "dataFolder"            : "2Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_200Points",
        "addr"                  :'./SavedModels/',
        "method"                : 'EMB_SUM',
        "variableEmbedding"     : 'NOT_VAR',
        "vocab_size"            : 49,
        "paddingID"             : 34,
        "size"                  : 499035,
        "paddingToken"          : '_',
        "itos"                  : {
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
                                    48: '}'
                                }
        }

    ARGS_3 = {
        "data_dir"              : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/Datasets/3Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_500Points",
        "model_dir_path"        : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/Models",
        "model_path"            : "XYE_3Var_500-501Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
        "fName"                 : "XYE_3Var_500-501Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt",
        "ckpt_path_dir"         : "/Users/aminravanbakhsh/vscode/symbolicgpt/untracked_folder/trained_models/var_2",
        "numEpochs" : 20, # number of epochs to train the GPT+PT model
        "embeddingSize" : 512, # the hidden dimension of the representation of both GPT and PT
        "numPoints":[500,501], # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
        "numVars":3, # the dimenstion of input points x, if you don't know then use the maximum
        "numYs":1, # the dimension of output points y = f(x), if you don't know then use the maximum
        "blockSize": 200, # spatial extent of the model for its context
        "testBlockSize": 400,
        "batchSize": 128, # batch size of training data
        "target": 'Skeleton', #'Skeleton' #'EQ'
        "const_range": [-2.1, 2.1], # constant range to generate during training only if target is Skeleton
        "decimals": 8, # decimals of the points only if target is Skeleton
        "trainRange": [-3.0,3.0], # support range to generate during training only if target is Skeleton  
        "dataDir": './datasets/',
        "dataFolder": '3Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_500Points',
        "addr": './SavedModels/', # where to save model
        "method": 'EMB_SUM', # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation.
        "variableEmbedding": 'NOT_VAR',
        "vocab_size"            : 49,
        "paddingID"             : 34,
        "size"                  : 192594,
        "paddingToken"          : '_',
        "itos"                  : {
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
                                    48: '}'
                                }
        }

    ARGS_5 = {}

    ARGS_9 = {}

"""
numEpochs = 20 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=[500,501] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=3 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
testBlockSize = 400
batchSize = 128 # batch size of training data
target = 'Skeleton' #'Skeleton' #'EQ'
const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
decimals = 8 # decimals of the points only if target is Skeleton
trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '3Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_500Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation.
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
"""