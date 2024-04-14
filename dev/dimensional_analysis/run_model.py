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



def test_001():
    
    TEST_INDEX = 1  
    print("\n-----------------------------------------------")
    print("Start test_{}:".format(TEST_INDEX))
    print("-----------------------------------------------")

    embeddingSize = 512
    numPoints = [20,250]
    numVars = 9
    numYs = 1
    vocab_size = 53
    blockSize = 32

    method = 'EMB_SUM'
    variableEmbedding = 'NOT_VAR'

    # train dataset

    # create the model
    pconf = PointNetConfig(
                    embeddingSize       = embeddingSize, 
                    numberofPoints      = numPoints[1] - 1, 
                    numberofVars        = numVars, 
                    numberofYs          = numYs,
                    method              = method,
                    variableEmbedding   =variableEmbedding,
                    )
    
    # config
    mconf = GPTConfig(
        vocab_size, 
        blockSize, 
        n_layer=8,
        n_head=8,
        n_embd=embeddingSize, 
        padding_idx=train_dataset.paddingID)

    model = GPT(mconf, pconf)




def test_002():

    untrackedDir = "/home/amin/vscodes/symbolicgpt/untracked_folder"
    dataDir = "/home/amin/vscodes/symbolicgpt/untracked_folder/datasets"
    dataFolder = "sample_dataset"
    dataTestFolder = "sample_dataset/Test"
    addr = untrackedDir + '/models' # where to save model

    method = 'EMB_SUM'
    variableEmbedding = 'NOT_VAR'
    addVars = True if variableEmbedding == 'STR_VAR' else False

    addVars = False


    batchSize = 128 # batch size of training data
    target = 'Skeleton' #'Skeleton' #'EQ'
    const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
    decimals = 8 # decimals of the points only if target is Skeleton
    trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton

    blockSize = 200
    testBlockSize = 800

    maxNumFiles = 10
    embeddingSize = 512
    numPoints = [20,250]
    numVars = 9
    numYs = 1

    text = ""

    path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
    
    files = glob.glob(path)[:maxNumFiles]
    text = processDataFiles(files)

    chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
    text = text.split('\n') # convert the raw text to a set of examples
    trainText = text[:-1] if len(text[-1]) == 0 else text
    random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment

    train_dataset = CharDataset(
                                data=           text, 
                                block_size=     blockSize, 
                                chars=          chars, 
                                numVars=        numVars, 
                                numYs=          numYs, 
                                numPoints=      numPoints, 
                                target=         target, 
                                addVars=        addVars,
                                const_range=    const_range, 
                                xRange=         trainRange, 
                                decimals=       decimals, 
                                augment=        False) 

    vocab_size = train_dataset.vocab_size
    vocab_size = 54
    

    # create the model
    pconf = PointNetConfig(
                    embeddingSize       =embeddingSize, 
                    numberofPoints      =numPoints[1]-1, 
                    numberofVars        =numVars, 
                    numberofYs          =numYs,
                    method              =method,
                    variableEmbedding   =variableEmbedding)
    
    # config
    mconf = GPTConfig(
        vocab_size, 
        blockSize, 
        n_layer=8,
        n_head=8,
        n_embd=embeddingSize, 
        padding_idx=train_dataset.paddingID)

    model = GPT(mconf, pconf)

    dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize'.format(numVars, numPoints[0], numPoints[1], embeddingSize)
    fName = '{}_SymbolicGPT_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                             'GPT_PT_{}_{}'.format(method, target), 
                                             'Padding',
                                             variableEmbedding)

    ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
    print(ckptPath)
    
    model.load_state_dict(torch.load(ckptPath))

    ## Test the model
    # alright, let's sample some character-level symbolic GPT 
    loader = torch.utils.data.DataLoader(
                                    train_dataset, 
                                    shuffle=False, 
                                    pin_memory=True,
                                    batch_size=1,
                                    num_workers=0)
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

    pdb.set_trace()

    DEVICE = "cpu"

    resultDict = {}
    try:
        with open(fName, 'w', encoding="utf-8") as o:
            resultDict[fName] = {'SymbolicGPT':[]}

            for i, batch in enumerate(loader):
                    
                inputs,outputs,points,variables = batch

                print('Test Case {}.'.format(i))
                o.write('Test Case {}/{}.\n'.format(i,len(textTest)-1))

                t = json.loads(textTest[i])

                inputs = inputs[:,0:1].to(DEVICE)
                points = points.to(DEVICE)
                variables = variables.to(DEVICE)

                outputsHat = sample_from_model(
                            model, 
                            inputs, 
                            blockSize, 
                            points=points,
                            variables=variables,
                            temperature=1.0, 
                            sample=True, 
                            top_k=0.0,
                            top_p=0.7)[0]


                # filter out predicted
                target = ''.join([train_dataset.itos[int(i)] for i in outputs[0]])
                predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])

                if variableEmbedding == 'STR_VAR':
                    target = target.split(':')[-1]
                    predicted = predicted.split(':')[-1]

                target = target.strip(train_dataset.paddingToken).split('>')
                target = target[0] #if len(target[0])>=1 else target[1]
                target = target.strip('<').strip(">")
                predicted = predicted.strip(train_dataset.paddingToken).split('>')
                predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
                predicted = predicted.strip('<').strip(">")
                
                print('Target:{}\nSkeleton:{}'.format(target, predicted))
                
                o.write('{}\n'.format(target))
                o.write('{}:\n'.format('SymbolicGPT'))
                o.write('{}\n'.format(predicted))

                # train a regressor to find the constants (too slow)
                c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
                
                # c[-1] = 0 # initialize the constant as zero
                b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables

                pdb.set_trace()

                # try:
                #     if len(c) != 0:
                #         # This is the bottleneck in our algorithm
                #         # for easier comparison, we are using minimize package  
                #         cHat = minimize(lossFunc, c, #bounds=b,
                #                     args=(predicted, t['X'], t['Y'])) 
            
                #         predicted = predicted.replace('C','{}').format(*cHat.x)

                # except ValueError:
                #     raise 'Err: Wrong Equation {}'.format(predicted)
                
                # except Exception as e:
                #     raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)
                
                try:
                    if len(c) != 0:
                        # This is the bottleneck in our algorithm
                        # for easier comparison, we are using minimize package  

                        args = (predicted, t['X'], t['Y'])


                        pdb.set_trace()

                        cHat = minimize(lossFunc, 
                                        c,
                                        bounds=b,
                                        args = (predicted, t['X'], t['Y']))

                        predicted = predicted.replace('C','{}').format(*cHat.x)

                except ValueError:
                    # If you're catching a ValueError, you might want to do some specific logging or handling here
                    raise ValueError(f'Err: Wrong Equation {predicted}')
                except Exception as e:
                    # This is a more generic exception handler
                    raise Exception(f'Err: Wrong Equation {predicted}, Err: {e}')

                

                print('Skeleton+LS:{}'.format(predicted))

                Ys = [] #t['YT']
                Yhats = []
                for xs in t['XT']:
                    try:
                        eqTmp = target + '' # copy eq
                        eqTmp = eqTmp.replace(' ','')
                        eqTmp = eqTmp.replace('\n','')
                        for i,x in enumerate(xs):
                            # replace xi with the value in the eq
                            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                            if ',' in eqTmp:
                                assert 'There is a , in the equation!'
                        YEval = eval(eqTmp)
                        # YEval = 0 if np.isnan(YEval) else YEval
                        # YEval = 100 if np.isinf(YEval) else YEval
                    except:
                        print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                        print(i)
                        raise
                        continue # if there is any point in the target equation that has any problem, ignore it
                        YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
                    Ys.append(YEval)
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
                    Yhats.append(Yhat)
                err = relativeErr(Ys,Yhats, info=True)

                if type(err) is np.complex128 or np.complex:
                    err = abs(err.real)

                resultDict[fName]['SymbolicGPT'].append(err)

                o.write('{}\n{}\n\n'.format( 
                                        predicted,
                                        err
                                        ))

                print('Err:{}'.format(err))
                
                print('') # just an empty line
        print('Avg Err:{}'.format(np.mean(resultDict[fName]['SymbolicGPT'])))
        
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    pdb.set_trace()
                

"""

'/home/amin/vscodes/symbolicgpt/untracked_folder/datasets/models/XYE_9Var_20-250Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt'
/home/amin/vscodes/symbolicgpt/untracked_folder//models/XYE_9Var_20-250Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt

    XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
    XYE_2Var_200-201Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
    XYE_3Var_500-501Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
    XYE_5Var_10-200Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt
    XYE_9Var_20-250Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt


"""