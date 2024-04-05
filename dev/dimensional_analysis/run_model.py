from models import GPT, GPTConfig, PointNetConfig
import pdb



def test_001():
    
    TEST_INDEX = 1  
    print("\n-----------------------------------------------")
    print("Start test_{}:".format(TEST_INDEX))
    print("-----------------------------------------------")

    embeddingSize = 512
    numPoints = [20,250]
    numVars = 9
    numYs = 1

    method = 'EMB_SUM'
    variableEmbedding = 'NOT_VAR'

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