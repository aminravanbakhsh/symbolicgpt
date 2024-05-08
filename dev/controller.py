from pipeline import Pipeline
import pdb


def train_from_scratch(args_index = 1):

    train_dataset   = Pipeline.load_train_data(args_index)
    val_dataset     = Pipeline.load_val_data(args_index)
    model           = Pipeline.instantiate_model(args_index)
    
    trained_model   = Pipeline.train_model(args_index, model, train_dataset, val_dataset, device="gpu")

    pdb.set_trace()

def test_001(arg_index = 1):

    model           = Pipeline.load_model()

    test_data       = Pipeline.load_test_data()
    
    Pipeline.eval_model(test_data)
    
    pdb.set_trace()

pdb.set_trace()
test_001()
pdb.set_trace()
