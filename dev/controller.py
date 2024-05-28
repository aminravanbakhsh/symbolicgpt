from pipeline import Pipeline
import pdb


def train_from_scratch(args_index = 1):

    train_dataset   = Pipeline.load_train_data(args_index, 10)
    # pdb.set_trace()
    val_dataset     = Pipeline.load_val_data(args_index, 10)
    model           = Pipeline.instantiate_model(args_index)
    trained_model   = Pipeline.train_model(args_index, model, train_dataset, val_dataset, device="gpu")

    pdb.set_trace()
    return trained_model

def test_model(arg_index = 1):

    test_data       = Pipeline.load_test_data(points_num=10)
    error = Pipeline.eval_model(test_data, args_index=1)
    
    pdb.set_trace()

# train_from_scratch(3)
test_model(3)