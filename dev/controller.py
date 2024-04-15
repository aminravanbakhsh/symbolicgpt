from pipeline import Pipeline
import pdb


def train_from_scratch(args_index = 1):

    pdb.set_trace()

    train_dataset   = Pipeline.load_train_data(args_index)
    pdb.set_trace()
    model           = Pipeline.instantiate_model(args_index)
    trained_model   = Pipeline.train_model(args_index, model, train_dataset)

    pdb.set_trace()


# train_from_scratch()

pdb.set_trace()

val_dataset = Pipeline.load_val_data(args_index=1)

pdb.set_trace()