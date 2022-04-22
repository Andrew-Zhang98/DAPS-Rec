from .bert import BERTModel
from .gru4rec import GRU4RECModel
from .sasrec import SASRecModel
from .dansrec import DANRecModel


MODELS = {
    DANRecModel.code(): DANRecModel,
    SASRecModel.code(): SASRecModel,
    GRU4RECModel.code(): GRU4RECModel,
    BERTModel.code(): BERTModel,
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
