from .bert import BERTTrainer
from .gru4rec import GRU4RECTrainer
from .sasrec import SASTrainer
from .dan import DANTrainer
TRAINERS = {
    DANTrainer.code(): DANTrainer,
    SASTrainer.code(): SASTrainer,
    GRU4RECTrainer.code(): GRU4RECTrainer,
    BERTTrainer.code(): BERTTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
