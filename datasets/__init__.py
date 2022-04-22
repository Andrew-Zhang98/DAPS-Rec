from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .Steam import SteamDataset
from .Beauty import BeautyDataset
from .Video import VideoDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    SteamDataset.code(): SteamDataset,
    BeautyDataset.code(): BeautyDataset,
    VideoDataset.code(): VideoDataset
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
