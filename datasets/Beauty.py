from .base import AbstractDataset

import pandas as pd

from datetime import date


class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'Beauty'

    @classmethod
    def url(cls):
        # return "***"
        return ''
        # return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        # folder_path = self._get_rawdata_folder_path()
        # file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv('/home/ruoyan/code/BERT4Rec-VAE-Pytorch/Data/Beauty/pre_beauty.txt', sep=' ', header=None)
        # f = pd.read_csv('/home/ruoyan/code/SASRec/data/Beauty2.txt', sep=' ', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        # import pdb; pdb.set_trace()
        return df


