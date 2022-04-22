from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter
import numpy as np

class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popular_items, prob = self.items_by_popularity()

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            while len(samples)< 1+self.sample_size:
                sampled_ids = np.random.choice(popular_items, self.sample_size, p=prob)
                sampled_ids = [x for x in sampled_ids if x not in seen and x not in samples]
                samples.extend(sampled_ids[:])
                
            samples = samples[:self.sample_size]
            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        prob = []
        for p in popular_items:
            prob.append(popularity[p])
        sum_p = np.sum(prob)
        prob = [i/sum_p for i in prob]
        return popular_items, prob


# class PopularNegativeSampler(AbstractNegativeSampler):
#     @classmethod
#     def code(cls):
#         return 'popular'

#     def generate_negative_samples(self):
#         popular_items = self.items_by_popularity()

#         negative_samples = {}
#         print('Sampling negative items')
#         for user in trange(self.user_count):
#             seen = set(self.train[user])
#             seen.update(self.val[user])
#             seen.update(self.test[user])

#             samples = []
#             for item in popular_items:
#                 if len(samples) == self.sample_size:
#                     break
#                 if item in seen:
#                     continue
#                 samples.append(item)

#             negative_samples[user] = samples

#         return negative_samples

#     def items_by_popularity(self):
#         popularity = Counter()
#         for user in range(self.user_count):
#             popularity.update(self.train[user])
#             popularity.update(self.val[user])
#             popularity.update(self.test[user])
#         popular_items = sorted(popularity, key=popularity.get, reverse=True)
#         return popular_items