import numpy as np
import random
import torch
import pickle
import json
import os
from tqdm import tqdm


class Load_Yelp():

    def __init__(self, train_rating_path, test_rating_path, test_negative_path):
        self.train_rating = self.load_rating(train_rating_path)
        self.test_rating = self.load_rating(test_rating_path)
        # self.test_negative = self.load_test_negative_from_file(test_negative_path)
        self.test_negative = self.load_test_negative()
        self.train_group = self.get_train_group()

    def load_rating(self, path):
        rating = np.loadtxt(path, delimiter='\t')
        record_count = len(rating[:, 0])
        user_count = int(max(rating[:, 0])) + 1
        item_count = int(max(rating[:, 1])) + 1
        print('Loaded:', path)
        print('Num of users:', user_count)
        print('Num of items:', item_count)
        print('Data sparsity:', record_count / (user_count * item_count))
        # remove the last column: timestamp
        return torch.from_numpy(rating[:, :-2])

    def get_train_group(self):
        neg = {}

        # get negative samples
#         if os.path.exists('./Data/train_neg_dict.pk'):
#             neg = self.load_neg_dict_from_pickle('./Data/train_neg_dict.pk')
#         else:
#             neg = self.get_negative(self.train_rating, 1000)
#             self.save_neg_dict_to_pickle(neg, './Data/train_neg_dict.pk')
            
        # get negative samples
        if os.path.exists('./Data/train_neg_dict.json'):
            neg = self.load_neg_dict_from_json('./Data/train_neg_dict.json')            
        else:
            neg = self.get_negative(self.train_rating, 1000)
            self.save_neg_dict_to_json(neg, './Data/train_neg_dict.json')

        # save negative sample for resampling
        self.train_negative = neg
        
        record_count = len(self.train_rating[:, 0])
        groups = []
        for r in range(record_count):
            u = int(self.train_rating[r, 0])
            i = int(self.train_rating[r, 1])
            j = int(random.sample(neg[u], 1)[0])
            groups.append([u, i, j])
        return torch.tensor(groups)

    def resample_train_group(self):
        record_count = len(self.train_rating[:, 0])
        groups = []
        for r in range(record_count):
            u = int(self.train_rating[r, 0])
            i = int(self.train_rating[r, 1])
            j = int(random.sample(self.train_negative[u], 1)[0])
            groups.append([u, i, j])
        return torch.tensor(groups)
        
    def get_negative(self, data, sample_count):
        print('Calculating negative samples...')
        neg = {}
        record_count = len(data[:, 0])
        user_count = int(max(data[:, 0])) + 1
        item_count = int(max(data[:, 1])) + 1
        for u in range(user_count):
            neg[u] = []
        last_u = 0
        neg[0] = set(range(item_count))
        # record_count = 100
        for r in tqdm(range(record_count)):
            u = int(data[r, 0])
            if u != last_u:
                neg[last_u] = random.sample(list(neg[last_u]), sample_count)
                neg[u] = set(range(item_count))
            last_u = u
            i = int(data[r, 1])
            neg[u] = neg[u] - set([i])
        # neg[last_u] = set(range(item_count))
        neg[last_u] = random.sample(list(neg[last_u]), sample_count)

        return neg

    def load_test_negative_from_file(self, path):
        result = {}
        neg = np.loadtxt('./Data/yelp.test.negative', delimiter='\t', dtype='str')
        print('Loaded:', path)
        print('1000 negative test cases for each user')
        record_count = len(neg)
        for r in tqdm(range(record_count)):
            ui = tuple(map(int, neg[r, 0][1:-1].split(',')))
            u = ui[0]
            i = ui[1]
            if u not in result:
                result[u] = []
            else:
                result[u].append(list(map(int, neg[r, 1:])))
                result[u].append(i)
        return result

    def load_test_negative(self):
        neg = {}

        # get negative sample
#         if os.path.exists('./Data/test_neg_dict.pk'):
#             neg = self.load_neg_dict_from_pickle('./Data/test_neg_dict.pk')            
#         else:
#             neg = self.get_negative(self.test_rating, 999)
#             self.save_neg_dict_to_pickle(neg, './Data/test_neg_dict.pk')

        # get negative samples
        if os.path.exists('./Data/test_neg_dict.json'):
            neg = self.load_neg_dict_from_json('./Data/test_neg_dict.json')
        else:
            neg = self.get_negative(self.test_rating, 999)
            self.save_neg_dict_to_json(neg, './Data/test_neg_dict.json')

        # append a positive sample
        user_count = int(max(self.test_rating[:, 0])) + 1
        for u in range(user_count):
            neg[u].append(self.test_rating[u, 1])
        return neg

    def save_neg_dict_to_json(self, neg, path):
        print('Saving negative samples to file:', path)
        with open(path, "w") as f:
            json.dump(neg, f, sort_keys=True)
    
    def save_neg_dict_to_pickle(self, neg, path):
        print('Saving negative samples to file:', path)
        with open(path, "wb") as f:
            pickle.dump(neg, f)

    def load_neg_dict_from_json(self, path):
        print('Loading negative samples from file', path)
        with open(path, 'r') as f:
            d = json.load(f)
            keys = list(map(int, d.keys()))
            neg = {}
            for i in range(len(keys)):
               neg[keys[i]] = list(d.values())[i]
            return neg

    def load_neg_dict_from_pickle(self, path):
        print('Loading negative samples from file', path)
        with open(path, 'rb') as f:
            return pickle.load(f)