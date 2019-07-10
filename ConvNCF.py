import load_dataset
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

def show_curve(ys, title):
    """plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: Loss or Accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()

def show_metric(ys, title):
    """plot curlve for HR and NDCG
    Args:
        ys: hr or ndcg list
        title: HR@k or NDCG@k
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('User')
    plt.ylabel('{} Value'.format(title))
    plt.show()

class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count):
        super(ConvNCF, self).__init__()

        # some variables
        self.user_count = user_count
        self.item_count = item_count

        # embedding setting
        self.embedding_size = 64

        # init target matrix of matrix factorization
#         self.P = nn.Parameter(torch.randn((self.user_count, self.embedding_size)).cuda(), requires_grad=True)
#         self.Q = nn.Parameter(torch.randn((self.item_count, self.embedding_size)).cuda(), requires_grad=True)
        self.P = nn.Embedding(self.user_count, self.embedding_size).cuda()
        self.Q = nn.Embedding(self.item_count, self.embedding_size).cuda()

        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.cnn = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )

        # fully-connected layer, used to predict
        self.fc = nn.Linear(32, 1)
        
        # dropout
#         self.drop_prop = 0.5
#         self.dropout = nn.Dropout(drop_prop)

    def forward(self, user_ids, item_ids, is_pretrain):

        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        # get embeddings, simplify one-hot to index directly
#         user_embeddings = self.P[user_ids]
#         item_embeddings = self.Q[item_ids]
        user_embeddings = self.P(torch.tensor(user_ids).cuda())
        item_embeddings = self.Q(torch.tensor(item_ids).cuda())
        
#         # inner product
#         prediction = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1)
            
        if is_pretrain:
            # inner product
            prediction = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1)
        else:
            # outer product
            # interaction_map = torch.ger(user_embeddings, item_embeddings) # ger is 1d
            interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
            interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))

            # cnn
            feature_map = self.cnn(interaction_map)  # output: batch_size * 32 * 1 * 1
            feature_vec = feature_map.view((-1, 32))

            # fc
            prediction = self.fc(feature_vec)
            prediction = prediction.view((-1))

#         print('is_pretrain:', is_pretrain, prediction.shape)
        return prediction


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        
#         print('loss:', loss)
        return loss


def train():
    lr = 0.5
    epoches = 5000
    batch_size = 2048
    losses = []
    accuracies = []

    model.train()
    
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=4)
    
    bpr_loss = BPRLoss().cuda()

    for epoch in range(epoches):    
#         yelp.train_group = yelp.resample_train_group()
#         train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=4)
        
        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader) as t:
                for batch_idx, train_data in enumerate(t):                    
        #             train_data = Variable(train_data.cuda())
                    user_ids = Variable(train_data[:, 0].cuda())
                    pos_item_ids = Variable(train_data[:, 1].cuda())
                    neg_item_ids = Variable(train_data[:, 2].cuda())
                    
                    optimizer.zero_grad()
                    
#                     # pretrain bpr
#                     pos_preds = model(user_ids, pos_item_ids, True)
#                     neg_preds = model(user_ids, neg_item_ids, True) 

                    if epoch < 100:    
                        # pretrain bpr
                        pos_preds = model(user_ids, pos_item_ids, True)
                        neg_preds = model(user_ids, neg_item_ids, True)                        
                    else:
                        # train convncf                        
                        pos_preds = model(user_ids, pos_item_ids, False)
                        neg_preds = model(user_ids, neg_item_ids, False)
                        
                    loss = bpr_loss(pos_preds, neg_preds)                    
                    total_loss += loss.item()                    
                    loss.backward()
                    optimizer.step()
                    
                    accuracy = float(((pos_preds - neg_preds) > 0).sum()) / float(len(pos_preds))
                    total_acc += accuracy
        #             if batch_idx % 20 == 0:
        #                 print('Epoch:', epo.ch, 'Batch:', batch_idx, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])                                
        except KeyboardInterrupt:
            t.close()
            raise
        
        losses.append(total_loss / (batch_idx + 1))
        accuracies.append(total_acc / (batch_idx + 1))
        print('Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])  
        if epoch % 50 == 0 and epoch != 0:      
            show_curve(losses, 'train loss')
            show_curve(accuracies, 'train acc')
            evaluate()
        if epoch == 3 or epoch == 20:
            evaluate()
        scheduler.step()
              

def evaluate():
    model.eval()
    hr5 = []
    hr10 = []
    hr20 = []
    ndcg5 = []
    ndcg10 = []
    ndcg20 = []

    user_count = len(yelp.test_negative)
    try:
        with tqdm(range(user_count)) as t:
            for u in t:
                item_ids = torch.tensor(yelp.test_negative[u]).cuda()
                user_ids = torch.tensor([u] * len(item_ids)).cuda()
                predictions = model(user_ids, item_ids, False)
                topv, topi = torch.topk(predictions, 20, dim=0)
                hr, ndcg = scoreK(topi, 5)
                hr5.append(hr)
                ndcg5.append(ndcg)
                hr, ndcg = scoreK(topi, 10)
                hr10.append(hr)
                ndcg10.append(ndcg)
                hr, ndcg = scoreK(topi, 20)
                hr20.append(hr)
                ndcg20.append(ndcg)

            show_metric(hr5, 'HR@5')
            show_metric(ndcg5, 'NDCG@5')
            show_metric(hr10, 'HR@10')
            show_metric(ndcg10, 'NDCG@10')
            show_metric(hr20, 'HR@20')
            show_metric(ndcg20, 'NDCG@20')            
            print('HR@5:', sum(hr5) / len(hr5))
            print('NDCG@5:', sum(ndcg5) / len(ndcg5))
            print('HR@10:', sum(hr10) / len(hr10))
            print('NDCG@10:', sum(ndcg10) / len(ndcg10))
            print('HR@20:', sum(hr20) / len(hr20))
            print('NDCG@20:', sum(ndcg20) / len(ndcg20))
    except KeyboardInterrupt:
        t.close()
        raise

# def scoreK(topi, k):
#     hr = 1.0 if 999 in topi[0:k, 0] else 0.0
#     if hr:
#         ndcg = math.log(2) / math.log(topi[:, 0].tolist().index(999) + 2)
#     else:
#         ndcg = 0
#     # auc = 1 - (position * 1. / negs)
#     return hr, ndcg#, auc

def scoreK(topi, k):
    hr = 1.0 if 999 in topi[0:k] else 0.0
    if hr:
        ndcg = math.log(2) / math.log(topi.tolist().index(999) + 2)
    else:
        ndcg = 0
    # auc = 1 - (position * 1. / negs)
    return hr, ndcg#, auc


if __name__ == '__main__':
    # torch.set_num_threads(12)
    print('Data loading...')
    yelp = load_dataset.Load_Yelp('./Data/yelp.train.rating', './Data/yelp.test.rating', './Data/yelp.test.negative')
    print('Data loaded')

    print('=' * 50)

    print('Model initializing...')
    model = ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1).cuda()

    print('Model initialized')

    print('=' * 50)

    print('Model training...')
    train()
    print('Model trained')

    print('=' * 50)

    print('Model evaluating...')
    evaluate()
    print('Model evaluated')