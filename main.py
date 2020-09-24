from __future__ import absolute_import, print_function
import torch

import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.memae import MEMAE
from model.dagmm import DAGMM
from model.ae import AE
from model.vae import VAE

from util.data_utils import *
from util.perf_utils import *

from sklearn.metrics import average_precision_score

ATK=1
SAFE=0

# Argument Setting
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size for train and test")
parser.add_argument("--seed", default=42, type=int,
                    help="random seed for reproductability")
parser.add_argument("--lr", default=0.0001, type=float,
                    help="learning rate")
parser.add_argument("--epoch", default=5, type=int,
                    help="training epochs")
parser.add_argument("--model", default="memae", type=str,
                    help="model list = ['dagmm', 'memae']")

args = parser.parse_args()

# Fix seed
set_seed(args.seed)
device = torch.device('cuda:0')

# Model list
model_all = {
    'dagmm': DAGMM(),
    'memae': MEMAE(),
    'ae': AE(),
    'vae': VAE()
}

kdd_path='../kdd_120'
data_name='10'
# data_name='og'

x_train,y_train=get_hdf5_data(kdd_path+'/processed/train_{}.hdf5'.format(data_name),labeled=True)
print("Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train==1].shape[0]))

x_test,y_test=get_hdf5_data(kdd_path+'/processed/test_{}.hdf5'.format(data_name),labeled=True)
print("Test: Normal:{}, Atk:{}".format(x_test[y_test==0].shape[0],x_test[y_test==1].shape[0]))

#Split Data
x_train,x_val,y_train,y_val=split_data(x_train,y_train)
print("Split Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train==1].shape[0]))
print("Split Val: Normal:{}, Atk:{}".format(x_val[y_val==0].shape[0],x_val[y_val==1].shape[0]))

#Filter Label
#Train with only Atk Samples
x_train,y_train=filter_label(x_train,y_train,select_label=ATK)

#Load to Cuda
x_train = torch.from_numpy(x_train).float().to(device)
val_cuda= torch.from_numpy(x_val).float().to(device)

data_sampler = RandomSampler(x_train)
data_loader = DataLoader(x_train, sampler=data_sampler, batch_size=64)

model = model_all[args.model].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.zero_grad()

model.train(True)

# Train
for epoch in range(args.epoch):
    epoch_loss = []
    for step, batch in enumerate(data_loader):
        target = batch.type(torch.float32)

        outputs = model(target)
        loss = model.compute_loss(outputs, target)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss.append(loss.item())

    #Validation loss
    model.eval()
    with torch.no_grad():
        outputs = model(val_cuda)
        val_loss = model.compute_loss(outputs, val_cuda)

    print("epoch {}: Train: {:.5f}, Val: {:.5f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss),val_loss))


model.train(False)


# Test
x_test_cuda = torch.from_numpy(x_test).float().to(device)
eval_sampler = SequentialSampler(x_test_cuda)
eval_dataloader = DataLoader(x_test_cuda, sampler=eval_sampler, batch_size=64)

model.eval()
error =
pred_test=[]
for batch in eval_dataloader:
    target = batch.type(torch.float32)

    outputs = model(target)
    pred_test.append(outputs['output'].cpu().detach().numpy())
    batch_error = model.compute_batch_error(outputs, target)

    error += batch_error.detach().tolist()

# visualize
test_recon=np.concatenate(pred_test)
print(test_recon.shape)

#Evaluate Test Data
test_dist=np.mean(np.square(x_test-test_recon),axis=1)

#Distance High -> Anomaly
print('Average Precision',average_precision_score(y_test, test_dist, pos_label=SAFE))
make_roc(test_dist,y_test,ans_label=SAFE)
