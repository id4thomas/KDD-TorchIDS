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
parser.add_argument("--model", default="dagmm", type=str,
                    help="model list = ['dagmm', 'memae']")

args = parser.parse_args()

# Fix seed
set_seed(args.seed)
device = torch.device('cuda:0')

# Model list
model_all = {
    'dagmm': DAGMM(),
}

#Load Data
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
#Train with only train_label Samples
TRAIN=ATK
ANOMALY=SAFE
x_train,y_train=filter_label(x_train,y_train,select_label=TRAIN)

#Undersampling Validation
x_val_atk=x_val[y_val==ATK]
y_val_atk=y_val[y_val==ATK]

x_val_safe=x_val[y_val==SAFE]
y_val_safe=y_val[y_val==SAFE]
x_val,y_val=under_sampling(x_val_atk,y_val_atk,x_val_safe,y_val_safe)

print("Sampled Val: Normal:{}, Atk:{}".format(x_val[y_val==0].shape[0],x_val[y_val==1].shape[0]))

#Load to Cuda
x_train = torch.from_numpy(x_train).float().to(device)
val_cuda= torch.from_numpy(x_val).float().to(device)

data_sampler = RandomSampler(x_train)
data_loader = DataLoader(x_train, sampler=data_sampler, batch_size=args.batch_size)

#Load Model
model = model_all[args.model].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model.zero_grad()

model.train(True)

# Train
for epoch in range(args.epoch):
    epoch_loss = []
    epoch_recon=[]
    epoch_energy=[]
    print("\nTraining Epoch",epoch+1)
    for step, batch in enumerate(data_loader):
        target = batch.type(torch.float32)

        outputs = model(target)
        losses = model.compute_loss(outputs, target)
        loss=losses['loss']
        train_recon=losses['recon']
        sample_energy=losses['energy']

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss.append(loss.item())
        epoch_recon.append(train_recon)
        epoch_energy.append(sample_energy)

    #Validation loss
    model.eval()
    with torch.no_grad():
        outputs = model(val_cuda)
        val_losses = model.compute_loss(outputs, val_cuda)
        val_loss=val_losses['loss']
        #Calculate Energy
        val_energy,_=model.compute_energy(outputs['z'], outputs['phi'], outputs['mu'], outputs['cov'],size_average=False)
        val_dist=val_energy.cpu().detach().numpy()

    print("\tepoch {}: Train: {:.5f}, Val: {:.5f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss),val_loss))
    print("\tSample Energy: {:.5f}".format(sum(epoch_energy)/len(epoch_energy)))
    print("\tTrain Recon: {:.5f}".format(sum(epoch_recon)/len(epoch_recon)))

    #Validation Perf
    print('\nVAL Average Precision',average_precision_score(y_val, val_dist, pos_label=ANOMALY))

    #Energy Highest 20% -> Anomaly
    thresh = np.percentile(val_dist, 100 - 20)
    print("Threshold :", thresh)
    y_pred=np.zeros_like(y_val)
    y_pred[val_dist>thresh]=ANOMALY
    y_pred[val_dist<=thresh]=TRAIN
    prf(y_val,y_pred)

model.train(False)

exit()
#Currently Working on Below

# Test
x_test_cuda = torch.from_numpy(x_test).float().to(device)
eval_sampler = SequentialSampler(x_test_cuda)
eval_dataloader = DataLoader(x_test_cuda, sampler=eval_sampler, batch_size=64)

model.eval()
error = []
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
print('Average Precision',average_precision_score(y_test, test_dist, pos_label=ANOMALY))
make_roc(test_dist,y_test,ans_label=ANOMALY)
