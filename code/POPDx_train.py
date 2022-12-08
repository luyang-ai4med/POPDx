from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import ModelSaving
from models import POPDxModel

def args_argument():
    parser = argparse.ArgumentParser(description=
    '''
    The script to train POPDx. 
    Please specify the train/val datasets path in the python script.
    ''',
    usage='use "python %(prog)s --help" for more information')
    parser.add_argument('-d', '--save_dir', required=True, help='The folder to save the trained POPDx model e.g. "./save/POPDx_train"')
    parser.add_argument('-s', '--hidden_size', type=int, default=150, help='Default hidden size is 150.')
    parser.add_argument('--use_gpu', default=True, help='Default setup is to use GPU.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Default learning rate is 0.0001')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.000, help='Default learning rate is 0')
    args = parser.parse_args()
    return args    

def load_data():
    data_folder = '/oak/stanford/groups/rbaltman/luyang/scripts/rareseeker/PheCode/data/'
    label_emb_file = 'phecode_label_embed.npy'
    train_feature_file = 'train_feature.npy'
    train_label_file = 'train_phecode_labels.npy'
    val_feature_file = 'val_feature.npy'
    val_label_file = 'val_phecode_labels.npy'
    train_feature = np.load(os.path.join(data_folder, train_feature_file), allow_pickle=True)
    val_feature = np.load(os.path.join(data_folder, val_feature_file), allow_pickle=True)
    train_label = np.load(os.path.join(data_folder, train_label_file))
    val_label = np.load(os.path.join(data_folder, val_label_file))
    label_emb = np.load(os.path.join(data_folder, label_emb_file), allow_pickle=True)
    return train_feature, val_feature, train_label, val_label, label_emb


def train(train_feature, val_feature, train_label, val_label, label_emb, 
          use_cuda=True, hidden_size=150, learning_rate=0.0001, weight_decay=0, save_dir=''):
    

    device = torch.device("cuda:0" if use_cuda else "cpu")
    label_emb = torch.tensor(label_emb, dtype=torch.float, device=device)
    net = POPDxModel(train_feature.shape[1], train_label.shape[1], hidden_size, label_emb)
    net.initialize()
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    traindata = Dataset(train_feature, train_label)
    valdata = Dataset(val_feature, val_label)
    train_loader = DataLoader(dataset = traindata, batch_size = 128, shuffle = True)
    val_loader = DataLoader(dataset = valdata, batch_size = 128, shuffle = True)
    
    n_epochs = 100000 
    early_break = ModelSaving(waiting=5, printing=True)
    train = []
    val = [] 
    val_lowest = np.inf
    save_dir = save_dir
    for epoch in range(n_epochs):
        # training the model 
        print('starting epoch ' + str(epoch))
        net.train()
        losses = []
        for batch_idx, (train_inputs, train_labels) in tqdm(enumerate(train_loader)):
            train_inputs, train_labels = Variable(train_inputs.to(device)), Variable(train_labels.to(device))
            train_inputs.requires_grad_()
            train_outputs = net(train_inputs)
            loss = criterion(train_outputs, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
            #print('[%d/%d] Training Loss: %.3f' % (epoch+1, batch_idx, loss))
        print('[%d] Training Loss: %.3f' % (epoch+1, loss))

        #validating the model
        net.eval()  
        val_losses = []
        for batch_idx, (val_inputs,val_labels) in tqdm(enumerate(val_loader)):
            val_inputs, val_labels = Variable(val_inputs.to(device)), Variable(val_labels.to(device))
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_losses.append(val_loss.data.mean().item())
            #print('[%d/%d] Validation Loss: %.3f' % (epoch+1, batch_idx, val_loss))
        print('[%d] Validation Loss: %.3f' % (epoch+1, val_loss))

        train.append(losses)
        val.append(val_losses)
        if np.mean(val_losses) < val_lowest:
            val_lowest = np.mean(val_losses)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.mean(losses),
                    'val_loss': np.mean(val_losses)
                    }, os.path.join(save_dir, 'best_classifier.pth.tar'))
            print(str(val_lowest), 'saved')

        early_break(np.mean(val_losses), net)

        if early_break.save:
            print("Maximum waiting reached. Break the training.")
            break
                            
    train_L = [np.mean(x) for x in train]
    val_L = [np.mean(x) for x in val]

    plt.plot(list(range(1,len(train_L)+1)), train_L,'-o',label='Train')
    plt.plot(list(range(1,len(val_L)+1)), val_L,'-x',label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(save_dir,'train_loss.png'))

def main(args):
    
    model_checkpoint_loc = args.save_dir
                            
    if not os.path.exists(model_checkpoint_loc):
        os.makedirs(model_checkpoint_loc)
        print("The save directory doesn't exist so it is created.")
    use_gpu = args.use_gpu
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate 
    weight_decay =  args.weight_decay                       
    train_feature, val_feature, train_label, val_label, label_emb = load_data()
    train(train_feature, val_feature, train_label, val_label, label_emb, 
          use_cuda=True, hidden_size=hidden_size, 
          learning_rate=learning_rate, weight_decay=weight_decay,           
          save_dir=model_checkpoint_loc)
                            
if __name__ == "__main__":
    time0 = time.time()
    args = args_argument()
    print(args)
    main(args)
    print('Time used', str(time.time()-time0))                            
                            
                           