from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import time
import numpy as np
import random
import pickle
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import ModelSaving
from models import POPDxModel
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve


def args_argument():
    parser = argparse.ArgumentParser(description='''
    The script to test POPDx. 
    Please specify the path to the test datasets in the python script.
    ''')
    parser.add_argument('-m', '--model_path', required=True, help='The path to POPDx model e.g. "./save/POPDx_train/best_classifier.pth.tar"')
    parser.add_argument('-o', '--output_path', required=True, help='The output directory  e.g. "./save/POPDx_train/test/"')
    parser.add_argument('-s', '--hidden_size', type=int, default=150, help='Default hidden size is 150. Consistent with training.')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='Default batch size is 512.')
    parser.add_argument('--use_gpu', default=False,  help='Default setup is to not use GPU for test.')
    args = parser.parse_args()
    return args    

def load_data():
    data_folder = '/oak/stanford/groups/rbaltman/luyang/scripts/rareseeker/PheCode/data/'
    train_label_file = 'train_phecode_labels.npy'
    test_feature_file = 'test_feature.npy'
    test_label_file = 'test_phecode_labels.npy'
    label_emb_file = 'phecode_label_embed.npy'
    train_label = np.load(os.path.join(data_folder, train_label_file)) 
    test_feature = np.load(os.path.join(data_folder, test_feature_file), allow_pickle=True)
    test_label = np.load(os.path.join(data_folder, test_label_file)) 
    label_emb = np.load(os.path.join(data_folder, label_emb_file), allow_pickle=True)
    return train_label, test_feature, test_label, test_label, label_emb

def test(train_label, test_feature, test_label, label_emb, model_checkpoint_loc=None, output_dir=None,
         use_cuda=False, hidden_size=150, batch_size=512):
    
    patient_counts_perlabels_train = train_label.sum(axis=0)
    patient_counts_perlabels_test = test_label.sum(axis=0)
    
    testdata = Dataset(test_feature, test_label)
    test_loader = DataLoader(dataset = testdata, batch_size = 512, shuffle = False)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    label_emb = torch.tensor(label_emb, dtype=torch.float, device=device)

    net = POPDxModel(test_feature.shape[1], test_label.shape[1], hidden_size, label_emb) 
    if use_cuda:
        net.cuda()
    checkpoint = torch.load(model_checkpoint_loc, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Done loading model.')
   
    test_outputs = []
    test_truth = []
    for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        test_output = net(inputs)
        test_outputs.append(test_output.detach().cpu())
        test_truth.append(labels.detach().cpu())

    test_outputs = torch.cat(test_outputs, dim=0)
    test_truth = torch.cat(test_truth, dim=0)
    predicted_test = (torch.sigmoid(test_outputs).data > 0.5).type(torch.float) 
    probs_all = torch.sigmoid(test_outputs).data 
    print('Predicted test.')
    
    save_folder = os.path.join(output_dir, 'test_sampling')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    print('Generating the test statistics!')
    sampling(test_outputs, test_truth, test_label, save_folder, neg_to_pos_ratio=10)
    get_metrics(test_label, save_folder, output_dir)
    

def sampling(test_outputs, test_truth, test_label, save_folder=None, neg_to_pos_ratio=10):
    print('Setting the negative to positive ratio.')
    for i in tqdm(range(test_label.shape[1])):
        results = {}
        y_test_subset = test_label[:,i]
        idx_icd10 = [i for i,j in enumerate(y_test_subset) if j==1]
        idex_neg = [i for i,j in enumerate(y_test_subset) if j!=1]
        if len(idx_icd10)>0:
            test_outputs_i = []
            test_truth_i = []

            if len(idex_neg)/len(idx_icd10)>=10:
                if len(idx_icd10)<1000:
                    #print(i,'few minority', len(idx_icd10))
                    for j in range(50):
                        selected_pos = random.choices(idx_icd10, k=16)
                        selected_neg = random.sample(idex_neg, 16*neg_to_pos_ratio)
                        selected_ = selected_pos + selected_neg
                        test_outputs_i.append(test_outputs[selected_])
                        test_truth_i.append(test_truth[selected_])
                else:
                    #print(i,'few majority', len(idx_icd10))
                    for j in range(50):
                        selected_pos = random.sample(idx_icd10, 16)
                        selected_neg = random.sample(idex_neg, 16*neg_to_pos_ratio)
                        selected_ = selected_pos + selected_neg
                        test_outputs_i.append(test_outputs[selected_])
                        test_truth_i.append(test_truth[selected_])
            else:
                #print(i,'majority majority')
                for j in range(1500):
                    selected_pos = random.sample(idx_icd10, 16)
                    selected_neg = random.sample(idex_neg, 160)
                    selected_ = selected_pos + selected_neg
                    test_outputs_i.append(test_outputs[selected_])
                    test_truth_i.append(test_truth[selected_])


            test_outputs_ = torch.cat(test_outputs_i, dim=0)
            test_truth_ = torch.cat(test_truth_i, dim=0)
            predicted_test_ = (torch.sigmoid(test_outputs_).data > 0.5).type(torch.float) 
            probs_all_ = torch.sigmoid(test_outputs_).data 

            results[i]=[test_outputs_[:,i], test_truth_, predicted_test_[:,i], probs_all_[:,i]]
            
            try:
                np.save(os.path.join(save_folder, 'rareseeker_negativesampling'+str(i)+'.npy'),results)
                #print('npy saved')
            except Exception as e: 
                print(e)
                pickle.dump(results, open(os.path.join(save_folder, 'rareseeker_negativesampling'+str(i)+'.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        else:
            continue

            
def get_metrics(test_label, save_folder, output_dir):
    print('AURPC and AUROC')
    precision_ = {}
    recall_ = {}
    AUPRC = {}
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(test_label.shape[1]):
        npy_fname = os.path.join(save_folder, 'rareseeker_negativesampling'+str(i)+'.npy')
        pk_fname = os.path.join(save_folder, 'rareseeker_negativesampling'+str(i)+'.pickle')
        if os.path.isfile(pk_fname):
            results = pickle.load(open(pk_fname, 'rb'))
        elif os.path.isfile(npy_fname):
            results = np.load(npy_fname, allow_pickle=True)
            results = results[()]
        else:
            continue
        #print(i)   
        precision_[i], recall_[i], _ = precision_recall_curve(results[i][1][:,i].numpy(), results[i][3].numpy())
        AUPRC[i] = auc(recall_[i], precision_[i])
        fpr[i], tpr[i], _ = roc_curve(results[i][1][:,i].numpy(), results[i][3].numpy())
        roc_auc[i] = roc_auc_score(results[i][1][:,i].numpy(), results[i][3].numpy(), average='weighted') 
    
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_fpr.npy'),fpr) 
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_tpr.npy'),tpr)
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_AUROC.npy'),roc_auc) 
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_precision.npy'), precision_) 
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_recall.npy'),recall_)
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_AUPRC.npy'),AUPRC) 
    
def main(args):
    model_checkpoint_loc = args.model_path
    output_dir = args.output_path
    use_gpu = args.use_gpu
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    model_name = model_checkpoint_loc.split('/')[-1]
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print("The test directory doesn't exist so it is created.")   
    train_label, test_feature, test_label, test_label, label_emb = load_data()
    print('Done loading data.')
    test(train_label, test_feature, test_label, label_emb, 
         model_checkpoint_loc, output_dir,
         use_gpu, hidden_size, batch_size)
    

if __name__ == "__main__":
    time0 = time.time()
    args = args_argument()
    print(args)
    main(args)
    print('Time used', str(time.time()-time0))