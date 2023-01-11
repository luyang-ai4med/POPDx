import pandas as pd
import numpy as np
import os
from sklearn.utils.graph import graph_shortest_path
from scipy.sparse.linalg import svds
import json 
import time
from transformers import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from transformers import AutoTokenizer, AutoModel
import torch

def svd_emb(mat, dim=20):
    U, S, V = svds(mat, k=dim)
    X = np.dot(U, np.sqrt(np.diag(S)))
    return X,S

def ontology_emb(dim=500, ICD_network_file = '../data/19.csv', save_dir = './embeddings/', use_pretrain = True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if use_pretrain:
        try:
            with open(os.path.join(save_dir,'embedding_node_to_idx_dict.json'), 'r') as inpt:
                l2i = json.load(inpt)
            with open(os.path.join(save_dir, 'idx_to_embedding_node_dict.json'), 'r') as inpt:
                i2l = json.load(inpt)
            X_emb = np.load(os.path.join(save_dir, 'SVD_embedding_dim'+str(dim)+'.npy'), allow_pickle=False)
            return (l2i, i2l, X_emb)
        except:
            err = 'No pretrained ONTOLOGY embeddings found.'
            print(err)
    
    df_network=pd.read_csv(ICD_network_file, delimiter='\t')

    s2p = {}
    lset = set()
    for i,row in df_network.iterrows():

        s = row.coding
        if row.parent_id==0:
            p='ROOT'
        else:
            p = df_network.loc[df_network['node_id']==row.parent_id, 'coding'].values.item()
        wt = 1.
        if s not in s2p:
            s2p[s] = {}
        s2p[s][p] = wt
        lset.add(s)
        lset.add(p)


    lset = np.sort(list(lset))
    nl = len(lset)
    l2i = dict(zip(lset, range(nl)))
    i2l = dict(zip(range(nl), lset))

    embeddings_to_index_dict = dict(zip(lset,range(len(lset))))
    index_to_embeddings_dict = dict(zip(range(len(lset)),lset))

    with open(os.path.join(save_dir, 'embedding_node_to_idx_dict.json'), 'w') as output:
        json.dump(embeddings_to_index_dict, output)
    with open(os.path.join(save_dir, 'idx_to_embedding_node_dict.json'), 'w') as output:
        json.dump(index_to_embeddings_dict, output)

    print('Num of relationships:',len(s2p))
    print('Num edges:',nl)

    A = np.zeros((nl, nl))
    for s in s2p:
        for p in s2p[s]:
            A[l2i[s], l2i[p]] = s2p[s][p]
            A[l2i[p], l2i[s]] = s2p[s][p]

    time0=time.time()
    sp = graph_shortest_path(A,method='FW',directed=False)
    print(time.time()-time0)
    X_emb, S = svd_emb(sp, dim=dim)
    sp *= -1.

    np.save(os.path.join(save_dir, 'SVD_embedding_dim'+str(dim)+'.npy'), X_emb)   
    return (l2i, i2l, X_emb)

def df_to_batch(df, batch_size=64):
    batch_dict = {}
    batch_s = list(range(len(df)))[::batch_size]
    batch_ = list(zip(batch_s,batch_s[1:]+[len(df)+1]))
    batch_i = 0
    for start_i, end_i in batch_:
        batch_i += 1
        batch_dict[batch_i] = df.iloc[start_i:end_i,:]                  
    return batch_dict

def biobert_embed(df, model, tokenizer, device):
    
    tokenized = df['meaning'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 500
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    
    input_ids = torch.tensor(padded).to(device)  
    attention_mask = torch.tensor(attention_mask).to(device)
    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask)
    return df['coding'].tolist(), hidden_states

def run_bert(batch_size=128, use_pretrain = True):
    if use_pretrain:
        try:
            biboert_embeddings_dict = dict(np.load(os.path.join('./embeddings/biobert_embeddings_dict.npz')))
            features = np.load(os.path.join('./embeddings/biobert_embeddings.npy'))
            return features, biboert_embeddings_dict
        except:
            print('No pretrained BERT embeddings found.')
    if torch.cuda.is_available():  
        device = "cuda:0" 
    else:  
        device = "cpu"  
    print(device)
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert.to(device)
    bert.eval()
    with open('../data/mc_icd10_labels.txt','r') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels] 
    coding19=pd.read_csv('../data/19.csv', delimiter='\t')
    df_labels = pd.merge(pd.DataFrame(labels, columns=['coding']), coding19, on='coding', how='left')
    batch_all = df_to_batch(df_labels, batch_size)
    devices_k = []
    features_all = []
    pooled_all = []
    for batch in batch_all:
        print('Batch #', batch)
        k, (out_hidden_states, pooled_states) = biobert_embed(batch_all[batch], bert, biotokenizer, device)
        last_hidden_states = out_hidden_states.cpu()
        features = last_hidden_states[:, 0, :]
        devices_k.append(k)
        features_all.append(features)
        pooled_all.append(pooled_states.cpu())
    devices_lst = np.concatenate(devices_k, axis=0)
    features = np.concatenate(features_all, axis=0)
    pooled = np.concatenate(pooled_all, axis=0)
    biboert_embeddings_dict = dict(zip(devices_lst,features))
    np.savez(os.path.join('./embeddings/biobert_embeddings_dict.npz'), **biboert_embeddings_dict)
    np.save(os.path.join('./embeddings/biobert_embeddings.npy'), features)
    assert labels == list(biboert_embeddings_dict.keys())
    return features, biboert_embeddings_dict
