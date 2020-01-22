#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:52:14 2017

@author: xuweijia
"""
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import trange

def to_var(np_input,use_cuda,evaluate=False):
    # if evaluate, volatile=True, no grad be computed
    if use_cuda:
        output=Variable(torch.from_numpy(np_input),volatile=evaluate).cuda()
    else:
        output=Variable(torch.from_numpy(np_input),volatile=evaluate)
    return output

def to_vars(np_inputs,use_cuda,evaluate=False):
    return [to_var(np_input,use_cuda,evaluate) for np_input in np_inputs]

def gru(rnn_model,batch_seq,batch_seq_mask):
# input:       B,T,D
# input_mask   B,T            (real length 1,1,1,0,0)
  sequence_length=torch.sum(batch_seq_mask,1,keepdim=True).squeeze(-1)         # B,
#  print(sequence_length.type())
#  print(sequence_length.size())
  sort_len,sort_index=sequence_length.sort(dim=0,descending=True) # B,
#  print(sort_index.type())
#  print(sort_index.size())
# sorted input:B,T,D
  sorted_batchseq=batch_seq[sort_index.data]
# pack input[0]:seq_len,D    every unvoid word embeddding
# pack_input[1]:T,           every time stamp word number

  # print(sort_len)
  cpu_sort_len = torch.as_tensor(sort_len.data, dtype=torch.int64, device='cpu')
  # print(cpu_sort_len)
  # pack_seq=torch.nn.utils.rnn.pack_padded_sequence(sorted_batchseq,sort_len.data.cpu().numpy(),batch_first=True)
  pack_seq=torch.nn.utils.rnn.pack_padded_sequence(sorted_batchseq,cpu_sort_len,batch_first=True)

  output_pack,hn=rnn_model(pack_seq)
  # unpack
  # output:   B,T,D
  output,out_seq_len=torch.nn.utils.rnn.pad_packed_sequence(output_pack,batch_first=True)
  # ori_order:B,T,D
  _,original_index=sort_index.sort(dim=0,descending=False)
  original_output=output[original_index.data]
  return original_output,sequence_length,hn # 2,B,h

def att_sum(t1,t2):
    # B,T,2h
    return t1 + t2

def att_mul(t1,t2):
    # B,T,2h
    return torch.mul(t1 , t2)

def att_cat(t1,t2):
    # B,T,2h
    return torch.cat([t1,t2],dim=-1) # B,T,4h

def feat_fuc(dw,qw):
    # dw:B,T
    # qw:B,Q
    feat=np.zeros(dw.shape)
    bsize=dw.shape[0]
    #print("feat batch %d, T:%d"%(bsize,dw.shape[1]))
#    # every batch's feature
#    #feat: B,T
#    if bsize==1:
#        feat=np.in1d(dw,qw)# (T,)
#    else:
    for i in range(bsize):
        feat[i,:]=np.in1d(dw[i,:],qw[i,:]) #(B,T)
    return feat.astype('int32')

def evaluate(model, data, use_cuda):
    acc = loss = n_examples = 0
    for dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos, fnames in data:
        
        bsize = dw.shape[0]
        feat=feat_fuc(dw,qw)
        
        dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat=to_vars(\
        [dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat],use_cuda,evaluate=True)
        
        loss_batch,acc_batch=model(dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat)
        
        loss+=loss_batch.item() #*bsize
        acc+=acc_batch.item()
        n_examples += bsize
    # finish all ex in valid
    return loss/n_examples,acc/n_examples
