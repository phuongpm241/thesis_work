# This script use BERTLM attention head to find a probability distribution over the entire document then use point attention sum to find predict the answer

import torch
import numpy as np
from transformers import *
from util.util import *


model_checkpoint = '/data/medg/misc/phuongpm/biobert_v1.1_pubmed'
tokenizer = BertTokenizer.from_pretrained('/data/medg/misc/phuongpm/biobert_v1.1_pubmed')

modelLM = BertForMaskedLM.from_pretrained('/data/medg/misc/phuongpm/biobert_v1.1_pubmed')
modelLM.eval()

def score(cand, doc_tokens, allprobs):
    """:param cand: list of tokens in a candidate answer
       :param doc_tokens: list of tokens in the document
       :param allprobs: tensor of probabilities
    """
    score = 0
    for i, t in enumerate(doc_tokens):
        j = i+len(cand)-1
        if j < len(doc_tokens) and t == cand[0] and doc_tokens[j] == cand[-1]:
            score += allprobs[i]*allprobs[j]
            
    return score

def pairread(query, subdoc):
    """
    Read the document by subdoc
    """
    text = '[CLS] {} [SEP] {} [SEP]'.format(subdoc, query)
    tokenized_text = tokenizer.tokenize(text)
    mask_index = tokenized_text.index('[MASK]')
    sep_index = tokenized_text.index('[SEP]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    candidates_ids = indexed_tokens[1: sep_index] # get the prediction before softmax for every token in the documents
    segments_ids = [0]*(sep_index+1) + [1]*(len(tokenized_text) - sep_index - 1)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        predictions = modelLM(tokens_tensor, token_type_ids=segments_tensors)[0]
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model

    predictions_cands = predictions[0, mask_index, candidates_ids]
    return candidates_ids, predictions_cands


def get_answers(document, query, candidates, sliding_window = None, max_sequence_length = 512):
    """

    """
    query = query.replace('▶ ', '').replace('@placeholder', '[MASK]')
    sents = document.split(' . ')
    if not sliding_window: #if not sliding window, each sentence is a subdoc
        subdocs = [s + ' . ' for s in sents]

    doc_tokens = []
    atts = []
    
    for sub in subdocs:
        try:
            cand, pred = pairread(query, sub)
            doc_tokens.extend(cand)
            atts.append(pred)
        except:
            print(s) 
            
#     print(atts)
    probs = torch.cat(atts)
    allprobs = torch.nn.functional.softmax(probs, dim = 0)

#     combined = {}
#     for i, t in enumerate(doc_tokens):
#         if t not in combined:
#             combined[t] = 0
#         combined[t] += allprobs[i].item()

    cand_ans = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(c)) for c in candidates]

#     def total_score(tokens):
#         return combined[tokens[0]]*combined[tokens[-1]]

    cand_ans_prob = np.array([score(c, doc_tokens, allprobs) for c in cand_ans])
    cand_ans_prob = cand_ans_prob/np.mean(cand_ans_prob)

    ans_ind = np.argmax(cand_ans_prob)
    
    return candidates[ans_ind]



    # else:
    #     query_tokens = tokenizer.tokenize(query)
    #     max_doc_length = max_sequence_length - query_tokens - 



