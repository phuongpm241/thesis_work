import re
import torch
import numpy as np
from tqdm import tqdm

import collections
from transformers import *
from util.util import *

model = BertModel.from_pretrained('/data/medg/misc/phuongpm/biobert_v1.1_pubmed')
model.eval()

tokenizer = BertTokenizer.from_pretrained('/data/medg/misc/phuongpm/biobert_v1.1_pubmed')

def ent_to_plain_doc(document):
    tokens = document.split()
    for i, t in enumerate(tokens):
        if t.startswith('@entity'):
            tokens[i] = t.replace("@entity","").replace("_", " ")
    return ' '.join(tokens)

def get_embedding(sent, dot=True):
    if dot:
        sent = sent + ' .'
    text = '[CLS] {} [SEP]'.format(sent)
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # not taking the masked tokens into account

    segments_ids = [0]*len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs[0]
        
    return indexed_tokens[1:-1], encoded_layers[0, 1:-1, :]
    

def doc_embedding(doc):
    sents = doc.split(' . ')
    doc_tokens = []
    doc_emb = []
    for i, s in enumerate(sents):
        if i == len(sents)-1:
            tok, emb = get_embedding(s, False)
        else:
            tok, emb = get_embedding(s)
        doc_tokens.extend(tok)
        doc_emb.append(emb)
    
    doc_emb = torch.cat(doc_emb, dim=0)
#     print(doc_emb.shape)
#     print(len(doc_tokens))
    return doc_tokens, doc_emb

def score(candidates, map_cands, doc_tokens, all_probs_start, all_probs_end, average=True):
    """:param map_cands: map of first token of a candidate answer to its position in candidates
       :param candidates: list of list of tokens in candidate answers
       :param doc_tokens: list of tokens in the document
       :param allprobs: tensor of probabilities
       :return: score of each candidate answer normalized over candidates
    """
    scores = [0]*len(candidates)
    counts = [0]*len(candidates)
    for i, t in enumerate(doc_tokens):
        for c in map_cands.get(t, []):
            cand = candidates[c]
            j = i+len(cand)-1
            if j < len(doc_tokens) and t == cand[0] and doc_tokens[j] == cand[-1]:
                scores[c] += (all_probs_start[i]*all_probs_end[j]).item()
#                 print(scores)
                counts[c] += 1
    if average:
        return np.array(scores)/np.array(counts)
    return np.array(scores)

def full_answer(answer, query, doc):
    """:param answer: potential answer
       :param query:
       :param doc: doc with entities marked
    """
    if len(answer.split()) > 1: #multiple entity
        return answer
    
    abbreviation = '( {} )'.format(answer)
    if abbreviation in query: #need to find better answer
        define_ind = doc.index(abbreviation)
        prev_words = doc[:define_ind].split()
        for j in range(len(prev_words)-1, -1, -1):
            if prev_words[j].startswith("@entity"):
                return ent_to_plain(prev_words[j])
            
    return answer

def get_answers(document, query, candidates):
    # get query embedding
#     query = query.replace('▶ ', '').replace('@placeholder', '[MASK]')
#     query_tokens, query_emb = get_embedding(query)
#     mask_ind = query_tokens.index(tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
#     mask_emb = query_emb[mask_ind:mask_ind+1, :]

    query = query.replace('▶ ', '').replace('@placeholder', '[MASK] [MASK]')
    query_tokens, query_emb = get_embedding(query)
    mask_ind_start = query_tokens.index(tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
    mask_emb_start = query_emb[mask_ind_start:mask_ind_start+1, :]
    mask_ind_end = query_tokens.index(tokenizer.convert_tokens_to_ids(['[MASK]'])[0])+1
    mask_emb_end = query_emb[mask_ind_end:mask_ind_end+1, :]
        
    # get document embeddings
    doc_tokens, doc_emb = doc_embedding(ent_to_plain_doc(document))
    
    dot_product_start = torch.mm(mask_emb_start, torch.transpose(doc_emb, 0, 1))
    dot_product_end = torch.mm(mask_emb_end, torch.transpose(doc_emb, 0, 1))
    all_probs_start = torch.nn.functional.softmax(dot_product_start, dim = 1).reshape(-1)
    all_probs_end = torch.nn.functional.softmax(dot_product_end, dim = 1).reshape(-1)
    
    # get the candidate answers position and embeddings
    cand_ans = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(c)) for c in candidates]
    
    maps_cand = collections.defaultdict(list)
    for i, ca in enumerate(cand_ans):
        maps_cand[ca[0]].append(i)
            
#     print(maps_cand.get(2569, []))
    cand_ans_prob = score(cand_ans, maps_cand, doc_tokens, all_probs_start, all_probs_end, True)
    
#     cand_ans_prob = [score(c, doc_tokens, all_probs_start, all_probs_end) for c in cand_ans]


#     print(cand_ans_prob)
    ans_ind = np.argmax(cand_ans_prob)
#     print(ans_ind)
    
    answer = candidates[ans_ind]
    
    return full_answer(answer, query, document)


if __name__ == '__main__':
	dataset = 'test1.0.json'
	filename = "/data/medg/misc/phuongpm/" + dataset

	data = JsonDataset(filename)

	sample_data = list(data.json_to_plain(remove_notfound=True, doc_ent=True))

	towrite = []
	for pt in tqdm(sample_data):
	    document, query, candidates, answer = pt['p'], pt['q'], pt['c'], pt['a']
	    predicted = get_answers(document, query, candidates)
	    towrite.append('{}::{}\n'.format(predicted, answer))

	with open('../results/test_averaged.txt', 'w') as f:
	    f.write(''.join(towrite))
	    f.close()
    
    

