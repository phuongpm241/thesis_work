import sys
sys.path.append('/scratch/phuongpm/bertviz')
from transformers import AlbertTokenizer, AlbertModel 
import pickle
import torch
from transformers.data.processors.squad import SquadResult, SquadV2Processor
from transformers import (
    squad_convert_examples_to_features
)
import numpy as np
from tqdm import tqdm

np.random.seed(0)

model_version = '/crimea/phuongpm/tuned/squadv2_albert_what'
do_lower_case = True
model = AlbertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = AlbertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

processor = SquadV2Processor()

print("Generating example ...")
examples = processor.get_dev_examples('/data/medg/misc/phuongpm/clicr_squad_v2_mask/', filename = 'dev1.0.json')

random_exps = np.random.choice(examples, 100, replace=False)

print("Generating features ...")
features, dataset = squad_convert_examples_to_features(
    examples=random_exps,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset='pt',
    threads=1
)

alldata = []
print("Generating attention map ... ")
for f in tqdm(features):
    input_ids = torch.tensor([f.input_ids])
    token_type_ids = torch.tensor([f.token_type_ids])
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    #generate a [n_layer, n_head, n_tokens, n_tokens] attention map
    attns = np.array([attention[l][0, :, :, :].tolist() for l in range(12)])
    alldata.append({"tokens": tokens, "attns": attns})

print("Saving attention map ... ")
pickle.dump(alldata, open( "/crimea/phuongpm/att_study/attn_map_albert.p", "wb" ) )

print("Successful!")
