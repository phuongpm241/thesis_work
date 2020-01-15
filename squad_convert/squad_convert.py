## Convert CLICR to SQUADv2.0 format
import sys
sys.path.append('../util')
from util import *
from tqdm import tqdm
import numpy as np

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)

filepath = "/data/medg/misc/phuongpm/"

def closestanswer(ans, cands):
    small_dist = [None, None]
    for c in cands:
        dist = w2vmodel.wmdistance(ans, c)
        if not small_dist[0] or dist < small_dist[0]:
            small_dist = [dist, c]
    return small_dist

def convert(inputname, savename):

	sampledata = load_json(filepath + inputname)

	data = []

	for datum in tqdm(sampledata[DATA_KEY]):
	    title = remove_entity_marks(datum[DOC_KEY][TITLE_KEY]).replace("\n", " ").lower()
	    context = remove_entity_marks(datum[DOC_KEY][CONTEXT_KEY]).replace("\n", " ").lower()
	    candidates = [w for w in to_entities(datum[DOC_KEY][CONTEXT_KEY]).lower().split() if w.startswith('@entity')]
	    cands = {ent_to_plain(e) for e in set(candidates)}
	    qas=[]
	    for qa in datum[DOC_KEY][QAS_KEY]:
	        a = ""
	        answers = []
	        for ans in qa[ANS_KEY]:
	            if ans[ORIG_KEY] == "dataset":
	                a = ans[TXT_KEY].lower()
	            
	#         print(' ' + a in context)
	        start = 0
	        while start < len(context) and start > -1:
	            try:
	                answer_start = context.index(' '+a, start+1)
	            except:
	                break
	            answers.append({'answer_start': answer_start, 'text': a})
	            start = answer_start
	        qas.append({
	            'answers':answers,
	            'question':remove_entity_marks(qa[QUERY_KEY]).replace("\n", " ").lower().replace('@placeholder', 'what').replace(
	    '.', '?').replace("▶ ",""),
	            'id':qa[ID_KEY],
	            'is_impossible':not bool(answers)
	        })
	    if len(qas) > 0:
	        data.append({
	            'title':title,
	            'paragraphs':[{'context': context, 'qas':qas}]
	        })

	save = {'data':data, 'version':sampledata[VERSION_KEY]}
	save_json(save, filepath+savename)
	return 'successfully saved at {}'.format(filepath+savename)

if __name__ == '__main__':
	inputname = 'test1.0.json'
	savename = 'clicr_test_squadstyle.1.0.json'
	convert(inputname, savename)






