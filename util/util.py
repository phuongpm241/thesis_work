"""
@author: SimonSuster
https://github.com/clips/clicr/blob/master/dataset-code/json_to_plain.py
"""

import json

### Constant
DATA_KEY = "data"
VERSION_KEY = "version"
DOC_KEY = "document"
QAS_KEY = "qas"
ANS_KEY = "answers"
TXT_KEY = "text"  # the text part of the answer
ORIG_KEY = "origin"
ID_KEY = "id"
TITLE_KEY = "title"
CONTEXT_KEY = "context"
SOURCE_KEY = "source"
QUERY_KEY = "query"
CUI_KEY = "cui"
SEMTYPE_KEY = "sem_type"
PLACEHOLDER_KEY = "@placeholder"
###

### JSON ###
def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)
    
def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))
    
### DOCUMENT ###
def document_instance(context, title, qas):
    return {"context": context, "title": title, "qas": qas}

def datum_instance(document, source):
        return {"document": document, "source": source}
    
def dataset_instance(version, data):
    return {"version": version, "data": data}

def to_entities(text, ent_marker="@entity"):
    """
    @text (str): text
    @ent_marker (str): marker to denote entity
    Text includes entities marked as BEG__w1 w2 w3__END. Transform to a single entity @entityw1_w2_w3.
    """
    word_list = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = [w_stripped.split("_")[2]]
            word_list.append(ent_marker + "_".join(concept))
            if inside:  # something went wrong, leave as is
                print("Inconsistent markup.")
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            if not inside:
                word_list.append(w_stripped[:-5])
            else:
                concept.append(w_stripped.rsplit("_", 2)[0])
                word_list.append(ent_marker + "_".join(concept))
                inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                word_list.append(w_stripped)

    return " ".join(word_list)

def ent_to_plain(e):
    """
    :param e: "@entityLeft_hand"
    :return: "Left hand"
    """
    # return " ".join(e[len("@entity"):].split("_"))
    return e.replace("@entity","").replace("_", " ")

def plain_to_ent(e):
    """
    :param e: "Left hand"
    :return: "@entityLeft_hand"
    """
    return "@entity" + "_".join(e.split())


def write_gareader(i, f_out):
    """
    :param i: {"id": "",
                  "p": "",
                  "q", "",
                  "a", "",
                  "c", [""]}
    """
    with open(f_out, "w") as fh_out:
        fh_out.write(i["id"] + "\n\n")
        fh_out.write(i["p"] + "\n\n")
        fh_out.write(i["q"] + "\n\n")
        fh_out.write(i["a"] + "\n\n")
        fh_out.write("\n".join(i["c"]) + "\n")
        
    
def remove_entity_marks(txt):
    return txt.replace("BEG__", "").replace("__END", "")
        
class JsonDataset:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.dataset = load_json(self.dataset_file)

    def json_to_plain(self, remove_notfound=False, stp="no-ent", doc_ent=False, include_q_cands=False, cand_a=False):
        """
        :param remove_notfound: replace the truth answer by an equivalent answer (UMLS) found in the document 
        :param include_q_cands: whether to include entities in query in list of candidate answers
        :param stp: no-ent | ent; whether to mark entities in passage; if ent, a multiword entity is treated as 1 token
        :param cand_a: whether to ensure truth answer must be contained in the list of candidate answers
        :return: {"id": "",
                  "p": "",
                  "q", "",
                  "a", "",
                  "c", [""]}
        """
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                fields = {}
                qa_txt_option = (" " + qa[QUERY_KEY]) if include_q_cands else ""
                #cand = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + " " +
                #                               datum[DOC_KEY][CONTEXT_KEY] + qa_txt_option).lower().split() if w.startswith('@entity')]
                cand = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + " " +
                                               datum[DOC_KEY][CONTEXT_KEY]).lower().split() if w.startswith('@entity')]
                cand_q = [w for w in to_entities(qa_txt_option).lower().split() if w.startswith('@entity')]
                if stp == "no-ent":
                    c = {ent_to_plain(e) for e in set(cand)}
                    a = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            a = ans[TXT_KEY].lower()
                    if remove_notfound:
                        if a not in c:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ans[TXT_KEY].lower()
                                    if umls_answer in c:
                                        found_umls = True
                                        a = umls_answer
                            if not found_umls:
                                continue
                    fields["c"] = list(c)
                    assert a
                    fields["a"] = a
                    if cand_a and a not in c:
                        fields["c"].append(a)

                    if doc_ent:
                        document = to_entities(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace("\n", " ").lower()
                    else:
                        document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace("\n", " ").lower()
                        
                    fields["p"] = document
                    fields["q"] = remove_entity_marks(qa[QUERY_KEY]).replace("\n", " ").lower()

                elif stp == "ent":
                    c = set(cand)
                    a = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            a = plain_to_ent(ans[TXT_KEY].lower())
                    if remove_notfound:
                        if a not in c:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = plain_to_ent(ans[TXT_KEY].lower())
                                    if umls_answer in c:
                                        found_umls = True
                                        a = umls_answer
                            if not found_umls:
                                continue
                    fields["c"] = list(c)
                    assert a
                    fields["a"] = a
                    if cand_a and a not in c:
                        fields["c"].append(a)
                    document = to_entities(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                        "\n", " ").lower()
                    fields["p"] = document
                    fields["q"] = to_entities(qa[QUERY_KEY]).replace("\n", " ").lower()
                else:
                    raise NotImplementedError

                fields["id"] = qa[ID_KEY]

                yield fields


def map_to_split_name(f_dataset):
    """
    :param f_dataset: any of "dev1.0.json", "train1.0.json", "test1.0.json"
    :return: any of "training", "validation", "test"
    """
    if f_dataset[:-len("1.0.json")] == "train":
        name = "training"
    elif f_dataset[:-len("1.0.json")] == "test":
        name = "test"
    elif f_dataset[:-len("1.0.json")] == "dev":
        name = "validation"
    else:
        raise ValueError

    return name

def sample_dataset(f, f_out,  n=50):
    """
    Reduce the dataset to include only the first n instances from n different case reports.
    """
    dataset = load_json(f)
    new_data = []

    for c, datum in enumerate(dataset[DATA_KEY]):
        if c == n:
            break
        qas = [datum[DOC_KEY][QAS_KEY][0]]
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    save_json(dataset_instance(dataset[VERSION_KEY], new_data), f_out)

