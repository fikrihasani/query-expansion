from word_embedding import word_embedding
from preprocess_data import Preprocess, Preprocess_query
import pandas as pd
from gensim.models import Word2Vec
from retrieval import PRF, first_ret, init_token_a, evaluation
import os
import sys
import math
import string


def import_files():
    # import files
    file_q = 'query_corpus.csv'
    file_all = 'alls.csv'
    file_qrel = 'qrels.csv'
    q = pd.read_csv(file_q)
    al = pd.read_csv(file_all)
    qr = pd.read_csv(file_qrel)
    return q, al, qr
    pass


if __name__ == "__main__":

    print(f"punctuation: {string.punctuation}")
    q, al, qr = import_files()

    print("data loaded")

    # copy to new array
    qs = q['Questions']

    a = pd.concat([al["Indeks data"], al['Content']], axis=1)
    print(a.head())
    # sys.exit()
    # print()
    # load word embedding model
    model = ""
    if os.path.exists('model/wordembed.model'):
        model = Word2Vec.load('model/wordembed.model')
    else:
        print("create word embedding model")
        word_embedding(a["Content"])
        model = Word2Vec.load('model/wordembed.model')

    # loop through qr_size
    recall_r = []
    recall_j = []
    recall_w = []
    recall_jw = []
    recall_wj = []
    recall_1st = []

    for pos in range(40):
        cek = qr.loc[qr['Query_ID'] == pos+1]
        if cek.empty:
            # skip
            print("no relevant docs exist")
            continue
        print("-----------------------------------------")
        # pos = 4
        # n -> get number of relevan doc
        n_rel_doc = len(cek)
        n_doc = 30
        docs_of_idx = a.loc[a["Indeks data"] == ".I "+str(pos+1)+" "]

        # get query from list of query
        inq = qs.loc[pos]
        print("query ke : {}\nquery: {}".format(pos+1, inq))

        # get first retrieval top n docs
        top_n_doc, idx = first_ret(a, inq, n_doc)
        # print("\nFirst retrieval: ")
        # print(top_n_doc)

        # doc id
        doc_ids = qr.loc[qr['Query_ID'] == pos+1]

        eval = evaluation(doc_ids, top_n_doc, n_rel_doc, n_doc)
        print("this is value of eval: \nPrecision: {} - Recall: {} - F.Score: {}".format(
            eval[0], eval[1], eval[2]))
        recall_1st.append(round(eval[1], 3))
        # tokenize a
        # token_docs = init_token_a(top_n_doc)
        if eval[1] == 100:
            recall_r.append(100)
            recall_j.append(100)
            recall_jw.append(100)
            recall_wj.append(100)
            recall_w.append(100)
            continue

        # # prf
        n_words = 4
        # random variable, between 0 and 1
        fa_pop_size = 20
        top_n_words = 100
        top_filtered_words = 20
        alpha = 1
        tetha = 0.95
        rec_r, rec_j, rec_jw, rec_wj, rec_w = PRF(a, eval[1], top_n_doc, inq, top_n_words, top_filtered_words, n_doc, tetha, alpha,
                                                  10, fa_pop_size, n_words, doc_ids, n_rel_doc, idx)
        recall_r.append(round(rec_r[1], 3))
        recall_j.append(round(rec_j[1], 3))
        recall_jw.append(round(rec_jw[1], 3))
        recall_wj.append(round(rec_wj[1], 3))
        recall_w.append(round(rec_w[1], 3))

        print(f"this is best solution rocchio: \n{rec_r[0]}")
        print(f"this is best solution jaccard: \n{rec_j[0]}")
        print(f"this is best solution jaccard+w2v: \n{rec_jw[0]}")
        print(f"this is best solution w2v+jaccard: \n{rec_wj[0]}")
        print(f"this is best solution w2v: \n{rec_w[0]}")
        print()
        print(f"recall fa+rocchio: {recall_r}")
        print(f"recall fa+jaccard: {recall_j}")
        print(f"recall fa+jaccard+w2v: {recall_jw}")
        print(f"recall fa+w2v+jaccard: {recall_wj}")
        print(f"recall fa+w2v: {recall_w}")
        print(f"recall bm25: {recall_1st}")

    print(f"average recall rocchio: {sum(recall_r)/len(recall_r)}")
    print(f"average recall jaccard: {sum(recall_j)/len(recall_j)}")
    print(f"average recall jaccard+w2v: {sum(recall_jw)/len(recall_jw)}")
    print(f"average recall w2v+jaccard: {sum(recall_wj)/len(recall_wj)}")
    print(f"average recall w2v: {sum(recall_w)/len(recall_w)}")
    print(f"average recall bm25: {sum(recall_1st)/len(recall_1st)}")
    print("finished")
