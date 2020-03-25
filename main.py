from word_embedding import word_embedding
from preprocess_data import Preprocess, Preprocess_query
import pandas as pd
from gensim.models import Word2Vec
from retrieval import PRF, first_ret, init_token_a, evaluation, process_list_idx, get_rel_doc
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
    score_r = []
    score_j = []
    score_w = []
    score_jw = []
    score_wj = []
    score_1st = []

    for pos in range(0, 20):
        cek = qr.loc[qr['Query_ID'] == pos+1]
        if cek.empty:
            # skip
            print("no relevant docs exist")
            continue
        print("-----------------------------------------")
        # pos = 4
        # n -> get number of relevan doc
        n_rel_doc = len(cek)
        print(f"jumlah dokumen relevan: {n_rel_doc}")
        n_doc = 50
        docs_of_idx = a.loc[a["Indeks data"] == ".I "+str(pos+1)+" "]

        # get query from list of query
        inq = qs.loc[pos]
        print("query ke : {}\nquery: {}".format(pos+1, inq))

        # get first retrieval top n docs
        top_n_doc, idx = first_ret(a, inq, n_doc)

        # doc id
        doc_ids = qr.loc[qr['Query_ID'] == pos+1]

        list_idx = process_list_idx(top_n_doc)
        rel_doc = get_rel_doc(doc_ids["Answer_ID"], list_idx)

        eval = evaluation(doc_ids, list_idx, n_rel_doc, n_doc, 'map', rel_doc)
        rec = evaluation(doc_ids, list_idx, n_rel_doc,
                         n_doc, 'recall', rel_doc)
        prec = evaluation(doc_ids, list_idx, n_rel_doc,
                          n_doc, 'precision', rel_doc)
        print(
            "this is score of map: {} - recall: {} - precision: {}".format(eval, rec, prec))
        score_1st.append(round(eval, 3))
        # tokenize a
        # token_docs = init_token_a(top_n_doc)
        if eval == 100:
            score_r.append(100)
            score_j.append(100)
            score_jw.append(100)
            score_wj.append(100)
            score_w.append(100)
            continue
        # # prf
        n_words = 2
        # random variable, between 0 and 1
        fa_pop_size = 20
        top_n_words = 50
        top_filtered_words = 20
        alpha = 1
        tetha = 0.95
        rec_r, rec_j, rec_jw, rec_wj, rec_w = PRF(a, eval, top_n_doc, inq, top_n_words, top_filtered_words, n_doc, tetha, alpha,
                                                  10, fa_pop_size, n_words, doc_ids, n_rel_doc, list_idx, rel_doc)
        score_r.append(round(rec_r[1], 3))
        score_j.append(round(rec_j[1], 3))
        score_jw.append(round(rec_jw[1], 3))
        score_wj.append(round(rec_wj[1], 3))
        score_w.append(round(rec_w[1], 3))

        print(f"this is best solution rocchio: \n{rec_r[0]}")
        print(f"this is best solution jaccard: \n{rec_j[0]}")
        print(f"this is best solution jaccard+w2v: \n{rec_jw[0]}")
        print(f"this is best solution w2v+jaccard: \n{rec_wj[0]}")
        print(f"this is best solution w2v: \n{rec_w[0]}")
        print()
        print(f"recall fa+rocchio: {score_r}")
        print(f"recall fa+jaccard: {score_j}")
        print(f"recall fa+jaccard+w2v: {score_jw}")
        print(f"recall fa+w2v+jaccard: {score_wj}")
        print(f"recall fa+w2v: {score_w}")
        print(f"recall bm25: {score_1st}")

    # sys.exit()
    print(f"average recall rocchio: {sum(score_r)/len(score_r)}")
    print(f"average recall jaccard: {sum(score_j)/len(score_j)}")
    print(f"average recall jaccard+w2v: {sum(score_jw)/len(score_jw)}")
    print(f"average recall w2v+jaccard: {sum(score_wj)/len(score_wj)}")
    print(f"average recall w2v: {sum(score_w)/len(score_w)}")
    print(f"average recall bm25: {sum(score_1st)/len(score_1st)}")
    print("finished")
