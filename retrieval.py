from preprocess_data import Preprocess, Preprocess_query
from word_embedding import word_embedding
import pandas as pd
from calculate_bm25 import BM25Plus, BM25Okapi
import heapq
import itertools
import random
from random import randrange
import time
import math
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
import os.path
from os import path
from gensim.models import Word2Vec, KeyedVectors
from functools import partial

# init global variable
r = Preprocess()
rq = Preprocess_query()
in_idx = []


# class Firefly:
#     # class firefly algoritm
#     # for individual firefly
#     def __init__(self, **kwargs):
#         self.gamma = kwargs.get("gamma", 0.95)
#         self.alpha = kwargs.get("alpha", 1)
#         self.term_expansion = kwargs.get("term_expansion")
#         self.term_size = kwargs.get("term_size")
#         self.population = self.init_population()
#         self.light = None

#     # init population
#     def init_population(self):
#         # initiate list with size of population size
#         return random.sample(self.term_expansion, self.term_size)

#     # update population. addition need to be a list
#     def update_population(self, current_pop, addition):
#         pop = current_pop + addition
#         self.population = pop

#     def fitness(self, rel_doc_idx, top_n_doc, n_query, n_docs):
#         # get evaluation value
#         # choose which evaluation value to use as fitness function.
#         self.light = evaluation(
#             rel_doc_idx, top_n_doc, n_query, n_docs, 'recall')

# update light intensity
# def update_light(self, parameter_list):
#     pass


class Term_Pool:
    def __init__(self, **kwargs):
        self.docs = kwargs.get("docs", None)
    # jaccard coefficient score

    def jaccard_co(self, ti, tj, token_a):
        token_a = pd.Series(token_a)
        base = r'^{}'
        expr = '(?=.*{})'
        exij = base.format(''.join(expr.format(w) for w in [ti, tj]))
        exi = base.format(expr.format(ti))
        exj = base.format(expr.format(tj))
        # print(f"exij: {exij}")
        dij = token_a.str.contains(exij, regex=True).sum()
        di = token_a.str.contains(exi, regex=True).sum()
        dj = token_a.str.contains(exj, regex=True).sum()
        if ((di+dj) - dij) == 0:
            return 0
        jc = dij / ((di + dj) - dij)
        return jc

    # parallelize dataframe operation
    def parallel_df(self, df, func, num_partitions, num_cores):
        df_split = np.array_split(df, num_partitions)
        pool = mp.Pool(mp.cpu_count())
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    # count jaccard using
    def count_jaccard_for_df(self, df):
        df['score'] = df.apply(lambda row: self.jaccard_co(
            row['Ti'], row['Tj'], row['token_a']), axis=1)
        return df

    # similarity score for w2v
    def get_similarity_score(self, model, w1, w2):
        try:
            return model.wv.similarity(w1, w2)
        except KeyError:
            return 0
        pass

    # get similarity score in parallel
    def filter_by_similarity(self, df):
        model = Word2Vec.load('model/wordembed.model')
        # get similarity score
        df['score'] = df.apply(
            lambda row: self.get_similarity_score(model, row['Ti'], row['Tj']), axis=1)
        return df

        # do term scoring with w2v similarity score
    def w2v_filter(self, df, filtered_words):
        before = time.clock()
        tmp_df = self.parallel_df(df, self.filter_by_similarity, 4, 4)
        after = time.clock()
        top_words = tmp_df.groupby(['Tj']).sum().sort_values(
            by=['score'], ascending=False).head(filtered_words)
        print("This is execution time of w2v score with parallelization: {}".format(
            round(after - before, 2)))
        # print(top_words.head())
        sim_w = top_words.index.tolist()
        return sim_w

    def get_rocchio(self, df):
        bm25p = BM25Plus(self.docs)
        df['score'] = df.apply(
            lambda row: bm25p.rocchio_weight(row['words']), axis=1)
        return df

    def rocchio(self, words, n_words):
        """
        get rocchio term weighting score
        """
        docs = r.start(self.docs)
        docs = docs.str.split()

        score = [0]*len(words)
        df = pd.DataFrame(words, columns=['words'])
        df['score'] = score
        scored_df = self.parallel_df(df, self.get_rocchio, 4, 4)
        scored_df = scored_df.sort_values(['score'], ascending=False)
        top_words = scored_df['words'].head(n_words).tolist()
        return top_words

    def prepro(self, top_n_doc, query):
        """
        preprocessing terms in docs. String preprocessing + get terms and delete duplicate
        """
        # preprocessing terms in docs and query
        q_terms = rq.start(query)
        q_terms = list(set(q_terms.split()))
        docs = top_n_doc["Content"].str.lower()
        docs = r.remove_punc(docs)
        docs = r.remove_stopwords(docs)
        docs = r.stemming(docs)
        term_pool = list(itertools.chain.from_iterable(docs.str.split()))
        # built term pool
        term_pool = pd.DataFrame(term_pool, index=None, columns=['pool'])
        term_pool = term_pool.drop_duplicates()
        term_pool = term_pool["pool"].tolist()
        return q_terms, term_pool

    def term_coupling(self, docs, term_pool, q_terms, score_type, pooled_words=None):
        """
        term coupling used for coupling terms in query with terms in docs. Methods using this step is jaccard   cooccurence and word embedding similarity filtering
        """
        if pooled_words is None:
            tmp = list(itertools.product(term_pool, q_terms))
        else:
            tmp = list(itertools.product(pooled_words, q_terms))
        df = pd.DataFrame(tmp, columns=['Tj', 'Ti'])
        df['score'] = ""

        # score type for jaccard if 0 then jaccard
        if score_type == 0:
            df['token_a'] = ""
            df['token_a'] = df.apply(
                lambda row:  docs["Content"].to_list(), axis=1)
        return df

    def term_pool_jaccard(self, df, n_words):
        """
        apply jaccard term scoring method. using parallelization and pandas vectorization for quicker step
        """
        before = time.clock()
        tmp_df = self.parallel_df(df, self.count_jaccard_for_df, 4, 4)
        after = time.clock()
        print("This is execution time of jaccard score with parallelization: {}".format(
            round(after - before, 2)))

        # sort df and drop identical tokens
        top_words = tmp_df.groupby(['Tj']).sum().sort_values(
            by=['score'], ascending=False).head(n_words)
        top_words = top_words.index.tolist()
        return top_words


def init_token_a(docs):
    token_a = docs["Content"]
    return token_a


def PRF_word_embed(all_docs, top_n_doc, query, token_a, n_words, n_docs):
    docs = top_n_doc["Content"].str.lower()
    docs = r.remove_punc(docs)
    docs = r.remove_stopwords(docs)
    docs = r.stemming(docs)
    # if os.path.exists('model/wordembed.model'):

    # dim = len(term_pool)


# move fa based on khennak algorithm
def move_fa(fa1, fa2, attr, rand):
    if rand > attr:
        return fa1
    fa1 = list(set(fa1 + [random.choice(fa2)]))
    return fa1

# count_idf


def count_idf(docs, term):
    N = len(docs)
    Nt = 0
    docs = pd.Series(docs)
    base = r'^{}'
    expr = '(?=.*{})'
    t = base.format(expr.format(term))
    Nt = docs.str.contains(t, regex=True).sum()
    idf = math.log((N/Nt), base=10)
    return idf

# firefly optimization but faster


class Do_FA:
    def __init__(self, all_docs, threshold,  fa_gamma, words_population, term_size, fa_num_iter, q_terms, n_docs, rel_doc_idx, rel_doc_n, fa_alpha, fa_pop_size, rel_doc):
        self.docs = all_docs
        self.gamma = fa_gamma
        self.words_pop = words_population
        self.term_size = term_size
        self.num_iter = fa_num_iter
        self.q_terms = q_terms
        self.n_docs = n_docs
        self.rel_doc_idx = rel_doc_idx
        self.rel_doc_n = rel_doc_n
        self.alpha = fa_alpha
        self.pop_size = fa_pop_size
        self.threshold = threshold
        self.rel_doc = rel_doc
    # initiate fal

    def initiate(self):
        fal_pop = []*self.pop_size
        # initiate fal and score
        for fa in range(self.pop_size):
            fal_pop.append([random.sample(self.words_pop, self.term_size), 0])
        fal = pd.DataFrame(fal_pop, columns=['population', 'score'])
        return fal

    # calculate fitness bootstrap
    def calc_fit(self, df):
        df['score'] = df.apply(
            lambda row: self.calculate_fitness(row['population']), axis=1)
        return df

    # calculate fitness
    def calculate_fitness(self, expand_word):
        tmp_term = self.q_terms + expand_word
        new_q = " ".join(tmp_term)
        # print("this new query: {}".format(new_q))
        ret, x = first_ret(self.docs, new_q, self.n_docs)
        # print(f"Expan words: {expand_word} with recall: {recall}")
        list_idx = process_list_idx(ret)
        rel_doc = get_rel_doc(self.rel_doc, list_idx)
        return evaluation(
            self.rel_doc_idx, list_idx, self.rel_doc_n, self.n_docs, 'map', rel_doc)
    # calculate distance between fa1 and fa2

    def calculate_dis(self, fa1, fa2):
        # comparing fa1 and fa2
        dif = [item for item in fa2 if not item in fa1]
        # print(f"different element: {dif}")
        return len(dif)
    # do
    # move fa based on khennak algorithm

    def move_fa(self, fa1, fa2, attr, rand, alpha):
        # first step
        # if rand <= attr:

        # if rand <= alpha:
        # fa1 = list(set(fa1 + [random.choice(fa2)]))
        return fa1

    # wrapper for comparing
    def compare_fireflies(self, df):
        # compare score by parallel
        # print(f"this is columns of df: {df.columns}")
        df['population'] = df.apply(lambda row: self.do_compare(
            row['population'], row['score'], df.values.tolist(), row['alpha']), axis=1)
        df['score'] = df.apply(
            lambda row: self.calculate_fitness(row['population']), axis=1)
        return df

    def do_compare(self, pop, score, fal, alpha):
        for i in range(len(fal)):
            rand = random.uniform(0, 1)
            if pop == fal[i][0]:
                continue
            cur_att = 1 / (1 + self.calculate_dis(pop, fal[i][0]))
            # move firefly after comparing their light intensity. local search
            if score < fal[i][1]:
                # move fa
                loc_f = []
                for j, fa in enumerate(pop):
                    rand = random.uniform(0, 1)
                    # print(f"rand: {rand} - curr_att: {cur_att}")
                    if fa not in fal[i][0] and rand <= cur_att:
                        # print(
                            # f"local: randomly choose to change {fa} in {pop}")
                        k = random.choice(fal[i][0])
                        loc_f.append(k)
                    else:
                        loc_f.append(fa)
                # print(f"after local movement from {pop} to {loc_f}")

                glob_f = []
                for j, fa in enumerate(loc_f):
                    rand = random.uniform(0, 1)
                    # print(f"rand: {rand} - alpha: {alpha}")
                    if fa not in fal[i][0] and rand <= alpha:
                        # print(
                            # f"global: randomly choose to change {fa} in {loc_f}")
                        k = random.choice(self.words_pop)
                        if k == fa:
                            k = random.choice(self.words_pop)
                        glob_f.append(k)
                    else:
                        glob_f.append(fa)
                # print(f"after global movement from {loc_f} to {glob_f}")

                # loc = [random.choice(fal[i][0]) if ((fa not in fal[i][0]) and (
                #     random.uniform(0, 1) <= cur_att)) else fa for fa in pop]
                # tmp_loc = [random.choice(self.words_pop) if ((fa not in fal[i][0]) and (
                #     random.uniform(0, 1) <= alpha)) else fa for fa in loc]
                # loc = self.move_fa(pop,fal[i][0], cur_att, rand, alpha)
                pop = glob_f
        # recalculate light  intensity
        return pop

    def shamble(self, df):
        df["population"] = df.apply(
            lambda row: self.scrambling(row["population"]), axis=1)
        df["score"] = df.apply(
            lambda row: self.calculate_fitness(row["population"]), axis=1)
        return df

    def scrambling(self, fal):
        print("global checking")
        idx = randrange(len(fal))
        # randomly searching global
        print(f"idx value: {idx}")
        print(f"fal before: {fal}")
        k = random.choice(self.words_pop)
        fal[idx] = k

        return fal

    def main(self):
        # initate iterartion and solution
        df_fa = self.initiate()
        iter = 0
        # curr_sol = df_fa.iloc[len(df_fa)-2]
        # prev_sol = df_fa.iloc[len(df_fa)-1]

        # while iter < max iter and current best solution != previous best solution
        alpha = self.alpha
        # print(df_fa.head(n=self.pop_size))
        ts = Term_Pool()

        # initiate fitness fx
        df_fa = ts.parallel_df(df_fa, self.calc_fit, 4, 4)
        # df_fa['score'] = df_fa.apply(lambda row: self.calculate_fitness(row['population']), axis=1)
        # print(df_fa.head())
        fal = df_fa.sort_values(['score'], ascending=False).values.tolist()
        df_fa = df_fa.sort_values(
            ['score'], ascending=False).reset_index(drop=True)
        curr_sol = fal[len(fal)-2]
        prev_sol = fal[len(fal)-1]

        # iterate and do firefly algorithm
        # print("do")
        print(curr_sol, prev_sol)
        print(df_fa.head(n=20))
        dis = self.calculate_dis(curr_sol[0], prev_sol[0])
        print(f"distance: {dis} from {curr_sol[0]} and {prev_sol[0]}")
        while iter <= self.num_iter and dis > 0:
            print(f"distance: {dis}")
            print("---------------------------------------------------------------------------------------------------")
            print(f"current iter: {iter}")
            print(f"before fa: \n{df_fa.head(n=10)}")
            # parallelizing firefly comparison
            print("paralellizing fa")
            ls = [alpha]*len(fal)
            df_fa['alpha'] = ls
            print()
            # print(df_fa.head())
            fa = ts.parallel_df(df_fa, self.compare_fireflies, 4, 4)

            # print("ini fa: ")
            # print(fa.head())

            fa = fa.sort_values(by=['score'], ascending=False)

            # drop alpha
            print("drop alpha column")
            fa = fa.drop(['alpha'], axis=1)

            print("convert to list")
            fal = fa.values.tolist()
            # if top solution still the same with previous solution, force global search
            prev_sol = curr_sol
            # if fal[0][1] <= self.threshold and random.uniform(0, 1) <= alpha:
            #     df_dp = pd.DataFrame(fal, columns=["population", "score"])
            #     print(df_dp.head())
            #     df_dp = ts.parallel_df(df_dp, self.shamble, 4, 4)
            #     print(df_dp.head())
            #     fal = df_dp.values.tolist()

            iter += 1
            # update alpha
            # get tetha value with 0.95
            alpha = self.alpha * (0.95 ** iter)
            df_fa = pd.DataFrame(fal, columns=["population", "score"]).sort_values(
                ["score"], ascending=False).reset_index(drop=True)
            fal = df_fa.values.tolist()
            curr_sol = fal[0]
            print(f"Current best solution: {curr_sol[0]}")
            print(f"Current best solution fitness value: {curr_sol[1]}")
            print(f"after fa: \n{df_fa.head(n=10)}")
            print(curr_sol)
            dis = self.calculate_dis(curr_sol[0], prev_sol[0])

        fal = df_fa.values.tolist()
        return fal[0]


def PRF(all_docs, threshold, top_n_doc, query, n_words, filtered_words, n_docs, fa_gamma, fa_alpha, fa_num_iter, fa_pop_size, fa_term_size, rel_doc_idx, rel_doc_n, list_idx, rel_doc):
    print("do PRF")

    tp = Term_Pool(docs=all_docs["Content"])
    q_terms, term_pool = tp.prepro(top_n_doc, query)

    # rocchio term scoring
    roc = tp.rocchio(term_pool, filtered_words)

    df = tp.term_coupling(top_n_doc, term_pool, q_terms, 0)
    # term pool construction with jaccard_co
    top_n_words = tp.term_pool_jaccard(df, n_words)

    # jaccard co + w2v
    df1 = tp.term_coupling(top_n_doc, term_pool, q_terms, 0, top_n_words)
    top_n_filtered = tp.w2v_filter(df1, filtered_words)
    print("-------- finished building jaccard term pool ---------")

    # building with w2v + jaccard
    similar_words = tp.w2v_filter(df, n_words)
    df2 = tp.term_coupling(top_n_doc, term_pool, q_terms, 0, similar_words)
    # print(df2.head())
    # get most similar words with top frequent
    freq_similar_words = tp.term_pool_jaccard(df2, filtered_words)

    q_terms = rq.start(query).split()

    print("----- done building term pool ------")
    print("fa + rocchio")

    tmp_r = Do_FA(all_docs, threshold, fa_gamma, roc, fa_term_size, fa_num_iter,
                  q_terms, n_docs, rel_doc_idx, rel_doc_n, fa_alpha, fa_pop_size, rel_doc)
    r_fa = tmp_r.main()
    print(f"this is solution from rocchio: {r_fa}")

    print("fa + jaccard")

    tmp_j = Do_FA(all_docs, threshold, fa_gamma, top_n_words, fa_term_size, fa_num_iter,
                  q_terms, n_docs, rel_doc_idx, rel_doc_n, fa_alpha, fa_pop_size, rel_doc)
    jc_fa = tmp_j.main()
    print(f"this is solution from jaccard: {jc_fa}")

    print("fa + jaccard + w2v")
    tmp_jw = Do_FA(all_docs, threshold, fa_gamma, top_n_filtered, fa_term_size, fa_num_iter,
                   q_terms, n_docs, rel_doc_idx, rel_doc_n, fa_alpha, fa_pop_size, rel_doc)
    jcw_fa = tmp_jw.main()
    print(f"this is solution from jaccard+w2v: {jcw_fa}")

    print("fa + w2v")
    tmp_w = Do_FA(all_docs, threshold, fa_gamma, similar_words, fa_term_size, fa_num_iter,
                  q_terms, n_docs, rel_doc_idx, rel_doc_n, fa_alpha, fa_pop_size, rel_doc)
    w_fa = tmp_w.main()
    print(f"this is solution from w2v{w_fa}")

    print("fa + w2v + jaccard")
    tmp_wj = Do_FA(all_docs, threshold, fa_gamma, freq_similar_words, fa_term_size, fa_num_iter,
                   q_terms, n_docs, rel_doc_idx, rel_doc_n, fa_alpha, fa_pop_size, rel_doc)
    wj_fa = tmp_wj.main()
    print(f"this is solution from w2v + jaccard{wj_fa}")

    return r_fa, jc_fa, jcw_fa, wj_fa, w_fa


def first_ret(docs, query, n):
    new_d = docs["Content"].str.lower()
    new_d = r.remove_punc(new_d)
    new_d = r.remove_stopwords(new_d)
    new_d = r.stemming(new_d)
    # new_d = r.start(docs['Content'])
    # qs = qs.dropna()

    # splitting array
    splitted_docs = new_d.str.split()
    # split query
    # splitted_docs = docs.str.split()
    # print("spliited docs:")
    # print(splitted_docs)
    # initialize bm25+ retrieval model

    bm25p = BM25Plus(splitted_docs)

    # preprocess query
    # tokenized_inq = rq.start(query).split()
    tokenized_inq = query.lower()
    tokenized_inq = rq.remove_punc(tokenized_inq)
    tokenized_inq = rq.remove_stopwords(tokenized_inq)
    tokenized_inq = rq.stemming(tokenized_inq)
    tokenized_inq = tokenized_inq.split()

    # get bm25+ score
    doc_scores = bm25p.get_scores(tokenized_inq)

    # get top n documents
    # n = 100
    top_idx = heapq.nlargest(n, range(len(doc_scores)), doc_scores.__getitem__)
    top_n_doc = docs.iloc[top_idx]

    # top_doc = []
    return top_n_doc, top_idx


def mean_avg_prec(rel_doc_ans, list_idx, n):
    score = 0
    for i in range(len(list_idx)):
        if list_idx[i] in rel_doc_ans:
            cur_rel_doc = get_rel_doc(rel_doc_ans, list_idx[:1])
            score_now = (len(cur_rel_doc)/(i+1)) * 100
            score += score_now
        else:
            continue
    if len(rel_doc_ans) == 0:
        return 0
    score /= len(rel_doc_ans)
    return score


# get number of relevan documents
def get_rel_doc(rel_doc_idx, list_idx):
    ret_rel = []
    for doc_id in rel_doc_idx:
        if doc_id in list_idx:
            ret_rel.append(doc_id)
    return ret_rel


def process_list_idx(top_n_doc):
    list_idx = top_n_doc["Indeks data"].to_list()
    tmp = []
    list_idx = [[int(idx) for idx in lidx.split() if idx.isdigit()]
                for lidx in list_idx]
    list_idx = [i[0] for i in list_idx]
    return list_idx


def evaluation(rel_doc_idx, list_idx, n, n_docs, eval_type, rel_doc_idx_loc=None):
        # get metrics
    # # rel_doc_idx = qr.loc[qr['Query_ID'] == pos]
    # list_idx = top_n_doc["Indeks data"].to_list()
    # tmp = []
    # # print(f"list idx before: {list_idx}")
    # list_idx = [[int(idx) for idx in lidx.split() if idx.isdigit()]
    #             for lidx in list_idx]
    # # print(f"list idx: {list_idx}")
    # list_idx = [i[0] for i in list_idx]
    # print(f"list idx now: {list_idx}")
    num = 0
    score = 0

    if rel_doc_idx_loc is None:
        rel_doc_idx_loc = get_rel_doc(rel_doc_idx, list_idx)

    if eval_type == "precision":
        score = (len(rel_doc_idx_loc)/n_docs)*100
    elif eval_type == 'recall':
        score = (len(rel_doc_idx_loc)/n)*100
    elif eval_type == 'f_score':
        pre = pre_rec(rel_doc_idx_loc, n_docs)
        rec = pre_rec(rel_doc_idx_loc, n)
        if pre == 0 and rec == 0:
            score = 0
        else:
            score = 2*(pre*rec)/(pre+re)
    elif eval_type == 'map' or eval_type == 'mean average precision':
        score = mean_avg_prec(rel_doc_idx_loc, list_idx, len(rel_doc_idx_loc))
    return score
