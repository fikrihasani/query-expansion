from gensim.models import Word2Vec
import pandas as pd
from preprocess_data import Preprocess, Preprocess_query


def word_embedding(docs):
    prep = Preprocess()

    prep_doc = prep.start(docs)
    # prep_doc = docs
    print(prep_doc.head())
    corpus_list = prep_doc.str.split().to_list()
    # corpus_list = ["present studi histori dewey decim classif first edit ddc publish 1876 eighteenth edit 1971 futur edit continu appear need spite ddc long healthi life howev full stori never told biographi dewey briefli describ system first attempt provid detail histori work spur growth librarianship countri abroad", "report analysi 6300 act use 104 technic librari unit kingdom librari use one aspect wider pattern inform use inform transfer librari restrict use document take account document use outsid librari still less inform transfer oral person person librari act channel proport situat inform transfer take technic inform transfer whole doubt proport major one user technic inform particularli technolog rather scienc visit librari rare reli desk collect handbook current period person contact colleagu peopl organ even regular librari user also receiv inform way", "relationship organ control write organ control knowledg inform inevit enter stori write contain along much els great deal mankind stock knowledg inform bibliograph control form power knowledg form power familiar slogan claim bibliograph control certain sens power power power obtain knowledg record written form write simpli simpl way storehous knowledg cannot satisfactorili discuss bibliograph control simpli control knowledg inform contain write",
    #                "establish nine new univers 1960 provok highli stimul reexamin natur purpos manag academ librari longestablish attitud method question although chang made basic difficulti remain lack object inform best way provid librari servic univers report ugc committe librari parri repot 267 gener endors chang also stress need research aspect academ librari provis", "although use game profession educ becom widespread last decad method use number field mani hundr year origin trace simpl war game use militari train real thing either unavail danger recent time game becom sophist mani use larg electron comput handl complex calcul involv sinc 1956 first welldevelop manag game introduc techniqu spread rapidli wide varieti disciplin today use level educ primari school class cours experienc profession men women one main caus game explos rapid develop sophist manag techniqu simul mathemat model made possibl rapid advanc comput technolog"]
    # print(corpus_list[:5])

    model = Word2Vec(corpus_list, size=100,
                     window=5, min_count=2, workers=4)
    # model = Word2Vec(corpus_list,)
    model.wv.save('model/corpusvectors.wv')
    model.save('model/wordembed.model')

    print("model saved")

# model loaded
