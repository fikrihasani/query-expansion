import pandas as pd

dataq = 'querys.csv'
dataa = 'alls.csv'
dfq = pd.read_csv(dataq).fillna('')
dfa = pd.read_csv(dataa).fillna('')
print(dfa.head())
dfa['combined'] = dfa['Title']+' '+dfa['Author'] + \
    ' '+dfa['Content']+' '+dfa['Bibliography']
dfq['combined'] = dfq['Title']+' '+dfa['Author']+' ' + \
    dfq['Questions']+' '+dfa['Bibliography']

# print(dfqnew.head())
dfa.to_csv('document_corpus.csv', index=False)
dfq.to_csv('query_corpus.csv', index=False)
