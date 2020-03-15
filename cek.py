import pandas as pd
import itertools


def cek(tmp, score):
    ret = 0
    ret += score
    for i in tmp:
        if i % 2 == 0:
            ret += 2
    return ret


def move(i_pop, i_score, j_pop, j_score):
    if j_score > i_score:
        return i_pop + j_pop
    return i_pop


a = [i for i in range(30)]
z = [0]*len(a)

match = [[[a[i]], z[i]] for i in range(len(a))]
df = pd.DataFrame(match, columns=['cek', 'num'])
# print(df)
df['num'] = df.apply(lambda row: cek(row['cek'], row['num']), axis=1)
x = df.values.tolist()
new_x = [[x[i][0], x[i][1], x[j][0], x[j][1]]
         for i in range(len(x)) for j in range(len(x)) if x[i][0] != x[j][0]]
# print(df)
# print(x)
new_x = pd.DataFrame(new_x, columns=['i', 'i_score', 'j', 'j_score'])
new_x['i'] = new_x.apply(lambda row: move(
    row['i'], row['i_score'], row['j'], row['j_score']), axis=1)
# print(new_x)
new_x['i_score'] = new_x.apply(
    lambda row: cek(row['i'], row['i_score']), axis=1)
print(new_x)
