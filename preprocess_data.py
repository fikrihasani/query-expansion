import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string


class Preprocess():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lowercasing(self, data):
        # print(data.head())
        x = pd.Series(data)
        # print(x.head())
        return x.str.lower()

    def remove_punc(self, data):
        x = pd.Series(data)
        x = data.str.replace('[^\w\s]', '')
        return x

    def remove_stopwords(self, data):
        temp = pd.Series(data).astype('str')
        stop = stopwords.words('english')
        temp1 = temp.str.split()
        temp = temp1.apply(lambda x: ' '.join(
            word for word in x if word not in (stop)))
        return temp

    def stemming(self, data):
        stemmer = PorterStemmer()
        temp = pd.Series(data).astype('str')
        temp = temp.str.split()
        temp = temp.apply(lambda x: ' '.join(stemmer.stem(word)
                                             for word in x))
        return (temp)

    def start(self, data):
        x = self.lowercasing(data)
        x = self.remove_punc(x)
        x = self.remove_stopwords(x)
        x = self.stemming(x)
        return x


class Preprocess_query():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lowercasing(self, query):
        return query.lower()

    def remove_punc(self, query):
        x = query.translate(str.maketrans("", "", string.punctuation))
        return x

    def remove_stopwords(self, query):
        stop = stopwords.words('english')
        temp1 = query.split()
        result = ' '.join(word for word in temp1 if word not in stop)
        return result

    def stemming(self, query):
        stemmer = PorterStemmer()
        temp = query.split()
        result = ' '.join(stemmer.stem(word) for word in temp)
        return result

    def start(self, query):
        x = self.lowercasing(query)
        x = self.remove_punc(x)
        x = self.remove_stopwords(x)
        x = self.stemming(x)
        return x
