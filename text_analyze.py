import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import preprocessor as p
import numpy as np

import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from scipy import sparse
from sklearn import svm
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
stp_words = set(stopwords.words("english"))
import string

def FindUrl(s):
    # findall() has been used
    # with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', s)
    return url

def FindAllTextTypes(df):
    urls = []
    hashTags = []
    other = []
    for wrd in df['Tweet']:
        s = str(wrd).split(" ")
        ulist = []
        hlist = []
        olist = []
        for token in s:
            a = FindUrl(token)
            if token.startswith("#"):
                hlist.append(token)
            elif not a:
                olist.append(token)
            else:
                ulist.extend(a)
        urls.append(' '.join(FindnetLocations(ulist)))
        hashTags.append(' '.join(hlist))
        other.append(' '.join(olist))
    df['Urls'] = urls
    df['HTags'] = hashTags
    df['Other'] = other

def FindnetLocations(urls):
    sites = []
    for url in urls:
        try:
            parsed_uri = urlparse(url)
            sites.append('{uri.netloc}'.format(uri=parsed_uri))
        except ValueError:
            # one site had an error like this
            parsed_uri = urlparse(url.replace("]", ""))
            sites.append('{uri.netloc}'.format(uri=parsed_uri))
    return sites

def FindUpperCasedWords(df):
    upper = []
    for txt in df['Other']:
        s = str(txt).split(" ")
        u = []
        for tag in s:
            if (len(tag) > 0 and tag[0].isupper()):
                u.append(tag)
        upper.append(' '.join(u))
    df['Upper'] = upper

def FindLowerCasedWords(df):
    lower = []
    for txt in df['Other']:
        s = str(txt).split(" ")
        u = []
        for tag in s:
            if (len(tag) > 0 and tag[0].islower()):
                u.append(tag)
        lower.append(' '.join(u))
    df['Lower'] = lower

def seperateText(df):
    FindAllTextTypes(df)
    FindUpperCasedWords(df)
    FindLowerCasedWords(df)
    return df

def clean_tweets(tweets, h_tags, label):
    cleaned_tweets = []
    cleaned_labels = []
    cleaned_tags = []
    cleaned_domain = []
    for t,h,l in zip(tweets,h_tags, label):
        if True: #t.find('rt') == -1:

            url = FindUrl(t)
            if url != '':
                print(' '.join(url))
                t.replace(' '.join(url),'')
                cleaned_domain.append(' '.join(FindnetLocations(url)))
            else:
                cleaned_domain.append('')
            lowerCase = t.lower()
            lowerCase = p.clean(lowerCase)
            lowerCase = lowerCase.translate(str.maketrans('', '', string.punctuation))
            cleaned_tweets.append(lowerCase)

            cleaned_labels.append(l)
            cleaned_tags.append(h)


    return cleaned_tweets, cleaned_tags, cleaned_labels, cleaned_domain

def clean_test_tweets(tweets, h_tags):
    cleaned_tweets = []
    cleaned_tags = []
    for t,h in zip(tweets,h_tags):
        if True: #t.find('rt') == -1:
            lowerCase = t.lower()
            lowerCase = lowerCase.translate(str.maketrans('', '', string.punctuation))
            cleaned_tweets.append(p.clean(lowerCase))
            cleaned_tags.append(h)


    return cleaned_tweets, cleaned_tags

def read_data(filename, isLabeled=True):
    f = open(filename, 'r', encoding="utf8")
    text = []
    labels = []

    for line in f:
        splits = line.strip().split('\t')

        if isLabeled:
            labels.append(splits[0])
        text.append(splits[-1].lower())
    return text, labels
def save_data(filename,pd):
    pd.to_csv(path_or_buf=filename, header=['Val'],index=False, sep=',')

def Tfidf(doc):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=5, max_df=0.7, stop_words=stp_words, lowercase=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(doc)

    '''first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    # print(df)

    print(tfidf_vectorizer_vectors.shape)'''
    return tfidf_vectorizer_vectors, tfidf_vectorizer
def Tfidf_multistep(doc):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(doc)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'])

    print(df_idf)
    tf_idf_vector = tfidf_transformer.transform(word_count_vector)
    df = pd.DataFrame(tf_idf_vector[0].T.todense(), index=cv.get_feature_names(),
                      columns=["tfidf"])
    print(df)
    return tf_idf_vector



#X_train, y_train = read_data('train_tweets.txt')
#x_test, _ = read_data('test_tweets_unlabeled.txt')
df_train = pd.read_csv('train_tweets.txt', names=['User','Tweet'],delimiter = '\t')
df_test = pd.read_csv('test_tweets_unlabeled.txt', names=['Tweet'],delimiter = '\n\t:',engine='python')


df_train = df_train.sample(5000, random_state=41)
print(df_train)
sep_txt = seperateText(df_train)
hashtags = sep_txt['HTags']
sep_txt = seperateText(df_test)
hashtags_test = sep_txt['HTags']


X_train,  X_test, h_train, h_test, y_train, y_test= train_test_split(df_train['Tweet'].values ,  hashtags.values, df_train['User'].values , test_size=0.1)

X_train, h_train, y_train, domain = clean_tweets(X_train, h_train, y_train)
#print("X", X_train, y_train)
#print("D", domain)
cleaned_X, hashtags_test = clean_test_tweets(df_test['Tweet'], hashtags_test)
#print(X_train)
#save_data("x_train.csv", pd.DataFrame(x_train))
#save_data("h_train.csv", pd.DataFrame(h_train))
#save_data("y_train.csv", pd.DataFrame(y_train))
#
#save_data("X_test.csv", pd.DataFrame(X_test))
#save_data("h_test.csv", pd.DataFrame(h_test))
#save_data("y_test.csv", pd.DataFrame(y_test))
print("dummy SVM_noPunct_hashtags.csv")
print("Write Complete: ")
'''x_train = pd.read_csv('x_train.csv', sep='\n', skip_blank_lines=False).fillna('')['Val'].values
h_train = pd.read_csv('h_train.csv', sep='\n', skip_blank_lines=False).fillna('')['Val'].values
y_train = pd.read_csv('y_train.csv', sep='\n', skip_blank_lines=False).fillna('')['Val'].values

X_test = pd.read_csv('X_test.csv', sep='\n', skip_blank_lines=False).fillna('')['Val'].values
h_test = pd.read_csv('h_test.csv', sep='\n', skip_blank_lines=False).fillna('')['Val'].values
y_test = pd.read_csv('y_test.csv', sep='\n', skip_blank_lines=False).fillna('')['Val'].values
print(h_train)   '''

print("Clean complete: ")


vec1, tf1 = Tfidf(X_train)
vec2, tf2 = Tfidf(h_train)
tfidf_out = sparse.hstack((vec1,vec2),format='csr')

'''from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
from sklearn.kernel_approximation import Nystroem
feature_map_nystroem = Nystroem(gamma=.2,
                            random_state=1,
                            n_components=300)'''


#tfidf_out =  feature_map_nystroem.fit_transform(tfidf_out)

#clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=50, random_state=42)
clf = svm.SVC(gamma='scale', degree=4)
clf.fit(tfidf_out, y_train)


train_predict= clf.predict(tfidf_out)
print("Train accuracy: ", np.mean(train_predict==y_train))

#text_clf = Pipeline([('vect', CountVectorizer( min_df=5, max_df=0.7, stop_words='english', lowercase=True)),
#                     ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
#                     ('clf', LogisticRegression())])
#text_clf = Pipeline([('vect', CountVectorizer(max_df=0.85, min_df=5)), ('tfidf', TfidfTransformer()),
#                         ('clf-svm',
#                          SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])


#dev = np.concatenate(X_test,h_test)
#X_test, h_test, y_test = clean_tweets(X_test, h_test, y_test)

vec1 = tf1.transform(X_test)
vec2 = tf2.transform(h_test)
dev = sparse.hstack((vec1,vec2),format='csr')
#dev =  rbf_feature.fit_transform(dev)
dev_predict = clf.predict(dev)
print("Dev accuracy: ", np.mean(dev_predict==y_test))



'''vec1 = tf1.transform(cleaned_X)
vec2 = tf2.transform(hashtags_test)
test = sparse.hstack((vec1,vec2),format='csr')
predicted = clf.predict(test)


tocsv = np.empty(shape=(len(predicted),2),dtype=int)
for i in range(tocsv.shape[0]):
    tocsv[i][0] = i+1
    tocsv[i][1] = predicted[i]

np.savetxt(Path('dummy SVM_noPunct_hashtags.csv'), tocsv, delimiter=",",
                   fmt='%i', header='Id,Predicted',comments='')
type(predicted)
print("Test completed", predicted)'''





