import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


import numpy as np

text = pd.read_csv('train_tweets.txt', sep='\t', header=None)
test = pd.read_csv('test_tweets_unlabeled.txt', names=['Tweets'], sep=' \n\t:', engine='python', header=None)


def Tfidf(doc):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(doc)

    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    # print(df)

    print(tfidf_vectorizer_vectors.shape)
    return tfidf_vectorizer_vectors
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


'''text_clf = Pipeline([('vect', CountVectorizer(max_features=1000, max_df=0.85)), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = text_clf.fit(text[1], text[0])


train_predict = text_clf.predict(text[1])
print("Accuracy: ", np.mean(train_predict==text[0]))

predicted = text_clf.predict(test['Tweets'])


for i in range(tocsv.shape[0]):
    tocsv[i][0] = i+1
    tocsv[i][1] = predicted[i]

print(tocsv)


np.savetxt(Path('predictions.csv'), tocsv, delimiter=",",
                   fmt='%10.2f', header='Id,Predicted',comments='')
type(predicted)
print("NB completed", predicted)'''


text_clf_svm = Pipeline([('vect', CountVectorizer(max_features=10000, max_df=0.85)), ('tfidf', TfidfTransformer()),
                         ('clf-svm',
                          SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
text_clf_svm = text_clf_svm.fit(text[1], text[0])
predicted = text_clf_svm.predict(test['Tweets'])
print("SVM completed", predicted)
tocsv = np.empty(shape=(len(predicted),2),dtype=int)
for i in range(tocsv.shape[0]):
    tocsv[i][0] = i+1
    tocsv[i][1] = predicted[i]

np.savetxt(Path('basic_predictions_svm.csv'), np.array(tocsv), delimiter=",",
                   fmt='%i', header='Id,Predicted',comments='')