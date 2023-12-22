import pandas as pd
import nltk
import warnings
import re
warnings.filterwarnings("ignore")

data = pd.read_csv(r"./sms_spam.csv")
y = data['type']    # spam or ham - target
X = data['text']    # message text - input data
# print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
# print('X_train:', X_train.head())
import string
for ch in string.punctuation:
    X_train = X_train.str.replace(ch, "")  # remove punctuation

X_train = X_train.str.lower()  # make lowercase
# print('X_train_lower:', X_train)
from nltk.corpus import stopwords
noise = stopwords.words('english')
"""DO THIS TO REMOVE NOISE FROM A STRING: """   # removing stop_words
for st in noise:
    X_train = X_train.apply(lambda x: ' '.join([word for word in x.split()
                                                if word not in noise]))
# print(X_train.head(10))

from nltk.tokenize import word_tokenize
X_train = X_train.apply(word_tokenize)  # tokenizing a dataframe into unigrams
# print(X_train.head())
porter = nltk.PorterStemmer()  # lemmatization/stemming
X_train = X_train.apply(lambda x: [porter.stem(y) for y in x])  # stemming each word in str
X_train = X_train.apply(lambda x: ' '.join(x))   # a column of lists is put back to str for vectorizer
# print(X_train.head(10))

"""VECTORIZATION"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
vectorizer_metrics_df = pd.DataFrame(columns=['n-grams', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
# will be used later for metrics


'''CHOOSING THE NUMBER OF N_GRAMS'''
for i in range (1, 5):
    smart_vectorizer = CountVectorizer(ngram_range=(1, i))  # 1, 2 - bigrams. 1, 3 - trigrams, 1, 1 -unig.
    smart_vectorized_x_train = smart_vectorizer.fit_transform(X_train)
# classification

    from sklearn.naive_bayes import MultinomialNB  # naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(smart_vectorized_x_train, y_train)   # fit data to classifier
    smart_vectorized_x_test = smart_vectorizer.transform(X_test)
    pred = clf.predict(smart_vectorized_x_test)
    report_count = classification_report(y_test, pred)

    lines = report_count.strip().split('\n')
    """the following code is used to concat all different report DFs into one by tag spam"""

    for line in lines[2:4]:
        tokens = re.split(r'\s+', line.strip())
        class_name, precision, recall, f1_score, support = tokens[0],\
                                                           float(tokens[1]),\
                                                           float(tokens[2]), \
                                                           float(tokens[3]), \
                                                           int(tokens[4])

        if class_name in ['spam']:
            vectorizer_metrics_df = vectorizer_metrics_df._append({
                'n-grams': i,
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'Support': support
            }, ignore_index=True)

print('metrics for different n_grams within Count Vectorizer: ', vectorizer_metrics_df, sep='\n', end='\n')
# appropriate number of n_grams is 2

smart_vectorizer = CountVectorizer(ngram_range=(1, 2))
smart_vectorized_x_train = smart_vectorizer.fit_transform(X_train)
# classification
from sklearn.naive_bayes import MultinomialNB  # naive Bayes classifier
clf = MultinomialNB()
clf.fit(smart_vectorized_x_train, y_train)   # fit data to classifier
smart_vectorized_x_test = smart_vectorizer.transform(X_test)
pred_n_grams = clf.predict(smart_vectorized_x_test)

""""TF-IDF, bigrams are used here asw"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # vectorizing using b-grams
tfidf_vectorized_x_train = tfidf_vectorizer.fit_transform(X_train)  # training the model
clf.fit(tfidf_vectorized_x_train, y_train)  # fitting the model
tfidf_vectorized_x_test = tfidf_vectorizer.transform(X_test)  # vectorizing the test data
pred_tf_idif = clf.predict(tfidf_vectorized_x_test)
report_tf_idf = classification_report(y_test, pred)
print('metrics for TF-IDF, bigrams: ', report_tf_idf, sep='\n', end='\n')

"""char n_grams"""
char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 6))
char_vectorized_x_train = char_vectorizer.fit_transform(X_train)
clf.fit(char_vectorized_x_train, y_train)
char_vectorized_x_test = char_vectorizer.transform(X_test)
pred = clf.predict(char_vectorized_x_test)
report_char = classification_report(y_test, pred)
print('metrics for char n-grams: ', report_char, sep='\n', end='\n')

"""CONSTRUCTION OF A FINAL DF WITH BIGRAM COUNT, BIGRAM TF-IDF AND CHAR. TAG - SPAM"""
comparison_df = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score', 'Support'])
for i in range (3):
    if i == 0:
        lines = report_count.strip().split('\n')
        model = 'CountVect'
    elif i == 1:
        lines = report_tf_idf.strip().split('\n')
        model = 'TF_IDF'
    elif i == 2:
        lines = report_char.strip().split('\n')
        model = 'Char n-grams'
    for line in lines[2:4]:
        tokens = re.split(r'\s+', line.strip())
        class_name, precision, recall, f1_score, support = tokens[0], \
                                                           float(tokens[1]), \
                                                           float(tokens[2]), \
                                                           float(tokens[3]), \
                                                           int(tokens[4])
        if class_name in ['spam']:
            comparison_df = comparison_df._append({
            'Model': model,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Support': support
        }, ignore_index=True)

print('Final comparison DF for all models used: ', comparison_df, sep='\n')

# which model is better? e.g. has the highest average of scores
metrics_df = comparison_df.drop(columns='Model')
mean_rows = metrics_df.mean(axis=1)
print('mean values of metrics by model:', mean_rows, sep='\n', end='\n')
print('CountVectorizer and TF-IDF show the same mean values so either can be used.')
test = 'you have won over 2000 pounds today wow sdjkfhjdsf'
data = {
    'type': ' ',
    'text': [test]
}
tfidf_vectorized_x_test = char_vectorizer.transform(data)
pred_test = clf.predict(tfidf_vectorized_x_test)
print(pred_test)