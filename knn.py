from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import math

#Because the denominator of cosine distance gets small
#the program will treat scalar as 0 or nan.
#The line below prevents it from treating it as 0 division
np.seterr(invalid = 'ignore')

def cos_distance(vector1, vector2, cols):
    x = vector1.toarray()
    y = vector2.toarray()
    dot_product = x.dot(y.T)
    vector_length1 = np.linalg.norm(x)
    vector_length2 = np.linalg.norm(y)
    distance = dot_product/(vector_length1*vector_length2)
    return distance

def kNearestNeighbor(train_data, test_instance, labels, rows, cols, k):
    distances = []
    for train_row in range(num_rows):
        distance_metric = cos_distance(train_data.getrow(train_row), test_instance, num_cols)
        distances.append((distance_metric, labels[train_row]))
    sorted(distances, key = lambda x: x[0])
    return majority(distances, k)

def majority(list, k):
    positive = 0
    negative = 0
    for i in range(k):
        if(list[i][1] == 1):
            positive += 1
        else:
            negative += 1
    if(positive >= negative):
        return "+1\n"
    return "-1\n"

#Reads testdata and writes to trainingdata, getting rid of all the labels

train_file = open("trainhw1.txt", "r")
train_corpus = []
labels = []
for line in train_file:
    string = line.replace('+1\t', '')
    if(string == line):
        string = line.replace('-1\t', '')
        labels.append(-1)
    else:
        labels.append(1)
    train_corpus.append(string)

train_file.close()

#Put documents in test data into corpus
test_file = open("testdatahw1.txt", "r")
test_corpus = []
for line in test_file:
    test_corpus.append(line)
test_file.close()


#Converts the corpus into a document-term matrix
vectorizer = TfidfVectorizer(stop_words = 'english')
train_matrix = vectorizer.fit_transform(train_corpus)
test_matrix = vectorizer.transform(test_corpus)

num_rows = train_matrix.get_shape()[0]
num_cols = train_matrix.get_shape()[1]

prediction_file = open("prediction.txt", "w")
for test_instance in range(num_rows):
    print("Review: %d", test_instance)
    sentiment = kNearestNeighbor(train_matrix, test_matrix.getrow(test_instance), labels, num_rows, num_cols, 130)
    prediction_file.write(sentiment)
prediction_file.close()
