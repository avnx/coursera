import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    starspace_embeddings = dict()

    for line in open('starspace_model.tsv'):
        line_split = line.split('\t')
        starspace_embeddings[line_split[0]] = np.array([float(el) for el in line_split[1:]])

    return starspace_embeddings, len(starspace_embeddings[line_split[0]])


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.

    tokenizer = RegexpTokenizer(r'\w+')
    result = []
    for word in tokenizer.tokenize(question):
        if word in embeddings:
            result.append(embeddings[word])
    if result == []:
        return np.zeros(dim)

    return np.vstack(result).mean(axis=0)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
