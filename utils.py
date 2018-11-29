import nltk
import re
import pickle
import numpy as np

nltk.download('stopwords')
import nltk.corpus

replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
bad_symbols_re = re.compile('[^0-9a-z #+_]')
stopwords_set = set(nltk.corpus.stopwords.words('english'))


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """

    result = np.zeros(dim)
    i = 0
    for word in question.split():
        if word in embeddings:
            result = result + embeddings[word]
            i += 1

    if i > 0:
        result = result / i

    return result


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    embeddings = {}
    for line in open(embeddings_path):
        l = line.split()
        embeddings[l[0]] = np.array(l[1:], dtype=np.float32)

    embeddings_dim = len(l[1:])

    return embeddings, embeddings_dim


RESOURCE_PATH = {
    'DIALOGUES': 'data/dialogues.tsv',
    'QUESTIONS': 'data/tagged_posts.tsv',
    'INTENT_TFIDF_VECTORIZER': 'models/intent_tfidf_vectorizer.pkl',
    'INTENT_RECOGNIZER': 'models/intent_recognizer.pkl',
    'TAG_TFIDF_VECTORIZER': 'models/tag_tfidf_vectorizer.pkl',
    'TAG_CLASSIFIER': 'models/tag_classifier.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'models/thread_embeddings_by_tags'
}