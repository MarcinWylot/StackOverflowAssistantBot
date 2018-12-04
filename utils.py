import re
import pickle
import nltk.corpus

nltk.download('stopwords')
nltk.download('punkt')

replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
bad_symbols_re = re.compile('[^0-9a-z #+_]')
stopwords_set = set(nltk.corpus.stopwords.words('english'))


def text_prepare(text):
    """
    Transform text to lower case.
    Clean text from unwanted symbols and stopwords.
    :param text: raw text
    :return: cleaned text
    """
    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def unpickle_file(filename):
    """
    unpickle a file
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


RESOURCE_PATH = {
    'DIALOGUES': 'data/dialogues.tsv',
    'QUESTIONS': 'data/tagged_posts.tsv',
    'INTENT_TFIDF_VECTORIZER': 'models/intent_tfidf_vectorizer.pkl',
    'INTENT_RECOGNIZER': 'models/intent_recognizer.pkl',
    'TAG_TFIDF_VECTORIZER': 'models/tag_tfidf_vectorizer.pkl',
    'TAG_CLASSIFIER': 'models/tag_classifier.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'models/thread_embeddings_by_tags'
}
