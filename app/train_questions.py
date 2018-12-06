import os
import numpy as np
import pandas as pd
import pickle
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.multiclass
import gensim.models.doc2vec
import nltk.tokenize

import utils

RESOURCE_PATH = utils.RESOURCE_PATH


def tfidf_features(text, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""

    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1, 2),
                                                            token_pattern='(\S+)')

    text_tfidf = tfidf.fit_transform(text)

    with open(vectorizer_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(tfidf, f, pickle.HIGHEST_PROTOCOL)

    return text_tfidf


def make_intent_recognizer():
    dialogues_df = pd.read_csv(RESOURCE_PATH['DIALOGUES'], sep='\t')
    sample_size = len(dialogues_df)
    questions_df = pd.read_csv(RESOURCE_PATH['QUESTIONS'], sep='\t').sample(sample_size)

    dialogues_df['text'] = dialogues_df['text'].apply(utils.text_prepare)
    questions_df['title'] = questions_df['title'].apply(utils.text_prepare)

    text = np.concatenate([dialogues_df['text'].values, questions_df['title'].values])
    intent = ['dialogue'] * dialogues_df.shape[0] + ['question'] * questions_df.shape[0]

    text_tfidf = tfidf_features(text, RESOURCE_PATH['INTENT_TFIDF_VECTORIZER'])

    intent_recognizer = sklearn.linear_model.LogisticRegression(C=5, solver='lbfgs', max_iter=1000).fit(text_tfidf, intent)
    with open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb') as f:
        pickle.dump(intent_recognizer, f, pickle.HIGHEST_PROTOCOL)


def make_tag_classifier(questions_df):
    questions_tfidf = tfidf_features(questions_df['title'].values, RESOURCE_PATH['TAG_TFIDF_VECTORIZER'])

    tag_classifier = sklearn.linear_model.LogisticRegression(C=5, solver='lbfgs', max_iter=1000, multi_class='ovr')
    tag_classifier.fit(questions_tfidf, questions_df['tag'].values)

    with open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb') as f:
        pickle.dump(tag_classifier, f, pickle.HIGHEST_PROTOCOL)


def tag_tokenize(sentences):
    for i, v in enumerate(sentences):
        yield gensim.models.doc2vec.TaggedDocument(nltk.tokenize.word_tokenize(v), [i])


def make_embeddings(sentences):
    documents = list(tag_tokenize(sentences['title']))

    # doc2vec parameters
    vector_size = 300
    window_size = 15
    train_epoch = 100
    worker_count = 4

    model = gensim.models.doc2vec.Doc2Vec(documents, vector_size=vector_size, window=window_size,
                                          workers=worker_count,  epochs=train_epoch)

    return model


def make_embeddings4tags(questions_df):
    counts_by_tag = questions_df.groupby('tag')['post_id'].count()

    os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

    for tag, count in counts_by_tag.items():
        print('building a model for:', tag)
        tag_posts = questions_df[questions_df['tag'] == tag]
        tag_post_ids = tag_posts['post_id'].values

        model = make_embeddings(tag_posts)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        # Dump post ids and vectors to a file.
        filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))

        with open(filename, 'wb') as f:
            pickle.dump((tag_post_ids, model), f, pickle.HIGHEST_PROTOCOL)


def prepare_questions():
    questions_df = pd.read_csv(RESOURCE_PATH['QUESTIONS'], sep='\t')
    questions_df['title'] = questions_df['title'].apply(utils.text_prepare)

    return questions_df


def main():
    print('make_intent_recognizer()')
    make_intent_recognizer()
    print('prepare_questions()')
    questions = prepare_questions()
    print('make_tag_classifier()')
    make_tag_classifier(questions)
    print('make_embeddings4tags()')
    make_embeddings4tags(questions)
    print('DONE')


if __name__ == "__main__":
    main()

