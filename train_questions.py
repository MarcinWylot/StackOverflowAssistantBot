import os
import numpy as np
import pandas as pd
import pickle
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.multiclass

import utils


RESOURCE_PATH = {
    'DIALOGUES': 'data/dialogues.tsv',
    'QUESTIONS': 'data/tagged_posts.tsv',
    'INTENT_TFIDF_VECTORIZER': 'models/intent_tfidf_vectorizer.pkl',
    'INTENT_RECOGNIZER': 'models/intent_recognizer.pkl',
    'TAG_TFIDF_VECTORIZER': 'models/tag_tfidf_vectorizer.pkl',
    'TAG_CLASSIFIER': 'models/tag_classifier.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'models/thread_embeddings_by_tags',
    'QUESTIONS_EMBEDDINGS': 'models/questions_embeddings.tsv' #made by starspace
}


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

    intent_recognizer = sklearn.linear_model.LogisticRegression(C=5).fit(text_tfidf, intent)
    pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))


def make_tag_classifier(questions_df):
    questions_tfidf = tfidf_features(questions_df['title'].values, RESOURCE_PATH['TAG_TFIDF_VECTORIZER'])

    tag_classifier = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression(C=5))
    tag_classifier.fit(questions_tfidf, questions_df['tag'].values)

    pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))


def make_embedings4tags(questions_df):
    questions_embeddings, questions_embeddings_dim = utils.load_embeddings(RESOURCE_PATH['QUESTIONS_EMBEDDINGS'])

    counts_by_tag = questions_df.groupby('tag')['post_id'].count()

    os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

    for tag, count in counts_by_tag.items():
        tag_posts = questions_df[questions_df['tag'] == tag]

        tag_post_ids = tag_posts['post_id'].values

        tag_vectors = np.zeros((count, questions_embeddings_dim), dtype=np.float32)
        for i, title in enumerate(tag_posts['title']):
            tag_vectors[i, :] = utils.question_to_vec(title, questions_embeddings, questions_embeddings_dim)

        # Dump post ids and vectors to a file.
        filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
        pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))


def prepare_questions():
    questions_df = pd.read_csv(RESOURCE_PATH['QUESTIONS'], sep='\t')
    questions_df['title'] = questions_df['title'].apply(utils.text_prepare)

    return questions_df


def main():
    make_intent_recognizer()
    questions = prepare_questions()
    make_tag_classifier(questions)
    make_embedings4tags(questions)


if __name__ == "__main__":
    main()
