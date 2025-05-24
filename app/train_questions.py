import os
import numpy as np
import pandas as pd
import pickle
import sklearn.feature_extraction.text
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset
import torch

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

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            enc = tokenizer(texts, truncation=True, padding=True)
            self.encodings = enc
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    labels = np.array([0 if t == 'dialogue' else 1 for t in intent])
    dataset = TextDataset(list(text), labels)
    training_args = TrainingArguments(output_dir='/models/intent', num_train_epochs=1, per_device_train_batch_size=8, logging_steps=10)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(RESOURCE_PATH['INTENT_RECOGNIZER'])
    tokenizer.save_pretrained(RESOURCE_PATH['INTENT_TFIDF_VECTORIZER'])


def make_tag_classifier(questions_df):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    labels = sorted(questions_df['tag'].unique())
    label_map = {l: i for i, l in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))

    class TextDataset(Dataset):
        def __init__(self, texts, labs):
            enc = tokenizer(texts, truncation=True, padding=True)
            self.encodings = enc
            self.labels = [label_map[l] for l in labs]
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    dataset = TextDataset(questions_df['title'].tolist(), questions_df['tag'].tolist())
    training_args = TrainingArguments(output_dir='/models/tag', num_train_epochs=1, per_device_train_batch_size=8, logging_steps=10)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(RESOURCE_PATH['TAG_CLASSIFIER'])
    tokenizer.save_pretrained(RESOURCE_PATH['TAG_TFIDF_VECTORIZER'])


def make_embeddings(sentences):
    """Compute sentence embeddings for question titles."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(sentences['title'].tolist())


def make_embeddings4tags(questions_df):
    counts_by_tag = questions_df.groupby('tag')['post_id'].count()

    os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

    for tag, count in counts_by_tag.items():
        print('building a model for:', tag)
        tag_posts = questions_df[questions_df['tag'] == tag]
        tag_post_ids = tag_posts['post_id'].values

        vectors = make_embeddings(tag_posts)

        filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))

        with open(filename, 'wb') as f:
            pickle.dump((tag_post_ids, vectors), f, pickle.HIGHEST_PROTOCOL)


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

