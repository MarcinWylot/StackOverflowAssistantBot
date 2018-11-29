import os
import nltk

import utils



# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer

RESOURCE_PATH = utils.RESOURCE_PATH

class ThreadRanker(object):
    def __init__(self):
        self.thread_embeddings_folder = RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")

        thread_ids, thread_embeddings = utils.unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):

        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        vector = thread_embeddings.infer_vector(nltk.tokenize.word_tokenize(question))

        best = thread_embeddings.docvecs.most_similar(positive=[vector], topn=3)
        results = []
        for b in best:
            results.append((thread_ids[b[0]], b[1]))

        return results


class DialogueManager(object):
    def __init__(self):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = utils.unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])
        self.intent_tfidf_vectorizer = utils.unpickle_file(RESOURCE_PATH['INTENT_TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about {}.\nThis threads might help you:'
        self.STACKOVERFLOW_URL_TEMPLATE = 'https://stackoverflow.com/questions/{} (score: {:.2f})'

        # Goal-oriented part:
        self.tag_classifier = utils.unpickle_file(RESOURCE_PATH['TAG_CLASSIFIER'])
        self.tag_tfidf_vectorizer = utils.unpickle_file(RESOURCE_PATH['TAG_TFIDF_VECTORIZER'])
        self.thread_ranker = ThreadRanker()

        # chatter:
        # self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        self.chatbot = ChatBot('stackoverflow-assistant')

        # Create a new trainer for the chatbot
        trainer = ChatterBotCorpusTrainer(self.chatbot.storage)

        # Train the chatbot based on the english corpus
        trainer.train("chatterbot.corpus.english")

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        prepared_question = utils.text_prepare(question)
        features = self.intent_tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            # response = self.chatbot.get_response(question)
            response = 'chitchat'
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            features = self.tag_tfidf_vectorizer.transform([prepared_question])
            tag = self.tag_classifier.predict(features)[0]

            # Pass prepared_question to thread_ranker to get predictions.
            threads = self.thread_ranker.get_best_thread(prepared_question, tag)

            urls = []
            for thread in threads:
                urls.append(self.STACKOVERFLOW_URL_TEMPLATE.format(thread[0], thread[1]))

            return '\n'.join([self.ANSWER_TEMPLATE.format(tag)] + urls)
