import os
import nltk

import chatterbot
import chatterbot.trainers

import utils

RESOURCE_PATH = utils.RESOURCE_PATH


class ThreadRanker(object):
    """Ranks questions similar to what a client is asking within a tag,
    then it returns most similar questions' IDs."""
    def __init__(self):
        self.thread_embeddings_folder = RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        """
        Loads only embeddings for a tag.
        :param tag_name: tag of a question
        :return:  (thread_ids, thread_embeddings) for the "tag"
        """
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")

        thread_ids, thread_embeddings = utils.unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """
        Retrieves questions most similat to "question"
        :param question: question string, cleaned
        :param tag_name: tag of the queston
        :return:  a list of most similar questions' IDs
        """

        # first we load embeddings for our tag
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # let's try to infer vector for out question
        vector = thread_embeddings.infer_vector(nltk.tokenize.word_tokenize(question))

        # and finally we retrieve vectors that are most similar to the infered one
        best = thread_embeddings.docvecs.most_similar(positive=[vector], topn=3)
        results = []
        for b in best:
            results.append((thread_ids[b[0]], b[1]))

        return results


class DialogueManager(object):
    """
    Manages overall flow of a dialog with a client.
    """
    def __init__(self):
        print("Loading resources...")

        # intent recognition models
        self.intent_recognizer = utils.unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])
        self.intent_tfidf_vectorizer = utils.unpickle_file(RESOURCE_PATH['INTENT_TFIDF_VECTORIZER'])


        # domain specific models
        self.tag_classifier = utils.unpickle_file(RESOURCE_PATH['TAG_CLASSIFIER'])
        self.tag_tfidf_vectorizer = utils.unpickle_file(RESOURCE_PATH['TAG_TFIDF_VECTORIZER'])
        self.thread_ranker = ThreadRanker()

        # answer templates
        self.ANSWER_TEMPLATE = 'I think its about {}.\nThis threads might help you:'
        self.STACKOVERFLOW_URL_TEMPLATE = 'https://stackoverflow.com/questions/{} (score: {:.2f})'

        # dialogue bot
        self.dialogue_bot = self.create_dialogue_bot()

    def create_dialogue_bot(self):
        """
        Create and train a new ChatBot
        :return: ChatBot instance
        """

        # Create a new ChatBot
        bot = chatterbot.ChatBot('StackOverflowAssistant',
                      storage_adapter="chatterbot.storage.SQLStorageAdapter",
                      database='/tmp/chatterbot.database.sqlite3')

        # Create a new trainer for the dialogue bot
        trainer = chatterbot.trainers.ChatterBotCorpusTrainer(bot.storage)

        # Train the dialogue bot based on the english corpus
        trainer.train("chatterbot.corpus.english")
        return bot

    def generate_answer(self, question):
        """
        This method recognizes intent of a question and combines dialogue and domain specific questions processors.
        For a dialogue it uses ChatterBot.
        For a domain specific question it recognizes tag of a question and gets most similar questions.
        :param question: question from a user
        :return: test to be delivered to the user
        """
        # prepare question, cleaning
        prepared_question = utils.text_prepare(question)
        # compute features (tfidf)
        features = self.intent_tfidf_vectorizer.transform([prepared_question])
        # recognize intent (dialogue/question)
        intent = self.intent_recognizer.predict(features)

        # dialogue is server by ChatterBot
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.dialogue_bot.get_response(question)
            return response

        # question is server by our models
        else:
            # compute features with domain specific model
            features = self.tag_tfidf_vectorizer.transform([prepared_question])
            # recognize tag for a question
            tag = self.tag_classifier.predict(features)[0]

            # get threads that match best to the question and it's tag
            threads = self.thread_ranker.get_best_thread(prepared_question, tag)

            # finally we generate answers
            urls = []
            for thread in threads:
                urls.append(self.STACKOVERFLOW_URL_TEMPLATE.format(thread[0], thread[1]))

            return '\n'.join([self.ANSWER_TEMPLATE.format(tag)] + urls)


class SimpleDialogueManager(object):
    """
    For development, just a simple hello world answer.
    """
    @staticmethod
    def generate_answer(question):
        return "Hello, world!"
