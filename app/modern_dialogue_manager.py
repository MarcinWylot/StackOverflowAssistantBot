import os
import numpy as np
from transformers import pipeline, Conversation
from sentence_transformers import SentenceTransformer
import utils

RESOURCE_PATH = utils.RESOURCE_PATH

class ThreadRanker(object):
    """Ranks questions similar to what a client is asking within a tag using
    semantic embeddings from SentenceTransformers."""
    def __init__(self):
        self.thread_embeddings_folder = RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER']
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, f"{tag_name}.pkl")
        thread_ids, thread_vectors = utils.unpickle_file(embeddings_path)
        return thread_ids, np.array(thread_vectors)

    def get_best_thread(self, question, tag_name):
        thread_ids, thread_vectors = self.__load_embeddings_by_tag(tag_name)
        question_vec = self.model.encode([question])[0]
        norms = np.linalg.norm(thread_vectors, axis=1) * np.linalg.norm(question_vec)
        sims = thread_vectors.dot(question_vec) / (norms + 1e-10)
        best_idx = sims.argsort()[-3:][::-1]
        return [(thread_ids[i], float(sims[i])) for i in best_idx]

class DialogueManager(object):
    """Modern DialogueManager using transformers."""
    def __init__(self):
        print("Loading transformer models ...")
        self.intent_recognizer = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.tag_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.dialogue_bot = pipeline('conversational', model='microsoft/DialoGPT-medium')
        self.thread_ranker = ThreadRanker()
        self.tags = ['python', 'java', 'javascript', 'c#', 'php']
        self.ANSWER_TEMPLATE = 'I think its about {}.\nThese threads might help you:'
        self.STACKOVERFLOW_URL_TEMPLATE = 'https://stackoverflow.com/questions/{} (score: {:.2f})'

    def generate_answer(self, question):
        intent_res = self.intent_recognizer(question, ['dialogue', 'question'])
        intent = intent_res['labels'][0]

        if intent == 'dialogue':
            conv = Conversation(question)
            response = self.dialogue_bot(conv)
            return str(response.generated_responses[-1])
        else:
            tag_res = self.tag_classifier(question, self.tags)
            tag = tag_res['labels'][0]
            threads = self.thread_ranker.get_best_thread(question, tag)
            urls = [self.STACKOVERFLOW_URL_TEMPLATE.format(tid, score) for tid, score in threads]
            return '\n'.join([self.ANSWER_TEMPLATE.format(tag)] + urls)
