import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        q_emb = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        cand_ids, cand_embeddings = self.__load_embeddings_by_tag(tag_name)
        best_cand = pairwise_distances_argmin([q_emb], cand_embeddings, metric='cosine')

        best_thread = cand_ids.reset_index()['post_id'].loc[best_cand[0]]
        
        return best_thread


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.__init_chitchat_bot()

    def __init_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # Create an instance of the ChatBot class.
        # Set a trainer set_trainer(ChatterBotCorpusTrainer) for the ChatBot.
        # Train the ChatBot with "chatterbot.corpus.english" param.
        # Note that we use chatterbot==0.7.6 in this project. 
        # You are welcome to experiment with other versions but they might have slightly different API.
        
        self.chatbot = ChatBot("StackOverflow answering")
        trainer = ChatterBotCorpusTrainer(self.chatbot)
        trainer.train("chatterbot.corpus.english")

       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
            
            return self.ANSWER_TEMPLATE % (tag, thread_id)
