import numpy as np
import itertools
import time
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:

    def __init__(self, gold_star_file, referent=False):
        self.gold_standard = self.get_gold_standard(gold_star_file, referent)
        self.referent = referent

    def get_gold_standard(self, file_name, referent=False):
        gold_lexicon = {}
        with open(file_name, 'r') as file:
            data = file.readlines()
            if referent:
                pass
            else:
                data = [line.strip().split() for line in data if line.strip() != '']
                data = [[line[0], " ".join(line[1:])] for line in data]
                preproc_data = [[line[0], list(map(lambda x: x[:x.find(":")], line[1].split(',')))[:-1]] for line in data]
            for line in preproc_data:
                gold_lexicon[line[0]] = line[1]
            return gold_lexicon

    def get_proportion_words_learned(self, learner, referent=False, freq=0):
        words_learned = 0
        wl_list = []
        md, wm, fm = learner.meaning_dist, learner.word_mapping, learner.feature_mapping
        if not referent:
            for word in wm:
                word_index = wm[word]
                try:
                    gold_features = self.gold_standard[word]
                except:
    #                 print("=========WORD {} IS NOT IN LEXICON =========".format(word))
                    continue
                comprehension_score = 0
                for feature in gold_features:
                    try:
                        feature_index = fm[feature]
                    except:
    #                     print("=========FEATURE {} IS NOT IN LEXICON =========".format(feature))
                        continue
                    comprehension_score += md[word_index, feature_index]
                if comprehension_score >= 0.7:
    #                 print("=========WORD {} HAS BEEN LEARNED WITH SCORE {} =========".format(word, comprehension_score))
                    if learner.words_seen[word] >= freq:
                        words_learned += 1
                        wl_list.append(word)
                else:
                    pass
        else:
            for word in wm:
                word_index = wm[word]
                gold_referent = self.gold_standard[word]
                gold_vector = learner.get_referent_vector(gold_referent)

                actual_vector = md[word_index, :]
                comprehension_score = learner.get_cosine_similarity(actual_vector, gold_vector)
                if comprehension_score >= 0.7:
                    words_learned += 1
#                 print("=========WORD {} HAS NOT BEEN LEARNED !!!=========".format(word))

        total_words_seen = {word: learner.words_seen[word] for word in learner.words_seen if learner.words_seen[word] >= freq}

        return words_learned / len(total_words_seen)


    def get_avg_acq_freq_more(self, learner, referent=False, freq=0):
        words_learned = 0
        wl_list = []
        md, wm, fm = learner.meaning_dist, learner.word_mapping, learner.feature_mapping
        total_acq = 0
        if not referent:
            for word in wm:
                if word in learner.words_seen and learner.words_seen[word] > freq:
                    word_index = wm[word]
                    try:
                        gold_features = self.gold_standard[word]
                        gold_vector = learner.get_referent_vector(gold_features)
                    except:
        #                 print("=========WORD {} IS NOT IN LEXICON =========".format(word))
                        continue
                    actual_vector = md[word_index, :]
                    total_acq += learner.get_cosine_similarity(actual_vector, gold_vector)
        else:
            for word in wm:
                if word in learner.words_seen and learner.words_seen[word] > freq:
                    word_index = wm[word]
                    gold_referent = self.gold_standard[word]
                    gold_vector = learner.get_referent_vector(gold_referent)

                    actual_vector = md[word_index, :]
                    total_acq += learner.get_cosine_similarity(actual_vector, gold_vector)

        total_words_seen = {word: learner.words_seen[word] for word in learner.words_seen if learner.words_seen[word] >= freq}


        return total_acq / len(total_words_seen)

    def get_avg_acq_freq_less(self, learner, referent=False, freq=0):
        words_learned = 0
        wl_list = []
        md, wm, fm = learner.meaning_dist, learner.word_mapping, learner.feature_mapping
        total_acq = 0
        if not referent:
            for word in wm:
                if word in learner.words_seen and learner.words_seen[word] < freq:
                    word_index = wm[word]
                    try:
                        gold_features = self.gold_standard[word]
                        gold_vector = learner.get_referent_vector(gold_features)
                    except:
        #                 print("=========WORD {} IS NOT IN LEXICON =========".format(word))
                        continue
                    actual_vector = md[word_index, :]
                    total_acq += learner.get_cosine_similarity(actual_vector, gold_vector)
        else:
            for word in wm:
                if word in learner.words_seen and learner.words_seen[word] < freq:
                    word_index = wm[word]
                    gold_referent = self.gold_standard[word]
                    gold_vector = learner.get_referent_vector(gold_referent)

                    actual_vector = md[word_index, :]
                    total_acq += learner.get_cosine_similarity(actual_vector, gold_vector)

        total_words_seen = {word: learner.words_seen[word] for word in learner.words_seen if learner.words_seen[word] < freq}


        return total_acq / len(total_words_seen)
