import numpy as np
import itertools
import matplotlib.pyplot as plt


class Evaluator:

    def __init__(self, gold_star_file, referent=False):
        self.gold_standard = self.get_gold_standard(gold_star_file, referent)
        self.first_learned = {}

    def get_gold_standard(self, file_name, referent=False):
        gold_lexicon = {}
        with open(file_name, 'r') as file:
            data = file.readlines()
            if referent:
                pass
            else:
                data = [line.strip().split() for line in data if line.strip() != '']
                data = [[line[0], " ".join(line[1:])] for line in data]
                preproc_data = [[line[0], list(map(lambda x: x[:x.find(":")], line[1].split(',')))[:-1]] for line in
                                data]
            for line in preproc_data:
                gold_lexicon[line[0]] = line[1]
            return gold_lexicon

    def get_proportion_words_learned(self, learner, freq=0):
        words_learned = 0
        wl_list = []
        md, wm, fm = learner.meaning_dist, learner.word_mapping, learner.feature_mapping
        total_words_seen = {word: learner.words_seen[word] for word in learner.words_seen if
                            learner.words_seen[word] >= freq}
        for word in total_words_seen:
            word_index = wm[word]
            try:
                gold_features = self.gold_standard[word]
            except:
                continue
            comprehension_score = 0
            for feature in gold_features:
                try:
                    feature_index = fm[feature]
                except:
                    continue
                comprehension_score += md[word_index, feature_index]
            if comprehension_score >= 0.7:
                words_learned += 1
                wl_list.append(word)
            else:
                pass
        return words_learned / len(total_words_seen)

    def view_fast_map_behaviour(self, learner, freq=0):
        words_learned = 0
        wl_list = []
        md, wm, fm = learner.meaning_dist, learner.word_mapping, learner.feature_mapping
        total_words_seen = {word: learner.words_seen[word] for word in learner.words_seen if
                            learner.words_seen[word] >= freq}
        for word in total_words_seen:
            word_index = wm[word]
            try:
                gold_features = self.gold_standard[word]
            except:
                continue
            comprehension_score = 0
            for feature in gold_features:
                try:
                    feature_index = fm[feature]
                except:
                    continue
                comprehension_score += md[word_index, feature_index]
            if comprehension_score >= 0.7:
                words_learned += 1
                wl_list.append(word)
                if word not in self.first_learned:
                    self.first_learned[word] = total_words_seen[word]
            else:
                pass
        return words_learned / len(total_words_seen)