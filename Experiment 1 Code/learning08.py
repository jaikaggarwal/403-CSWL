import numpy as np
import itertools
import matplotlib.pyplot as plt

class Learner:

    def __init__(self, utterances, scenes, smoothing_param=1e-7):
        self.num_words, self.word_mapping = self.get_word_mapping(utterances)
        self.num_features, self.feature_mapping = self.get_feature_mapping(scenes)
        self.meaning_dist = self.init_meaning_dist(self.num_words, self.num_features)
        self.association_scores = self.init_association_scores(self.num_words, self.num_features)
        self.smoothing_param = smoothing_param
        self.words_seen = {}
        self.first_seen = {}

    def get_word_mapping(self, utterances):
        """ Get the size of the set of words present in the data, as well as a mapping of word:index in the matrix. """
        words = list(set(itertools.chain.from_iterable(utterances)))
        words.sort()
        word_map = {}
        for i in range(len(words)):
            word_map[words[i]] = i
        return len(words), word_map

    def get_feature_mapping(self, scenes):
        """ Get the size of the set of features present in the data, as well as a mapping of feature:index in the matrix. """

        features = list(set(itertools.chain.from_iterable(scenes)))
        features.sort()
        feature_map = {}
        for i in range(len(features)):
            feature_map[features[i]] = i
        return len(features), feature_map

    def init_meaning_dist(self, num_words, num_features):
        return np.ones((num_words, num_features), dtype=float) / num_features

    def init_association_scores(self, num_words, num_features):
        return np.zeros((num_words, num_features))

    def train_on_pair(self, utterance, scene, t):
        alignment_scores, words_in_utterance = self.get_alignment_scores(utterance, scene, t)
        self.update_association_scores(alignment_scores)
        self.update_meaning_probabilities(alignment_scores, words_in_utterance)

    def get_alignment_scores(self, utterance, scene, t):
        """Find alignment scores of word given feature. The numerator is the probability of a feature given a word
        at time t-1. The denominator is the sum of the probability of a given feature across all words. This simulates bias.

        Returns a dictionary where keys are the indices of words, and the values are sub-dictionaries. The keys of these
        sub-dictionaries are indices of features, and the value is the alignment score of the feature with the given word."""

        alignment_scores = {}
        # Only normalize over words in utterance, not every word in the vocabulary
        words_in_utterance = np.array([self.word_mapping[word] for word in utterance])
        for word in utterance:
            #             self.words_seen[word] += 1
            if word not in self.words_seen:
                self.words_seen[word] = 1
            else:
                self.words_seen[word] += 1
            if word not in self.first_seen:
                self.first_seen[word] = t
            word_index = self.word_mapping[word]
            #             print(word, word_index)
            feature_score = {}
            for feature in scene:
                feature_index = self.feature_mapping[feature]
                curr_prob = self.meaning_dist[word_index, feature_index]
                denom = np.sum(self.meaning_dist[words_in_utterance, feature_index])
                feature_score[feature_index] = curr_prob / denom
            alignment_scores[word_index] = feature_score


        return alignment_scores, words_in_utterance

    def update_association_scores(self, alignment_scores):
        #         self.show_association_scores()
        for word in alignment_scores:
            feature_scores = alignment_scores[word]
            for feature in feature_scores:
                self.association_scores[word, feature] = self.association_scores[word, feature] + feature_scores[
                    feature]

        #         print("======== UPDATED ASSOCIATION SCORES ===========")
        #         self.show_association_scores()
        return

    def show_association_scores(self):
        for row in self.association_scores:
            print(row)

    def update_meaning_probabilities(self, alignment_scores, words_in_utterance):
        """This function takes the association scores at time t to update the probability distribution of P(feature|word)
        at time t."""

        for word in alignment_scores:
            feature_scores = alignment_scores[word]
            for feature in feature_scores:
                numerator = self.association_scores[word, feature]
                self.meaning_dist[word, feature] = numerator
            self.meaning_dist[word] += self.smoothing_param
        self.meaning_dist[words_in_utterance] = np.divide(self.meaning_dist[words_in_utterance],
                                                          np.sum(self.meaning_dist[words_in_utterance], axis=1).reshape(
                                                              -1, 1))
        return