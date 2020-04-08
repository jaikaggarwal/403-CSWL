import numpy as np
import itertools
import time
from sklearn.metrics.pairwise import cosine_similarity


class Learner:

    def __init__(self, utterances, scenes, referents=False, competition=None, smoothing_param=1e-7):
        self.referents = referents
        self.competition = competition
        self.num_words, self.word_mapping= self.get_word_mapping(utterances)
        self.num_features, self.feature_mapping = self.get_feature_mapping(scenes)
        self.meaning_dist = self.init_meaning_dist(self.num_words, self.num_features)
        self.association_scores = self.init_association_scores(self.num_words, self.num_features)
        self.smoothing_param = smoothing_param
        self.words_seen = {}


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

        if self.referents:
            features = list(itertools.chain.from_iterable(scenes))
            features = list(set(itertools.chain.from_iterable(features)))
        else:
            features = list(set(itertools.chain.from_iterable(scenes)))
        features.sort()
        feature_map = {}
        for i in range(len(features)):
            feature_map[features[i]] = i
        return len(features), feature_map


    def init_meaning_dist(self, num_words, num_features):
        return np.ones((num_words, num_features)) / num_features


    def init_association_scores(self, num_words, num_features):
        return np.zeros((num_words, num_features))


    def train_on_pair(self, utterance, scene):
        if self.referents and self.competition == 'ref_comp':
            alignment_scores, words_in_utterance = self.get_alignment_scores_ref_comp(utterance, scene)
        elif self.referents and self.competition == 'word_comp':
            alignment_scores, words_in_utterance = self.get_alignment_scores_word_comp(utterance, scene)
        elif self.referents:
            alignment_scores, words_in_utterance = self.get_alignment_scores_no_comp(utterance, scene)
        else:
            alignment_scores, words_in_utterance = self.get_alignment_scores_base(utterance, scene)
        self.update_association_scores(alignment_scores)
        self.update_meaning_probabilities(alignment_scores, words_in_utterance)


    def get_cosine_similarity(self, word_vec, referent_vec):
        word_vec = word_vec.reshape(1, -1)
        referent_vec = referent_vec.reshape(1, -1)
        return cosine_similarity(word_vec, referent_vec)[0][0]


    def get_referent_vector(self, referent):
        mask = np.array([self.feature_mapping[feature] for feature in referent])
        base_arr = np.zeros(self.num_features)
        base_arr[mask] = 1
        return base_arr


    def get_alignment_scores_base(self, utterance, scene):
        """Find alignment scores of word given feature. The numerator is the probability of a feature given a word
        at time t-1. The denominator is the sum of the probability of a given feature across all words. This simulates bias.

        Returns a dictionary where keys are the indices of words, and the values are sub-dictionaries. The keys of these
        sub-dictionaries are indices of features, and the value is the alignment score of the feature with the given word."""

        alignment_scores = {}

        # Only normalize over words in utterance, not every word in the vocabulary
        words_in_utterance = np.array([self.word_mapping[word] for word in utterance])
        for word in utterance:
            if word not in self.words_seen:
                self.words_seen[word] = 1
            else:
                self.words_seen[word] += 1
            word_index = self.word_mapping[word]
            feature_score = {}
            for feature in scene:
                feature_index = self.feature_mapping[feature]
                curr_prob = self.meaning_dist[word_index, feature_index]
                denom = np.sum(self.meaning_dist[words_in_utterance, feature_index])
                feature_score[feature_index] = curr_prob / denom
            alignment_scores[word_index] = feature_score
#         print("======== CALCULATED ALIGNMENT SCORES ===========")
        return alignment_scores, words_in_utterance


    def get_alignment_scores_no_comp(self, utterance, scene):
        alignment_scores = {self.word_mapping[word]: {} for word in utterance}

        words_in_utterance = np.array([self.word_mapping[word] for word in utterance])
        # Only normalize over words in utterance, not every word in the vocabulary
        for word in utterance:
            if word not in self.words_seen:
                self.words_seen[word] = 1
            else:
                self.words_seen[word] += 1
            word_index = self.word_mapping[word]
            for referent in scene:
                ref_vector = self.get_referent_vector(referent)
                word_vector = self.meaning_dist[word_index, :]
                sim = self.get_cosine_similarity(word_vector, ref_vector)
                for feature in referent:
                    feature_index = self.feature_mapping[feature]
                    if feature not in alignment_scores[word_index]:
                        alignment_scores[word_index][feature_index] = [sim]
                    else:
                        alignment_scores[word_index][feature_index].append(sim)
        return alignment_scores, words_in_utterance


    def get_alignment_scores_ref_comp(self, utterance, scene):
        num_words_in_utterance = len(utterance)
        num_refs_in_scene = len(scene)
        word_ref_matrix = np.zeros((num_words_in_utterance, num_refs_in_scene))

        alignment_scores = {self.word_mapping[word]: {} for word in utterance}
        # Only normalize over words in utterance, not every word in the vocabulary
        words_in_utterance = np.array([self.word_mapping[word] for word in utterance])
        for i in range(num_words_in_utterance):
            if utterance[i] not in self.words_seen:
                self.words_seen[utterance[i]] = 1
            else:
                self.words_seen[utterance[i]] += 1
            word_index = self.word_mapping[utterance[i]]
            for j in range(num_refs_in_scene):
                ref_vector = self.get_referent_vector(scene[j])
                word_vector = self.meaning_dist[word_index, :]
                sim = self.get_cosine_similarity(word_vector, ref_vector)
                word_ref_matrix[i, j] = sim
        normalized_matrix = word_ref_matrix / np.sum(word_ref_matrix, axis=1).reshape(-1, 1)
        for i in range(num_words_in_utterance):
            word_index = self.word_mapping[utterance[i]]
            for j in range(num_refs_in_scene):
                referent = scene[j]
                for feature in referent:
                    feature_index = self.feature_mapping[feature]
                    if feature not in alignment_scores[word_index]:
                        alignment_scores[word_index][feature_index] = [normalized_matrix[i, j]]
                    else:
                        alignment_scores[word_index][feature_index].append(normalized_matrix[i, j])
        return alignment_scores, words_in_utterance


    def get_alignment_scores_word_comp(self, utterance, scene):
        num_words_in_utterance = len(utterance)
        num_refs_in_scene = len(scene)
        word_ref_matrix = np.zeros((num_words_in_utterance, num_refs_in_scene))

        alignment_scores = {self.word_mapping[word]: {} for word in utterance}
        # Only normalize over words in utterance, not every word in the vocabulary
        words_in_utterance = np.array([self.word_mapping[word] for word in utterance])
        for i in range(num_words_in_utterance):
            if utterance[i] not in self.words_seen:
                self.words_seen[utterance[i]] = 1
            else:
                self.words_seen[utterance[i]] += 1
            word_index = self.word_mapping[utterance[i]]
            for j in range(num_refs_in_scene):
                ref_vector = self.get_referent_vector(scene[j])
                word_vector = self.meaning_dist[word_index, :]
                # Sim(word,ref) should be the alignment score between them
                sim = self.get_cosine_similarity(word_vector, ref_vector)
                word_ref_matrix[i, j] = sim
        normalized_matrix = word_ref_matrix / np.sum(word_ref_matrix, axis=0).reshape(1, -1)
        for i in range(num_words_in_utterance):
            word_index = self.word_mapping[utterance[i]]
            for j in range(num_refs_in_scene):
                referent = scene[j]
                for feature in referent:
                    feature_index = self.feature_mapping[feature]
                    if feature not in alignment_scores[word_index]:
                        alignment_scores[word_index][feature_index] = [normalized_matrix[i, j]]
                    else:
                        alignment_scores[word_index][feature_index].append(normalized_matrix[i, j])
        return alignment_scores, words_in_utterance


    def update_association_scores(self, alignment_scores):
#         self.show_association_scores()
        for word in alignment_scores:
            feature_scores = alignment_scores[word]
            for feature in feature_scores:
                if not self.referents:
                    self.association_scores[word, feature] = self.association_scores[word, feature] + feature_scores[feature]
                else:
                    self.association_scores[word, feature] = self.association_scores[word, feature] + np.max(feature_scores[feature])

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
            word_feature_scores = self.association_scores[word, np.nonzero(self.association_scores[word, :])]
#             denominator = np.sum(word_feature_scores) + self.num_features*self.smoothing_param
            for feature in feature_scores:
                numerator = self.association_scores[word, feature]
                self.meaning_dist[word, feature] = numerator
            self.meaning_dist[word] += self.smoothing_param
        self.meaning_dist[words_in_utterance] = np.divide(self.meaning_dist[words_in_utterance], np.sum(self.meaning_dist[words_in_utterance], axis=1).reshape(-1, 1))
#         print("======== UPDATED MEANING PROBABILITIES ===========")
        return
