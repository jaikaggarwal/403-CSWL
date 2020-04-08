import numpy as np
import itertools
import matplotlib.pyplot as plt


def utt_to_scene(file_name):
    """ The purpose of this function is to parse the gold standard lexicon to get the appropriate feature mappings
    for a given word. The return value is a dictionary, where the key is the word and the value is a list of features
    it maps to."""
    with open(file_name, 'r') as file:
        data = file.readlines()
        data = [line.strip().split() for line in data if line.strip() != '']
        data = [[line[0], " ".join(line[1:])] for line in data]
        preproc_data = [[line[0], list(map(lambda x: x[:x.find(":")], line[1].split(',')))[:-1]] for line in data]
        scene_mapping = {line[0]: line[1] for line in preproc_data}
        return scene_mapping

def get_data(file_name, uncertainty=False):
    """ Gets scene and utterance pairs from the given data file. If uncertainty is True, then every other utterance is skipped.
    For the remaining utterances, two consecutive scenes are paired together as the scene for one utterance. This
    simulates referential uncertainty."""
    with open(file_name, 'r') as file:
        data = file.readlines()
        data = [line.strip() for line in data if line != '1-----\n' and line != '1-----']
        scene_mapping = utt_to_scene('gold_lexicon.txt')
        all_utterances = [e.split(" ")[1:] for e in data[::2]]
#         print(all_utterances)
        if uncertainty: # simulates the presence of referential uncertainty
            new_utterances = []
            scenes = []
            # Add first to utterances, and use scene representation from both first and second to construct the scene representation for first
            for first, second in zip(all_utterances[::2], all_utterances[1::2]):
                # Only want every other sentence
                new_utterances.append(first + ['UNK'])
                scene = [scene_mapping[word] for word in first + second]
                scene = list(itertools.chain.from_iterable(scene))
                scenes.append(scene)
            return new_utterances, scenes
        else:
            scenes = []
            for utt in all_utterances:
                scene = [scene_mapping[word] for word in utt]
                scene = list(itertools.chain.from_iterable(scene))
                scenes.append(scene)
            return all_utterances, scenes