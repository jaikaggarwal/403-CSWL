import numpy as np
import itertools
import time


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


def get_data(file_name, referents=False):
    """ Gets scene and utterance pairs from the given data file. If referents is True, then we construct referent sets
    instead of lists."""
    with open(file_name, 'r') as file:
        data = file.readlines()
        data = [line.strip() for line in data if line != '1-----\n' and line != '1-----']
        scene_mapping = utt_to_scene('gold_lexicon.txt')
        all_utterances = [e.split(" ")[1:] for e in data[::2]]
#         print(all_utterances)
        scenes = []
        for utt in all_utterances:
            scene = [scene_mapping[word] for word in utt]
            if referents:
                scenes.append(scene)
            else:
                scene = list(itertools.chain.from_iterable(scene))
                scenes.append(scene)
        return all_utterances, scenes


def get_data_split(file_name, referents=False):
    """ Gets scene and utterance pairs from the given data file. If referents is True, then we construct referent sets
    instead of lists."""
    with open(file_name, 'r') as file:
        data = file.readlines()
        data = [line.strip() for line in data if line != '1-----\n' and line != '1-----']
        scene_mapping = utt_to_scene('gold_lexicon.txt')
        all_utterances = [e.split(" ")[1:] for e in data[::2]]
        short_utterances = []
        long_utterances = []
        short_scenes = []
        long_scenes = []
        for utt in all_utterances:
            if len(utt) >= 5:
                long_utterances.append(utt)
                scene = [scene_mapping[word] for word in utt]
                if referents:
                    long_scenes.append(scene)
                else:
                    scene = list(itertools.chain.from_iterable(scene))
                    long_scenes.append(scene)
            elif len(utt) <= 3:
                short_utterances.append(utt)
                scene = [scene_mapping[word] for word in utt]
                if referents:
                    short_scenes.append(scene)
                else:
                    scene = list(itertools.chain.from_iterable(scene))
                    short_scenes.append(scene)
        return short_utterances, long_utterances, short_scenes, long_scenes
