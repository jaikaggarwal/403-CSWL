import numpy as np
import itertools
import matplotlib.pyplot as plt
from learning08 import Learner
from evaluation08 import Evaluator
from preproc08 import get_data

def main(file_name, label, uncertainty=False, freq=0, max_iters=12001):
    u, s = get_data(file_name, uncertainty)
    word_learner = Learner(u, s)
    e = Evaluator('gold_lexicon.txt')
    prop_words_learned = []
    times = []
    t = 1
    print(label)
    for utt, scene in zip(u, s):
        word_learner.train_on_pair(utt, scene)
        if t % 50 == 0:
            prop_learned = e.get_proportion_words_learned(word_learner, freq)
            times.append(t)
            prop_words_learned.append(prop_learned)
#             print(t)
        t += 1
        if t == max_iters:
            break
    plt.plot(times, prop_words_learned, label=label)
    plt.xlabel("Time")
    plt.ylabel("Proportion of learned words")
    plt.title("Proportion of Words Learned over Time")
    plt.ylim(0, 1)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.savefig('TEST_learned_{}_{}.png'.format(uncertainty, freq))


def main_fast_map(file_name, label, uncertainty=False, freq=0, max_iters=12001):
    u, s = get_data(file_name, uncertainty)
    word_learner = Learner(u, s)
    e = Evaluator('gold_lexicon.txt')
    prop_words_learned = []
    times = []
    t = 1
    print(label)
    for utt, scene in zip(u, s):
        word_learner.train_on_pair(utt, scene, t)
        if t % 50 == 0:
            prop_learned = e.view_fast_map_behaviour(word_learner, freq)
            times.append(t)
            prop_words_learned.append(prop_learned)
            print(t)
        t += 1
        if t == max_iters:
            break

    time_of_exposure = []
    num_usages = []
    for word in e.first_learned:
        time_of_exposure.append(word_learner.first_seen[word])
        num_usages.append(e.first_learned[word])

    plt.scatter(time_of_exposure, num_usages)
    plt.xlabel("Time of first exposure")
    plt.ylabel("Number of Usages Needed to Learn")
    plt.title("Fast Mapping Behaviour in {} Condition".format(label))
    plt.ylim(0, 20)
    plt.yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * 20)
    plt.xlim(0, 12000)
    #     plt.legend()
    plt.savefig('fast_map_{}_{}.png'.format(uncertainty, freq))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main('childes_data.txt', 'no RU')
    main('childes_data.txt', 'RU (all)', True)
    main('childes_data.txt', 'RU (f>=2) ', True, 2)
    main('childes_data.txt', 'RU (f>=3) ', True, 3)
    main('childes_data.txt', 'RU (f>=5) ', True, 5)
    main_fast_map('childes_data.txt', 'no RU')
    main_fast_map('childes_data.txt', 'RU (all)', True)