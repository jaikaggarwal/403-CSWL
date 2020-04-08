import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
import evaluation
import learning
import preproc


def main_more(file_name, label, referents=False, competition=None, freq=0, max_iters=10001):
    u, s = get_data(file_name, referents)
#     print(s)
    word_learner = Learner(u, s, referents, competition)
    e = Evaluator('gold_lexicon.txt')
    prop_words_learned = []
    times = []
    t = 1
    print(label)
    for utt, scene in zip(u, s):
        word_learner.train_on_pair(utt, scene)

        t += 1
        if t == max_iters:
            break

    return e.get_avg_acq_freq_more(word_learner, referents, freq)


def main_less(file_name, label, referents=False, competition=None, freq=0, max_iters=10001):
    u, s = get_data(file_name, referents)
#     print(s)
    word_learner = Learner(u, s, referents, competition)
    e = Evaluator('gold_lexicon.txt')
    prop_words_learned = []
    times = []
    t = 1
    print(label)
    for utt, scene in zip(u, s):
        word_learner.train_on_pair(utt, scene)

        t += 1
        if t == max_iters:
            break

    return e.get_avg_acq_freq_less(word_learner, referents, freq)


def main_utt_split(file_name, label, referents=False, competition=None, freq=0, max_iters=10001):
    print(label)
    short_u, long_u, short_s, long_s = get_data_split(file_name, referents)
#     print('short utt:', len(short_u))
#     print('long utt:', len(long_u))
    #     print(s)
    e = Evaluator('gold_lexicon.txt')

    short_word_learner = Learner(short_u, short_s, referents, competition)
    print('short split learned')
    t = 1
    for utt, scene in zip(short_u, short_s):
        short_word_learner.train_on_pair(utt, scene)

        t += 1
        if t == max_iters:
            break

    long_word_learner = Learner(long_u, long_s, referents, competition)
    print('long split learned')
    t = 1
    for utt, scene in zip(long_u, long_s):
        long_word_learner.train_on_pair(utt, scene)

        t += 1
        if t == max_iters:
            break

    return e.get_avg_acq_freq_more(short_word_learner, referents, freq), e.get_avg_acq_freq_more(long_word_learner, referents, freq)


def main_utt_freq(file_name, label, referents=False, competition=None, freq1ow=5, freqhigh=10, max_iters=10001):
    print(label)
    short_u, long_u, short_s, long_s = get_data_split(file_name, referents)
#     print('short utt:', len(short_u))
#     print('long utt:', len(long_u))
    #     print(s)
    e = Evaluator('gold_lexicon.txt')

    short_word_learner = Learner(short_u, short_s, referents, competition)
    print('short split learned')
    t = 1
    for utt, scene in zip(short_u, short_s):
        short_word_learner.train_on_pair(utt, scene)

        t += 1
        if t == max_iters:
            break

    long_word_learner = Learner(long_u, long_s, referents, competition)
    print('long split learned')
    t = 1
    for utt, scene in zip(long_u, long_s):
        long_word_learner.train_on_pair(utt, scene)

        t += 1
        if t == max_iters:
            break

    return e.get_avg_acq_freq_more(short_word_learner, referents, freqhigh), e.get_avg_acq_freq_more(long_word_learner, referents, freqhigh), e.get_avg_acq_freq_less(short_word_learner, referents, freq1ow), e.get_avg_acq_freq_less(long_word_learner, referents, freq1ow)


def main(file_name, label, referents=False, competition=None, freq=0, max_iters=10001):
    u, s = get_data(file_name, referents)
#     print(s)
    word_learner = Learner(u, s, referents, competition)
    e = Evaluator('gold_lexicon.txt')
    prop_words_learned = []
    times = []
    t = 1
    print(label)
    for utt, scene in zip(u, s):
        word_learner.train_on_pair(utt, scene)
        if t % 50 == 0:
            prop_learned = e.get_proportion_words_learned(word_learner, referents, freq)
            times.append(t)
            prop_words_learned.append(prop_learned)
        t += 1
#         if t % 100 == 0:
#             print(t)
        if t == max_iters:
            break
    plt.plot(times, prop_words_learned, label=label)
    plt.xlabel("Time")
    plt.ylabel("Proportion of learned words")
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.title("Proportion of Words Learned over Time")
    plt.legend()


if __name__ == "__main__":
    overall_analysis = False
    # Notice freq and utt analyses are done together
    freq_analysis = False
    utt_analysis = False
    cross_analysis = False

    # Overall learning analysis with competitions
    if overall_analysis:
        main('childes_data.txt', 'FAS', False)
        main('childes_data.txt', 'no-comp', True)
        main('childes_data.txt', 'ref-comp', True, 'ref_comp')
        main('childes_data.txt', 'word-comp', True, 'word_comp')

    # Learning for split frequency analysis
    if freq_analysis:
        word_freq_high = []
        word_freq_less = []

        word_freq_high.append(main_more('childes_data.txt', 'FAS', False, freq=10))
        word_freq_high.append(main_more('childes_data.txt', 'no-comp', True, freq=10))
        word_freq_high.append(main_more('childes_data.txt', 'ref-comp', True, 'ref_comp', freq=10))
        word_freq_high.append(main_more('childes_data.txt', 'word-comp', True, 'word_comp', freq=10))

        word_freq_less.append(main_less('childes_data.txt', 'FAS', False, freq=5))
        word_freq_less.append(main_less('childes_data.txt', 'no-comp', True, freq=5))
        word_freq_less.append(main_less('childes_data.txt', 'ref-comp', True, 'ref_comp', freq=5))
        word_freq_less.append(main_less('childes_data.txt', 'word-comp', True, 'word_comp', freq=5))

    # Learning for split utterance analysis
    if utt_analysis:
        fas_short, fas_long = main_utt_split('childes_data.txt', 'FAS', False)
        no_comp_short, no_comp_long = main_utt_split('childes_data.txt', 'no-comp', True)
        ref_comp_short, ref_comp_long = main_utt_split('childes_data.txt', 'ref-comp', True, 'ref_comp')
        word_comp_short, word_comp_long = main_utt_split('childes_data.txt', 'word-comp', True, 'word_comp')

        utt_split_short = [fas_short, no_comp_short, ref_comp_short, word_comp_short]
        utt_split_long = [fas_long, no_comp_long, ref_comp_long, word_comp_long]

    # Output results
    if freq_analysis and utt_analysis:
        objects = ('FAS', 'no-comp', 'ref-comp', 'word-comp')

        # data to plot
        n_groups = 2
        means_frank = utt_split_long
        means_guido = utt_split_short

        # create plot
        fig, axs = plt.subplots(1,2,figsize=(10,5))
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 0.8

        # plt.subplot(121)
        fas = axs[1].bar(index, (utt_split_long[0], utt_split_short[0]), bar_width,
                      alpha=opacity,
                      label='FAS')

        no_comp = axs[1].bar(index+bar_width, (utt_split_long[1], utt_split_short[1]), bar_width,
                      alpha=opacity,
                      label='no-comp')

        ref_comp = axs[1].bar(index+2*bar_width, (utt_split_long[2], utt_split_short[2]), bar_width,
                      alpha=opacity,
                      label='ref-comp')

        word_comp = axs[1].bar(index+3*bar_width, (utt_split_long[3], utt_split_short[3]), bar_width,
                      alpha=opacity,
                      label='word-comp')

        axs[1].set_xlabel('mean length of utterance')
        axs[1].set_ylabel('average acquisition score')
        axs[1].set_title('(b) Split over utterance')
        axs[1].set_xticks(index + bar_width, ('long', 'short'))
        axs[1].legend()

        #######

        objects = ('FAS', 'no-comp', 'ref-comp', 'word-comp')

        # data to plot
        n_groups = 2
        means_frank = word_freq_high
        means_guido = word_freq_less

        # create plot
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 0.8

        fas = axs[0].bar(index, (word_freq_high[0], word_freq_less[0]), bar_width,
                      alpha=opacity,
                      label='FAS')

        no_comp = axs[0].bar(index+bar_width, (word_freq_high[1], word_freq_less[1]), bar_width,
                      alpha=opacity,
                      label='no-comp')

        ref_comp = axs[0].bar(index+2*bar_width, (word_freq_high[2], word_freq_less[2]), bar_width,
                      alpha=opacity,
                      label='ref-comp')

        word_comp = axs[0].bar(index+3*bar_width, (word_freq_high[3], word_freq_less[3]), bar_width,
                      alpha=opacity,
                      label='word-comp')

        axs[0].set_xlabel('frequency')
        axs[0].set_ylabel('average acquisition score')
        axs[0].set_title('(a) Split over frequency')
        axs[0].set_xticks(index + bar_width, ('high', 'low'))
        axs[0].legend()

        plt.savefig('combined_split.png')
        plt.show()

    # Cross analysis for freq and utterance length
    if cross_analysis:
        fas_short_high, fas_long_high, fas_short_low, fas_long_low = main_utt_freq('childes_data.txt', 'FAS', False)
        no_comp_short_high, no_comp_long_high, no_comp_short_low, no_comp_long_low = main_utt_freq('childes_data.txt', 'no-comp', True)
        ref_comp_short_high, ref_comp_long_high, ref_comp_short_low, ref_comp_long_low = main_utt_freq('childes_data.txt', 'ref-comp', True, 'ref_comp')
        word_comp_short_high, word_comp_long_high, word_comp_short_low, word_comp_long_low = main_utt_freq('childes_data.txt', 'word-comp', True, 'word_comp')

        high_short = [fas_short_high, no_comp_short_high, ref_comp_short_high, word_comp_short_high]
        high_long = [fas_long_high, no_comp_long_high, ref_comp_long_high, word_comp_long_high]
        low_short = [fas_short_low, no_comp_short_low, ref_comp_short_low, word_comp_short_low]
        low_long = [fas_long_low, no_comp_long_low, ref_comp_long_low, word_comp_long_low]

        objects = ('FAS', 'no-comp', 'ref-comp', 'word-comp')
        # data to plot
        n_groups = 4
        means_frank = utt_split_long
        means_guido = utt_split_short

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.8

        fas = plt.bar(index, (high_long[0], high_short[0], low_long[0], low_short[0]), bar_width,
                      alpha=opacity,
                      label='FAS')

        no_comp = plt.bar(index+bar_width, (high_long[1], high_short[1], low_long[1], low_short[1]), bar_width,
                      alpha=opacity,
                      label='no-comp')

        ref_comp = plt.bar(index+2*bar_width, (high_long[2], high_short[2], low_long[2], low_short[2]), bar_width,
                      alpha=opacity,
                      label='ref-comp')

        word_comp = plt.bar(index+3*bar_width, (high_long[3], high_short[3], low_long[3], low_short[3]), bar_width,
                      alpha=opacity,
                      label='word-comp')

        plt.xlabel('frequency X mean length of utterance')
        plt.ylabel('average acquisition score')
        plt.title('Split over MLU and word frequency')
        plt.xticks(index + bar_width, ('high-long', 'high-short', 'low-long', 'low-short'))
        plt.legend()
        plt.savefig('avg_acq_freq_x_utt.png')

        plt.tight_layout()
        plt.show()
