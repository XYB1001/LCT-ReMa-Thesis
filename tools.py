from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix
import statistics as stats

def customised_PRF(Ygold, Yguess):
    '''
    Implementation of the cutomised PRF for ordinal data. Much more 'lenient' than traditional PRF, system also rewarded for misclassification when it's a class close to the gold class on the scale
    Type of both params: list(str)
    '''

    # _IDEA: looping through each Ygold-Yguess pair, adding to the counts list of the class(es) concerned

    TP, FP, FN = 0,1,2 # for ease of interpretation
    # True Negatives won't be needed! They don't play any role in calculating precision and recall!
    assert len(Ygold) == len(Yguess), 'Unequal length between Ygold and Yguess!'

    # each class is to have its own counts list, consisting of TP, FP, FN, initially 0
    counts = defaultdict(lambda:[0,0,0])

    for idx in range(len(Ygold)):
        l_gold = Ygold[idx]
        l_guess = Yguess[idx]

        true_ness = get_true_ness(l_gold, l_guess)
        false_ness = 1 - true_ness

        # add true_ness value to TP of l_guess -- To what extent do we correctly say that it's l_label
        counts[l_guess][TP] += true_ness
        # add false_ness value to FP of l_guess -- To what extent do we falsely say that it's l_label
        counts[l_guess][FP] += false_ness
        # add false_ness value to FN of l_gold -- To what extent do we falsely say that it's NOT l_gold
        counts[l_gold][FN] += false_ness

    # counts is now populated as: {'a1':(TP, FP, FN), 'a2':(TP, FP, FN), ... 'c2':(TP, FP, FN)}
    # Calculate Precision, Recall, F1 for each class! Need to handle zero division

    metrics = []
    for cl in sorted(counts.items()):
        # each cl as ('a1', (23,4,12))
        cl_name = cl[0]
        counted = cl[1]
        # Precision
        prec = counted[TP] / (counted[TP] + counted[FP]) if (counted[TP] + counted[FP]) != 0 else 0.0
        # Recall
        rec = counted[TP] / (counted[TP] + counted[FN]) if (counted[TP] + counted[FN]) != 0 else 0.0
        # F1
        f1 = 2 * (prec * rec) / (prec + rec) if prec + rec != 0.0 else 0.0

        metrics.append((cl_name, [prec, rec, f1]))

    return metrics


def get_true_ness(label_gold, label_guess):
    '''
    Finds true_ness score, given two CEFR labels
    Heuristics:
    If the two labels identical: 1.0 true;
    For labels next to each other on the scale: 0.6 true;
    For labels with one in between: 0.3 true;
    If further apart, 0.0 true
    '''

    # first, map to integers
    mapping = {'a1':1, 'a2':2,'b1':3,'b2':4, 'c1':5,'c2':6}

    # get true_ness value according to heuristics adopted
    try:
        diff = abs(mapping[label_gold] - mapping[label_guess])
    except KeyError as e:
        e.args = ('Unrecognised class label: ' + e.args[0] + '!',)
        raise

    if diff == 0:
        true_ness = 1.0
    elif diff == 1:
        true_ness = 0.6
    elif diff == 2:
        true_ness = 0.3
    elif diff > 2:
        true_ness = 0.0
    else:
        raise ValueError('Error when getting true_ness score!')

    return true_ness

def customised_evaluate(Ygold, Yguess):
    '''
    This just mainly prints out the metrics computed by customised_PRF in a pretty way
    '''

    PRF_by_class = customised_PRF(Ygold, Yguess) # This is [('a1', [0.233, 0.412, 0.02]) ... ('c2', [0.233, 0.412, 0.02])]
    print('-'*50)
    print('Accuracy (unaffected by customisation):', accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F1-score per class:")
    print('{:5s} {:>10s} {:>10s} {:>10s}'.format('', 'Precision', 'Recall', 'F1'))
    for cl_name, metrics in PRF_by_class:
        print('{:5s} {:10.3f} {:10.3f} {:10.3f}'.format(cl_name, metrics[0], metrics[1], metrics[2]))
    print('-'*50)

    # Calculate F1 macro average
    F1s = [tup[1][2] for tup in PRF_by_class]
    print('Average (macro) F1-score: {:.3f}'.format(stats.mean(F1s)))



if __name__ == '__main__':
    print(get_true_ness('a1','a2'))
    print(get_true_ness('b2', 'b2'))
    print(get_true_ness('c1', 'a2'))
    print(get_true_ness('b1', 'c1'))
    print(get_true_ness('a2', 'c2'))


    Yguess = ['a2','b1','c2','c1','b1','b2','a1','a1','a2','c2']
    Ytest = ['a2','b2','b2','b2','c1','b2','a2','c1','c1','c2']

    customised_evaluate(Ytest, Yguess)
    print(confusion_matrix(Ytest, Yguess))
    # metrics = customised_PRF(Ytest, Yguess)
    # print(metrics)
    # for item in metrics:
    #     print(item)
    print('Test finished')







#### space ######
