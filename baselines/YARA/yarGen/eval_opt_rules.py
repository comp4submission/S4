import os
import yara
from contextlib import suppress

TESTSET_PATH = '/dev/shm/split_dataset/test_for_yara_opt'
RULES_PATH = 'opt_rules'


def evaluate(rule, cnt_opt):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for opt in os.listdir(TESTSET_PATH):
        for sample in os.listdir(os.path.join(TESTSET_PATH, opt)):
            if opt == cnt_opt:
                # rule.match() returns a list of matched rules for the given file
                if len(rule.match(os.path.join(TESTSET_PATH, opt, sample))) > 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if len(rule.match(os.path.join(TESTSET_PATH, opt, sample))) > 0:
                    FP += 1
                else:
                    TN += 1

    return TP, FP, TN, FN


def main():
    Ps = []
    Rs = []
    As = []
    Fs = []

    for opt_rule in os.listdir(RULES_PATH):

        if opt_rule.endswith('.yar'):
            opt = opt_rule[:-len('.yar')]
        else:
            opt = opt_rule

        rule = yara.compile(os.path.join(RULES_PATH, opt_rule))
        TP, FP, TN, FN = evaluate(rule, opt)

        P = R = A = F1 = 0
        with suppress(ZeroDivisionError):
            P = TP / (TP + FP)
        with suppress(ZeroDivisionError):
            R = TP / (TP + FN)
        with suppress(ZeroDivisionError):
            A = (TP + TN) / (TP + FP + TN + FN)
        with suppress(ZeroDivisionError):
            F1 = 2 * P * R / (P + R)

        Ps.append(P)
        Rs.append(R)
        As.append(A)
        Fs.append(F1)
        print(f'opt: {opt}, P: {P}, R: {R}, A: {A}, F1: {F1}')

    print(f'Average - P: {sum(Ps) / len(Ps)}, R: {sum(Rs) / len(Rs)}, A: {sum(As) / len(As)}, F1: {sum(Fs) / len(Fs)}')


if __name__ == '__main__':
    main()
