import os
import yara
from contextlib import suppress

TESTSET_PATH = '/dev/shm/split_dataset/test_for_yara_compiler'
RULES_PATH = 'compiler_rules'


def evaluate(rule, cnt_compiler):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for compiler in os.listdir(TESTSET_PATH):
        for sample in os.listdir(os.path.join(TESTSET_PATH, compiler)):
            if compiler.replace(".", "_").replace("-", "_") == cnt_compiler:
                # rule.match() returns a list of matched rules for the given file
                if len(rule.match(os.path.join(TESTSET_PATH, compiler, sample))) > 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if len(rule.match(os.path.join(TESTSET_PATH, compiler, sample))) > 0:
                    FP += 1
                else:
                    TN += 1

    return TP, FP, TN, FN


def main():
    Ps = []
    Rs = []
    As = []
    Fs = []

    for compiler_rule in os.listdir(RULES_PATH):
        compiler = compiler_rule
        rule = yara.compile(os.path.join(RULES_PATH, compiler_rule))
        TP, FP, TN, FN = evaluate(rule, compiler)

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
        print(f'compiler: {compiler}, P: {P}, R: {R}, A: {A}, F1: {F1}')

    print(f'Average - P: {sum(Ps) / len(Ps)}, R: {sum(Rs) / len(Rs)}, A: {sum(As) / len(As)}, F1: {sum(Fs) / len(Fs)}')


if __name__ == '__main__':
    main()
