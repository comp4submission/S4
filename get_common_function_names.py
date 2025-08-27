import glob
import pickle
import networkx as nx

from tqdm import tqdm
from collections import Counter
from utils import InternalMethod, Instruction


def main():
    pkl_paths = glob.glob('/mnt/ssd1/anonymous/wo_debug/x86_64/**/*.fcg', recursive=True)
    print('# of files:', len(pkl_paths))

    all_func_names = []
    for pkl in tqdm(pkl_paths):
        fcg: nx.DiGraph = pickle.load(open(pkl, 'rb'))
        for m in nx.get_node_attributes(fcg, 'data').values():
            m: InternalMethod
            all_func_names.append(m.get_name())

    print(Counter(all_func_names).most_common(30))
    pickle.dump(all_func_names, open('all_func_names.pkl', 'wb'))


if __name__ == '__main__':
    main()
