import random
import networkx as nx
from .serialization import pickle_load
from .ida_class import Instruction, InternalMethod


def sample_element_by_order(l):
    """
    Sample an element from a list based on the order of each element
    The selection probability is proportional to the order of the element.
    The earlier the element, the easier it is to be selected.
    The selected element will be put at the end of the list.
    """
    temp_index = []
    for i in range(len(l)):
        temp_index.extend([i] * (len(l) - i))

    selected_index = random.choice(temp_index)

    return l[:selected_index] + l[selected_index+1:] + l[selected_index:selected_index+1]


def sample_instructions_from_fcg(pkl_path, threshold=10000) -> list[Instruction]:
    fcg: nx.DiGraph = pickle_load(pkl_path)

    n_dc_map = nx.degree_centrality(fcg)

    # # sort nodes by degree centrality from high to low
    # sorted_nodes = [n for n, dc in sorted(n_dc_map.items(), key=lambda x: x[1], reverse=True)]

    # sort nodes by degree centrality from low to high
    sorted_nodes = [n for n, dc in sorted(n_dc_map.items(), key=lambda x: x[1])]

    # get the total number of instructions
    total_instructions = sum([len(fcg.nodes[n]['data'].get_instructions()) for n in sorted_nodes])
    if threshold == 0:
        threshold = int(total_instructions * 0.7)

    # excludes = [
    #     'deregister_tm_clones',
    #     'register_tm_clones',
    #     '__do_global_dtors_aux',
    #     '__do_global_ctors_aux',
    #     '_start', 'frame_dummy',
    #     '__libc_csu_init',
    #     '__libc_csu_fini'
    #     ]
    excludes = []

    # # locate the entry point (i.e., start function)
    # entry_point = None
    #
    # for addr, is_ep in nx.get_node_attributes(fcg, 'is_ep').items():
    #     if is_ep is True:
    #         entry_point = addr
    #         break

    # if entry_point is not None:
    #     sampled_instruction_list: list[Instruction] = explore(fcg, entry_point)
    # else:
    #     sampled_instruction_list: list[Instruction] = []

    sampled_instruction_list: list[Instruction] = []

    while True:
        sorted_nodes = sample_element_by_order(sorted_nodes)
        sampled_node = sorted_nodes[-1]
        if fcg.nodes[sampled_node]['data'].get_name() in excludes:
            continue

        sampled_instruction_list.extend(explore(fcg, sampled_node))

        if len(sampled_instruction_list) > threshold:
            return sampled_instruction_list


def random_sample_instructions_from_fcg(pkl_path, threshold=10000) -> list[Instruction]:
    fcg: nx.DiGraph = pickle_load(pkl_path)

    sampled_instruction_list: list[Instruction] = []

    nodes = list(fcg.nodes)

    while True:
        sampled_node = random.choice(nodes)
        sampled_instruction_list.extend(fcg.nodes[sampled_node]['data'].get_instructions())

        if len(sampled_instruction_list) > threshold:
            return sampled_instruction_list


def explore(g: nx.DiGraph, start_node: int) -> list[Instruction]:
    paths: list[int] = []
    instruction_list: list[Instruction] = []

    def get_successors(n):
        _successors = list(g.successors(n))

        for old_n in paths:
            if old_n in _successors:
                _successors.remove(old_n)

        return _successors

    # start explore
    cnt_node = start_node
    m: InternalMethod = g.nodes[cnt_node]['data']
    instruction_list.extend(m.get_instructions())
    paths.append(cnt_node)

    while True:
        successors = get_successors(cnt_node)
        if len(successors) == 0:
            break

        cnt_node = random.choice(successors)
        m: InternalMethod = g.nodes[cnt_node]['data']
        instruction_list.extend(m.get_instructions())
        paths.append(cnt_node)

    return instruction_list
