import os
import site

os.environ["PYTHONPATH"] = os.pathsep.join(
    site.getsitepackages() + [site.getusersitepackages()]) + os.pathsep + os.environ.get("PYTHONPATH", "")

import idc
import sark
import pickle
import idaapi
import ida_pro
import itertools
import networkx as nx

from sark import exceptions
from contextlib import suppress


class Instruction:
    def __init__(self, address: int, size: int, opcode: str, operands: list[str], regs: list[str],
                 is_call: bool = False,
                 is_ret: bool = False, is_indirect_jump: bool = False):
        self.__address: int = address
        self.__size: int = size
        self.__opcode: str = opcode
        self.__operands: list[str] = operands
        self.__regs: list[str] = regs
        self.__is_call: bool = is_call
        self.__is_ret: bool = is_ret
        self.__is_indirect_jump: bool = is_indirect_jump

    def get_address(self) -> int:
        return self.__address

    def get_size(self) -> int:
        return self.__size

    def get_opcode(self) -> str:
        return self.__opcode

    def get_operands(self) -> list[str]:
        return self.__operands

    def get_regs(self) -> list[str]:
        return self.__regs

    def is_call(self) -> bool:
        return self.__is_call

    def is_ret(self) -> bool:
        return self.__is_ret

    def is_indirect_jump(self) -> bool:
        return self.__is_indirect_jump

    def __str__(self) -> str:
        if len(self.__operands) == 0:
            literal = self.__opcode
        else:
            literal = f'{self.__opcode} {", ".join(self.__operands)}'

        return 'Instruction.{}'.format(literal)

    def __repr__(self) -> str:
        if len(self.__operands) == 0:
            literal = self.__opcode
        else:
            literal = f'{self.__opcode} {", ".join(self.__operands)}'

        return 'Instruction.{}'.format(literal)


class BasicBlock:
    def __init__(self, address: int, instructions: list[Instruction]):
        self.__address: int = address
        self.__instructions: list[Instruction] = instructions

    def get_instructions(self) -> list[Instruction]:
        return self.__instructions

    def get_address(self) -> int:
        return self.__address

    def __str__(self):
        return 'BasicBlock.{}'.format(self.__address)

    def __repr__(self):
        return 'BasicBlock.{}'.format(self.__address)


class InternalMethod:
    def __init__(self, name: str, start_addr: int, instructions: list[Instruction], pseudocode: str = None):
        self.__name: str = name
        self.__start_addr: int = start_addr
        self.__instructions: list[Instruction] = instructions
        self.__pseudocode: str = pseudocode

    def get_name(self) -> str:
        return self.__name

    def get_start_addr(self) -> int:
        return self.__start_addr

    def get_instructions(self) -> list[Instruction]:
        return self.__instructions

    def get_pseudocode(self) -> str:
        return self.__pseudocode

    def __str__(self):
        return 'InternalMethod.{}'.format(self.__name)

    def __repr__(self):
        return 'InternalMethod.{}'.format(self.__name)


def filter_reg(op):
    # return op.text
    return '<REG>'


def filter_imm(op):
    return '<NUM>'


def filter_disp_in_displacement(displacement):
    return '<NUM>'


def filter_mem(op):
    return '<NUM>'


def filter_phrase(op):
    # return '[' + op.base + ']'
    return '[<REG>]'


def filter_displacement(op):
    if op.index is None:
        # return '[{}*{}+{}]'.format(op.base, op.scale, filter_disp_in_displacement(op.displacement))
        # return '[{},<NUM>]'.format(op.base)
        return '[{}*{}+{}]'.format('<REG>', '<NUM>', filter_disp_in_displacement(op.displacement))

    else:
        # return '[{}+{}*{}+{}]'.format(op.base, op.index, op.scale, filter_disp_in_displacement(op.displacement))
        # return '[{},{},<NUM>]'.format(op.base, op.index)
        return '[{}+{}*{}+{}]'.format('<REG>', '<REG>', '<NUM>', filter_disp_in_displacement(op.displacement))


def filter_void(op):
    return '<VOID>'


def filter_special(op):
    return '<SPECIAL>'


def filter_near(op):
    return '<NUM>'


def filter_far(op):
    return '<NUM>'


def normalize_instruction(instruction, plt_funcs_name):
    inst = '' + instruction.mnem

    if instruction.is_call:

        if instruction.operands[0].type.is_reg is True:
            inst = 'indirect_call'
        else:
            callee = instruction.operands[0].text

            if callee in plt_funcs_name:
                inst = 'external_call'
            else:
                inst = 'internal_call'
        return inst

    for op in instruction.operands:
        if op.text == '':
            continue

        if op.type.is_reg:
            inst += ' ' + filter_reg(op)
        elif op.type.is_imm:
            inst += ' ' + filter_imm(op)
        elif op.type.is_mem:
            inst += ' ' + filter_mem(op)
        elif op.type.is_phrase:
            inst += ' ' + filter_phrase(op)
        elif op.type.is_displ:
            inst += ' ' + filter_displacement(op)
        elif op.type.is_near:
            inst += ' ' + filter_near(op)
        elif op.type.is_far:
            inst += ' ' + filter_far(op)
        elif op.type.is_void:
            inst += ' ' + filter_void(op)
        elif op.type.is_special:
            inst += ' ' + filter_special(op)
        else:
            inst += '<UNK>'

        if len(instruction.operands) > 1:
            inst += ','

    if inst.endswith(','):
        inst = inst[:-1]

    return inst


def get_pseudocode(ea):
    with suppress(Exception):
        # Retrieve the function object
        func = idaapi.get_func(ea)
        # Create a cfuncptr_t object to decompile the function
        cfunc = idaapi.decompile(func)
        # Get the pseudocode as a string
        pseudocode = str(cfunc)
        return pseudocode
    return None


outputFileName: str = os.path.basename(idc.get_idb_path()[:-4])

all_segments: list[str] = [seg.name for seg in sark.segments()]
if '.plt' in all_segments:
    plt_segment = sark.Segment(name='.plt')
    plt_funcs = plt_segment.functions
    # replace "." to "_"
    plt_funcs_name = [plt_func.name.replace('.', '_') for plt_func in plt_funcs]
else:
    plt_funcs_name = []

# locate text section in a robust way
text_segment = sark.Segment(name='.text')
entry_point = idc.get_inf_attr(idc.INF_START_EA)
if text_segment is None:
    for section in sark.segments():
        if section.start_ea <= entry_point < section.end_ea:
            text_segment = section
            break

if text_segment is None or len(list(text_segment.functions)) == 0:
    ida_pro.qexit(0)

# Function Call Graph
fcg = nx.DiGraph()

for func in text_segment.functions:
    print(func.ea, func.name)

    func_instructions = []
    for line in func.lines:
        instruction = normalize_instruction(line.insn, plt_funcs_name)

        # xor ebp, ebp -> ['xor', 'ebp, ebp']
        instruction_decomposition = instruction.split(' ', maxsplit=1)
        opcode: str = instruction_decomposition[0]
        operands: list[str] = []

        if len(instruction_decomposition) > 1:
            # operands exist
            operands.extend(instruction_decomposition[1].split(', '))

        regs = []
        for operand in line.insn.operands:
            regs.extend(sorted(operand.regs))

        func_instructions.append(
            Instruction(address=line.ea, size=idc.get_item_size(ea=line.ea), opcode=opcode, operands=operands,
                        regs=regs, is_call=line.insn.is_call,
                        is_ret=line.insn.is_ret, is_indirect_jump=line.insn.is_indirect_jump))

    fcg.add_node(func.ea, data=InternalMethod(name=func.name, start_addr=func.ea, instructions=func_instructions, pseudocode=get_pseudocode(func.ea)),
                 is_ep=func.ea == entry_point)

for func in text_segment.functions:
    for xref in itertools.chain(func.xrefs_from, func.xrefs_to):
        with suppress(exceptions.SarkNoFunction):
            frm = sark.Function(ea=xref.frm)
            to = sark.Function(ea=xref.to)
            if frm.ea in fcg.nodes and to.ea in fcg.nodes:
                fcg.add_edge(frm.ea, to.ea)

with open('{}.fcg'.format(outputFileName), 'wb') as f:
    pickle.dump(fcg, f, protocol=2)

ida_pro.qexit(0)
