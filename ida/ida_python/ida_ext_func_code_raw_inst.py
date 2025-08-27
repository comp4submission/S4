import os
import site

os.environ["PYTHONPATH"] = os.pathsep.join(site.getsitepackages() + [site.getusersitepackages()]) + os.pathsep + os.environ.get("PYTHONPATH", "")

import idc
import sark
import pickle
import idaapi
import ida_pro

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

methods = {}

for func in text_segment.functions:
    func_instructions = []
    for line in func.lines:
        operands = [operand.text for operand in line.insn.operands if operand.text != '']

        regs = []
        for operand in line.insn.operands:
            regs.extend(sorted(operand.regs))

        func_instructions.append(Instruction(address=line.ea, size=idc.get_item_size(ea=line.ea), opcode=line.insn.mnem, operands=operands, regs=regs, is_call=line.insn.is_call, is_ret=line.insn.is_ret, is_indirect_jump=line.insn.is_indirect_jump))

    m = InternalMethod(name=func.name, start_addr=func.ea, instructions=func_instructions, pseudocode=get_pseudocode(func.ea))
    methods[func.name] = m

with open('{}.code'.format(outputFileName), 'wb') as f:
    pickle.dump(methods, f, protocol=2)

ida_pro.qexit(0)

