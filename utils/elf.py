
def is_elf_file(file_path: str):
    with open(file_path, 'rb') as f:
        return f.read(4) == b'\x7fELF'
