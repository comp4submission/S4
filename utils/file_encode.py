import chardet


def get_charset(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        charset = chardet.detect(f.read())['encoding']
    return charset
