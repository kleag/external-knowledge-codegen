from .parser import Parser

from typing import (Dict, List, Set, Tuple)


def parse(cpp_code: str = None, filepath: str = None,
          compile_command: Dict[str, str] = None, debug: bool = False):
    parser = Parser(cpp_code, filepath, compile_command)
    parser.set_debug(debug)
    return parser.parse()


def parse_member_declaration(s: str = None, filepath: str = None,
                             compile_command: Dict[str, str] = None,
                             debug: bool = False):
    parser = Parser(s, filepath, compile_command)
    parser.set_debug(debug)
    return parser.parse_member_declaration()
