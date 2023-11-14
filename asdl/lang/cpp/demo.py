# coding=utf-8

import argparse
import colorama
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import tqdm
from multiprocessing import Pool
from typing import (Dict, List, Set, Tuple)

import cppastor
import cpplang
import istarmap
from asdl.lang.cpp.cpp_transition_system import *
from asdl.hypothesis import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)


class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, *args, **kwargs):
        kwargs['file'] = self.file
        kwargs['end'] = ''

        # I had to print each argument separately, with kwargs['end'] = ''
        for i in args:
            tqdm.write(str(i), **kwargs)

    def __eq__(self, other):
        return other is self.file

    def flush(self):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


@contextlib.contextmanager
def pushd(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)


# read in the grammar specification of Cpp SE8, defined in ASDL
asdl_text = open('cpp_asdl.simplified.txt').read()
grammar = ASDLGrammar.from_text(asdl_text)
# print(grammar)

# initialize the Cpp transition parser
parser = CppTransitionSystem(grammar)


class bcolors:
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DEFAULT = '\033[99m'


def cprint(color: bcolors, string: str, **kwargs) -> None:
    print(f"{color}{string}{bcolors.ENDC}", **kwargs)


def removeComments(string: str) -> str:
    string = re.sub(re.compile(r"^\s*#.*\n"), "", string)
    string = re.sub(re.compile(r"\s*#.*\n"), "", string)
    # remove all occurance streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile(r"/\*.*?\*/", re.DOTALL), "", string)
    # remove all occurance singleline comments (//COMMENT\n ) from string
    string = re.sub(re.compile(r"//.*"), "", string)
    return string


def code_from_hyp(asdl_ast):
    # get the sequence of gold-standard actions to construct the ASDL AST
    actions = parser.get_actions(asdl_ast)
    # a hypothesis is a(n) (partial) ASDL AST generated using a sequence of
    # tree-construction actions
    hypothesis = Hypothesis()
    for t, action in enumerate(actions, 1):
        # the type of the action should belong to one of the valid continuing
        # types of the transition system
        valid_cont_types = parser.get_valid_continuation_types(hypothesis)
        if action.__class__ not in valid_cont_types:
            logger.error(f"Error: Valid continuation types are {valid_cont_types} "
                  f"but current action class is {action.__class__} "
                  f"on frontier {hypothesis.frontier_node} / {hypothesis.frontier_field}")
            raise Exception(f"{action.__class__} is not in {valid_cont_types}")

        # if it's an ApplyRule action, the production rule should belong to the
        # set of rules with the same LHS type as the current rule
        if isinstance(action, ApplyRuleAction) and hypothesis.frontier_node:
            if action.production not in grammar[
                  hypothesis.frontier_field.type]:
                raise Exception(f"{bcolors.BLUE}{action.production}"
                                f"{bcolors.ENDC} should be in {bcolors.GREEN}"
                                f"{grammar[hypothesis.frontier_field.type]}\n"
                                f"{bcolors.ENDC} within current hypothesis "
                                f"{bcolors.CYAN} {hypothesis.frontier_node.to_string()}"
                                f"{bcolors.ENDC}")
            assert action.production in grammar[hypothesis.frontier_field.type]

        p_t = (hypothesis.frontier_node.created_time
                if hypothesis.frontier_node else -1)
        logger.debug(f't={t}, p_t={p_t}, Action={action}')
        hypothesis.apply_action(action)
    cpp_ast = asdl_ast_to_cpp_ast(hypothesis.tree, grammar)
    source = cppastor.to_source(cpp_ast)
    return source


def simplify(code: str) -> str:
    return (code.replace(" ", "")
            .replace("\t", "")
            .replace("\n", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
            .lower())


def common_prefix(str1: str, str2: str) -> None:
    common_prefix = os.path.commonprefix([str1, str2])
    percent_ok = int(float(len(common_prefix))*100/len(str1))
    print(f"Common prefix end: {common_prefix[-100:]} ({percent_ok}%)")


def preprocess_code(cpp_code):
    preprocess = subprocess.Popen(
        cpplang.parser.preprocess_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    preprocess_stdout_data, preprocess_stderr_data = preprocess.communicate(
        input=cpp_code.encode())
    preprocessed_code = preprocess_stdout_data.decode()
    lines = preprocessed_code.split('\n')
    lines_in = []
    in_stdin = False
    for i, line in enumerate(lines):
        if line.startswith("#") and "<stdin>" in line:
            in_stdin = True
        elif line.startswith("#"):
            in_stdin = False
        elif in_stdin:
            lines_in.append(line)
    return "\n".join(lines_in)


def roundtrip(cpp_code: str = None, filepath: str = None,
              compile_command: Dict[str, str] = None,
              check_hypothesis=False, fail_on_error=False, member=False,
              skip_checks=False):

    # get the (domain-specific) cpp AST of the example Cpp code snippet
    logger.debug(f'Cpp code: \n{cpp_code}')
    try:
        if member:
            cpp_ast = cpplang.parse.parse_member_declaration(
                s=cpp_code, filepath=filepath, compile_command=compile_command)
        else:
            cpp_ast = cpplang.parse.parse(s=cpp_code, filepath=filepath,
                                          compile_command=compile_command)
    except subprocess.CalledProcessError:
        # This has already been handled in the caller
        return False
    # convert the cpp AST into general-purpose ASDL AST used by tranX
    asdl_ast = cpp_ast_to_asdl_ast(cpp_ast, grammar)
    logger.debug(f'String representation of the ASDL AST:')
    logger.debug(f'{asdl_ast.to_string()}')
    logger.debug(f'Size of the AST: {asdl_ast.size}')

    # we can also convert the ASDL AST back into Cpp AST
    cpp_ast_reconstructed = asdl_ast_to_cpp_ast(asdl_ast, grammar)
    logger.debug(f'String representation of the reconstructed CPP AST:')
    logger.debug(f'{cpp_ast_reconstructed}')
    #logger.debug(f'Size of the AST: {asdl_ast.size}')

    # get the surface code snippets from the original Python AST,
    # the reconstructed AST and the AST generated using actions
    # they should be the same
    if cpp_code is None and compile_command is not None:
        with pushd(compile_command["directory"]):
            cpp_file_name = compile_command["arguments"][-1]
            with open(cpp_file_name, "r") as f:
                cpp_code = f.read()

    src0 = removeComments(preprocess_code(cpp_code))
    simp0 = simplify(src0)
    src1 = removeComments(cppastor.to_source(cpp_ast))
    simp1 = simplify(src1)
    src2 = removeComments(cppastor.to_source(cpp_ast_reconstructed))
    simp2 = simplify(src2)
    if skip_checks:
        return True

    if check_hypothesis:
        #try:
        src3 = code_from_hyp(asdl_ast)
        #except Exception as e:
            #print(f"{e}")
            #return False
        src3 = removeComments(src3)
        simp3 = simplify(src3)
    if ((not (simp1 == simp2 == simp0)) or (
               (check_hypothesis and (simp3 != simp1)))):
        if simp0 != simp1:
            cprint(bcolors.BLUE,
                   f"))))))) Original Cpp code      :\n{src0}\n(((((((\n")
            cprint(bcolors.CYAN,
                   f"}}}}}}}}}}}}}} Cpp AST                :\n{src1}\n"
                   f"{{{{{{{{{{{{{{\n")
            common_prefix(simp0, simp1)
        elif simp1 != simp2:
            cprint(bcolors.CYAN,
                   f"}}}}}}}}}}}}}} Cpp AST                :\n{src1}"
                   f"\n{{{{{{{{{{{{{{\n")
            cprint(bcolors.GREEN,
                   f"]]]]]]] Cpp AST from ASDL      :\n{src2}\n[[[[[[[\n")
            common_prefix(simp1, simp2)
        elif check_hypothesis:
            cprint(bcolors.BLUE,
                   f"))))))) Original Cpp code      :\n{src0}\n(((((((\n")
            cprint(bcolors.CYAN,
                   f"}}}}}}}}}}}}}} Cpp AST                :\n{src1}\n"
                   f"{{{{{{{{{{{{{{\n")
            cprint(bcolors.MAGENTA,
                   f">>>>>>> Cpp AST from hyp       :\n{src3}\n<<<<<<<\n")
            common_prefix(simp1, simp3)
        # if fail_on_error:
            # raise Exception("Test failed")
        # else:
        return False

    else:
        return True


cpp_code = [
    """public class Test {}""",
    """package cpplang.brewtab.com; class Test {}""",
    """class Test { String s = "bepo"; }""",
    """class Test {
        public static void main(String[] args) {}
    }""",
    """class Test {
        public static void main(String[] args) {int i = 42; ++i;}
    }""",
    """class Test {
        public static void main(String[] args) {
            System.out.println();
        }
    }""",
    """class Test {
        public static void main(String[] args) {
            for (int i = 42; i < args.length; i++)
                System.out.print(i == 666 ? args[i] : " " + args[i]);
            System.out.println();
        }
    }""",
    """package cpplang.brewtab.com;
        class Test {
        public static void main(String[] args) {
            for (int i = 42; i < args.length; i++)
                System.out.print(i == 666 ? args[i] : " " + args[i]);
            System.out.println();
        }
    }""",
    ]


def check_filepath(filepath: str,
                   check_hypothesis: bool = False,
                   skip_checks: bool = False,
                   fail_on_error=False,
                   member=False,
                   number: int = 0,
                   total: int = 0):
    if (filepath.endswith(".cpp") or filepath.endswith(".cc") or filepath.endswith(".h")
            or filepath.endswith(".i") or filepath.endswith(".ii")
            or filepath.endswith(".hpp")):
        # cprint(bcolors.ENDC,
        #        f"\n−−−−−−−−−−\nTesting Cpp file {number:5d}/{total:5d} "
        #        f"{bcolors.MAGENTA}{filepath}")
        with open(filepath, "r") as f:
            try:
                cpp = f.read()
                if not roundtrip(cpp, filepath,
                                 check_hypothesis=check_hypothesis,
                                 fail_on_error=fail_on_error, member=member,
                                 skip_checks=skip_checks):
                    # cprint(bcolors.RED,
                    #        f"**Warn**{bcolors.ENDC} Test failed for "
                    #        f"file: {bcolors.MAGENTA}{filepath}")
                    return False
                else:
                    # cprint(bcolors.GREEN,
                    #        f"Success for file: {bcolors.MAGENTA}{filepath}")
                    return True
            except UnicodeDecodeError:
                cprint(bcolors.RED,
                       f"Error: Cannot decode file as UTF-8. Ignoring: "
                       f"{filepath}")
                return False
    else:
        return None


def check_compile_command_db(compile_command: Dict[str, str],
                              check_hypothesis: bool = False,
                              skip_checks: bool = False,
                              fail_on_error=False):
    cprint(bcolors.ENDC,
            f"\n−−−−−−−−−−\nTesting Cpp file "
            f"{bcolors.MAGENTA}{compile_command}")
    try:
        if not roundtrip(compile_command=compile_command,
                            check_hypothesis=check_hypothesis,
                            skip_checks=skip_checks,
                            fail_on_error=fail_on_error,
                            member=False):
            cprint(bcolors.RED,
                    f"**Warn**{bcolors.ENDC} Test failed for "
                    f"file: {bcolors.MAGENTA}{compile_command}")
            if fail_on_error:
                return False
        # else:
        #     cprint(bcolors.GREEN,
        #             f"Success for file: {bcolors.MAGENTA}{compile_command}")
        #     # return True
    except UnicodeDecodeError:
        cprint(bcolors.RED,
                f"Error: Cannot decode file as UTF-8. Ignoring: "
                f"{compile_command}")
        return False
    return True


def check_compile_commands_db(compile_commands: List[Dict[str, str]],
                              check_hypothesis: bool = False,
                              skip_checks: bool = False,
                              fail_on_error=False):
    # for number, compile_command in enumerate(compile_commands, start=1):
    #     cprint(bcolors.ENDC,
    #            f"\n−−−−−−−−−−\nTesting Cpp file {number:5d}/{len(compile_commands):5d} "
    #            f"{bcolors.MAGENTA}{compile_command}")
    #     res = check_compile_command_db(compile_command, check_hypothesis, skip_checks, fail_on_error)
    #     if fail_on_error and not res:
    #         return False
    # return True
    commands = [(command, check_hypothesis, skip_checks, fail_on_error)
                for command in compile_commands]
    with Pool() as pool:
        for _ in tqdm.tqdm(pool.istarmap(check_compile_command_db, commands),
                           total=len(commands)):
            pass
        pool.close()
        pool.join()
    return True


def stats(nb_ok: int, nb_ko: int, ignored: List[str]):
    if nb_ok == nb_ko == 0:
        logger.info(f"No tests, no stats")
        return None
    print(f"Success: {nb_ok}/{nb_ok+nb_ko} ({int(nb_ok*100.0/(nb_ok+nb_ko))}%)")
    if ignored:
        logger.info(f"Ignored: {', '.join(ignored)})")


def collect_files(dir: str) -> int:
    res = []
    for subdir, _, files in os.walk(dir):
        for filename in files:
            filepath = os.path.join(subdir, filename)
            res.append(filepath)
    return res


def load_exclusions(exclusions_file: str) -> List[str]:
    exclusions = []
    if exclusions_file:
        with open(exclusions_file, 'r') as ex:
            for exclusion in ex.readlines():
                exclusion = exclusion.strip()
                if exclusion and exclusion[0] != '#':
                    exclusions.append(exclusion)
    logger.info(f"loaded exclusions are: {exclusions}")
    return exclusions


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-D', '--debug', default=False,
                            action='store_true',
                            help='If set, print additional debug messages.')
    arg_parser.add_argument('-c', '--check_hypothesis', default=False,
                            action='store_true',
                            help='If set, the hypothesis parse tree will be '
                            'tested.')
    arg_parser.add_argument('-d', '--dir', default='test',
                            type=str,
                            help='Set the files in the given dir to be tested.')
    arg_parser.add_argument('--db', default=None, type=str,
                            help=('Set the compile_commands.json daabase listing the '
                                  'files to analyze.'))
    arg_parser.add_argument('-F', '--fail_on_error', default=False,
                            action='store_true',
                            help=('If set, exit at first error. Otherwise, '
                                  'continue on next file.'))
    arg_parser.add_argument('-l', '--list', default=False,
                            action='store_true',
                            help='If set, use the hardcoded Cpp files list. '
                                  'Otherwise, walk the test directory for '
                                  'Cpp files')
    arg_parser.add_argument('-m', '--member', default=False,
                            action='store_true',
                            help='If set, consider the file content as the '
                                  'code of a member instead of a complete '
                                  'compilation unit.')
    arg_parser.add_argument('-s', '--skip-checks', dest='skip_checks',default=False,
                            action='store_true',
                            help='If set, parse without any sanity checks.')
    arg_parser.add_argument('-x', '--exclude', action='append', default=[],
                            type=str,
                            help='Exclude the given file from being tested.')
    arg_parser.add_argument('-X', '--exclusions', type=str,
                            help='Read the exclusions from the given file.')
    arg_parser.add_argument('-f', '--file', action='append',
                            type=str,
                            help='Set the given file to be tested.')
    args = arg_parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.root.setLevel(logging.DEBUG)

    fail_on_error = args.fail_on_error
    check_hypothesis = args.check_hypothesis
    skip_checks = args.skip_checks
    exclusions = args.exclude
    exclusions.extend(load_exclusions(args.exclusions))
    nb_ok = 0
    nb_ko = 0
    ignored = []
    filepaths = [
        "test.cpp"
    ]

    files = None
    if args.db:
        pass
    elif args.list:
        files = filepaths
    elif args.file:
        files = args.file
    else:
        files = collect_files(args.dir)
        files.remove(f"{args.dir}/exclusions.txt")
    logger.debug(files)

    if files:
        filtered_files = [f for f in files if f not in exclusions]

        total = len(filtered_files)
        for test_num, filepath in enumerate(filtered_files, start=1):
            test_result = check_filepath(filepath,
                                         check_hypothesis=check_hypothesis,
                                         skip_checks=skip_checks,
                                         fail_on_error=fail_on_error,
                                         member=args.member,
                                         number=test_num,
                                         total=total)
            if test_result is not None:
                if test_result:
                    nb_ok = nb_ok + 1
                else:
                    nb_ko = nb_ko + 1
                    if fail_on_error:
                        stats(nb_ok, nb_ko, ignored)
                        exit(1)
            else:
                ignored.append(filepath)
        stats(nb_ok, nb_ko, ignored)
    elif args.db:
        with open(args.db) as json_file:
            compile_commands = json.load(json_file)
            check_compile_commands_db(compile_commands,
                                      check_hypothesis=check_hypothesis,
                                      skip_checks=skip_checks,
                                      fail_on_error=fail_on_error)
