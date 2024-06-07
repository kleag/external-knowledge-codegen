import argparse
import errno
import json
import numpy as np
import os
import pickle
import sys
import tqdm
import uuid

import cppastor
import cpplang
import cpplang.parse

from asdl.asdl import ASDLGrammar
from asdl.lang.cpp.cpp_transition_system import (asdl_ast_to_cpp_ast,
                                                 cpp_ast_to_asdl_ast,
                                                 CppTransitionSystem)
from components.action_info import get_action_infos
from components.dataset import Example
from components.vocab import Vocab, VocabEntry
from datasets.codagen.evaluator import CodagenEvaluator
from datasets.codagen.util import (compare_ast,
                                  canonicalize_intent, canonicalize_code,
                                  decanonicalize_code, tokenize_intent)

from asdl.hypothesis import *


def preprocess_example(transition_system: CppTransitionSystem,
                       compile_command: str,
                       debug: bool = False):
    # logger.debug("codagen/dataset preprocess_example")
    cpp_ast = cpplang.parse.parse(compile_command=compile_command, debug=debug)
    canonical_code = cppastor.to_source(cpp_ast).strip()
    # if debug:
    #     print(f"canonical_code:\n{canonical_code}", file=sys.stderr)
    asdl_ast = cpp_ast_to_asdl_ast(cpp_ast, transition_system.grammar)
    tgt_actions = transition_system.get_actions(asdl_ast)

    # sanity check
    hyp = transition_system.get_hypothesis(tgt_actions)
    cpp_ast_reconstructed = asdl_ast_to_cpp_ast(asdl_ast,
                                                transition_system.grammar)
    cpp_ast_reconstructed = asdl_ast_to_cpp_ast(hyp.tree,
                                                transition_system.grammar)
    code_from_hyp = cppastor.to_source(cpp_ast_reconstructed).strip()

    hyp.code = code_from_hyp
    # if debug:
    #     print(f"code_from_hyp:\n{code_from_hyp}", file=sys.stderr)
    assert code_from_hyp == canonical_code

    # decanonicalized_code_from_hyp = decanonicalize_code(
    #     code_from_hyp, example_dict['slot_map'])
    # cpp_ast = cpplang.parse.parse(s=cpp_code, filepath=None,
    #                                 compile_command=None)
    # decanonicalized_code_from_hyp_ast = cpplang.parse.parse(
    #     s=decanonicalized_code_from_hyp, filepath=filepath,
    #     compile_command=compile_command)
    #
    # assert compare_ast(cpp_ast, decanonicalized_code_from_hyp_ast)

    # assert transition_system.compare_ast(
    #     transition_system.surface_code_to_ast(
    #         decanonicalized_code_from_hyp),
    #     transition_system.surface_code_to_ast(example_json['snippet']))

    tgt_action_infos = get_action_infos({}, tgt_actions)
    return (asdl_ast, canonical_code, tgt_action_infos)


def preprocess_dataset(file_path: str,
                       transition_system: CppTransitionSystem,
                       num_examples: int = None,
                       debug: bool = False):
    with open(file_path) as fp:
        try:
            dataset = json.load(fp)
        except Exception as e:
            # TODO handle opening errors
            with open(file_path) as fp:
                dataset = [json.loads(jline) for jline in fp.readlines()]
    if num_examples:
        dataset = dataset[:num_examples]
    examples = []
    evaluator = CodagenEvaluator(transition_system)
    for compile_command in tqdm.tqdm(dataset):
        try:
            (asdl_ast, canonical_code, tgt_action_infos) = preprocess_example(
                transition_system, compile_command, debug)

            if debug:
                print(f"canonical_code:\n{canonical_code}", file=sys.stderr)
        except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
            raise
        example = Example(idx=str(uuid.uuid4()),
                          src_sent="",
                          tgt_actions=tgt_action_infos,
                          tgt_code=canonical_code,
                          tgt_ast=asdl_ast,
                          meta={})

        examples.append(example)

    return examples


def preprocess_codagen_dataset(compile_commands: str,
                               grammar_file: str,
                               code_freq: int = 3,
                               vocab_size: int = 20000,
                               num_examples: int = 0,
                               debug: bool = False,
                               out_dir: str = 'data/codagen'):
    np.random.seed(1234)
    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    asdl_text = None
    try:
        grammar_file_fd = open(grammar_file, "r")
    except:
        print(f"Error opening grammar file: {grammar_file}", file=sys.stderr)
        sys.exit(1)
    else:
        with grammar_file_fd:
            asdl_text = grammar_file_fd.read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = CppTransitionSystem(grammar)

    print(f'process gold training data {compile_commands}',
          file=sys.stderr)
    train_examples = preprocess_dataset(compile_commands,
                                        transition_system=transition_system,
                                        num_examples=num_examples,
                                        debug=debug)

    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action,
                                                        GenTokenAction),
                            e.tgt_actions))
                        for e in train_examples]
    primitive_vocab = VocabEntry.from_corpus(primitive_tokens,
                                             size=vocab_size,
                                             freq_cutoff=code_freq)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder')
                   for e in train_examples]

    code_vocab = VocabEntry.from_corpus(code_tokens, size=vocab_size,
                                        freq_cutoff=code_freq)

    vocab = Vocab(code=code_vocab)
    print(f'generated vocabulary {repr(vocab)}', file=sys.stderr)

    action_lens = [len(e.tgt_actions) for e in train_examples]
    print(f'Max action len: {max(action_lens)}', file=sys.stderr)
    print(f'Avg action len: {np.average(action_lens)}', file=sys.stderr)
    print(f'Actions larger than 100: '
          f'{len(list(filter(lambda x: x > 100, action_lens)))}',
          file=sys.stderr)

    pickle.dump(train_examples,
                open(os.path.join(out_dir, 'train.all.bin'), 'wb'))
    vocab_name = f'vocab.code_freq{code_freq}.bin'
    pickle.dump(vocab, open(os.path.join(out_dir, vocab_name), 'wb'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    # ### General configuration ####
    arg_parser.add_argument('--compile-commands', type=str,
                            help='Path to a compile commands database',
                            )
    arg_parser.add_argument('--grammar', type=str,
                            help='Path to language grammar',
                            # default='src/asdl/lang/py3/py3_asdl.simplified.txt'
                            )
    arg_parser.add_argument('--out-dir', type=str, default='data/codagen',
                            help='Path to output file')
    arg_parser.add_argument('--freq', type=int, default=3,
                            help='minimum frequency of tokens')
    arg_parser.add_argument('--vocabsize', type=int, default=20000,
                            help='First k number from pretrain file')
    arg_parser.add_argument('--num-examples', type=int, default=0,
                            help='Max number of examples to use in any set')
    arg_parser.add_argument('-d', '--debug', action='store_true',
                            help='Run in debug mode if set.')
    args = arg_parser.parse_args()

    preprocess_codagen_dataset(compile_commands=args.compile_commands,
                               grammar_file=args.grammar,
                               code_freq=args.freq,
                               vocab_size=args.vocabsize,
                               num_examples=args.num_examples,
                               out_dir=args.out_dir,
                               debug=args.debug)
