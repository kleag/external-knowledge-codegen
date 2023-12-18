import argparse
import errno
import json
import os
import pickle
import sys

import numpy as np

from asdl.hypothesis import *
from asdl.transition_system import *
from components.action_info import get_action_infos
from components.dataset import Example
from components.vocab import Vocab, VocabEntry
from datasets.codagen.evaluator import CodagenEvaluator
from datasets.codagen.util import (compare_ast,
                                  canonicalize_intent, canonicalize_code,
                                  decanonicalize_code, tokenize_intent)

import cppastor
import cpplang
import istarmap
from asdl.lang.cpp.cpp_transition_system import (asdl_ast_to_cpp_ast,
                                             cpp_ast_to_asdl_ast)
from asdl.hypothesis import *


def preprocess_codagen_dataset(train_file: str,
                              test_file: str,
                              grammar_file: str,
                              tokenizer: str,
                              src_freq: int = 3,
                              code_freq: int = 3,
                              rewritten: bool = True,
                              mined_data_file: str = None,
                              api_data_file: str = None,
                              vocab_size: int = 20000,
                              num_examples: int = 0,
                              num_mined: int = 0,
                              num_dev: int = 200,
                              debug: bool = False,
                              start_at: int = 0,
                              out_dir: str = 'data/codagen'):
    np.random.seed(1234)
    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    asdl_text = open(grammar_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = CppTransitionSystem(grammar)

    print('process gold training data...')
    train_examples = preprocess_dataset(train_file,
                                        name='train',
                                        tokenizer=tokenizer,
                                        transition_system=transition_system,
                                        num_examples=num_examples,
                                        debug=debug,
                                        start_at=start_at,
                                        rewritten=rewritten)

    # held out 200 examples for development
    full_train_examples = train_examples[:]
    np.random.shuffle(train_examples)
    dev_examples = train_examples[:num_dev]
    train_examples = train_examples[num_dev:]

    mined_examples = []
    api_examples = []
    if mined_data_file and num_mined > 0:
        print("use mined data: ", num_mined)
        print("from file: ", mined_data_file)
        mined_examples = preprocess_dataset(mined_data_file,
                                            name='mined',
                                            tokenizer=tokenizer,
                                            transition_system=transition_system,
                                            num_examples=num_mined,
                                            debug=debug,
                                            start_at=start_at,
                                            rewritten=rewritten)
        pickle.dump(mined_examples, open(os.path.join(out_dir, 'mined_{}.bin'.format(num_mined)), 'wb'))

    if api_data_file:
        print("use api docs from file: ", api_data_file)
        name = os.path.splitext(os.path.basename(api_data_file))[0]
        api_examples = preprocess_dataset(api_data_file,
                                          name='api',
                                          tokenizer=tokenizer,
                                          transition_system=transition_system,
                                          debug=debug,
                                          start_at=start_at,
                                          rewritten=rewritten)
        pickle.dump(api_examples, open(os.path.join(out_dir, name + '.bin'), 'wb'))

    if mined_examples and api_examples:
        pickle.dump(mined_examples + api_examples, open(os.path.join(out_dir, 'pre_{}_{}.bin'.format(num_mined, name)), 'wb'))

    # combine to make vocab
    train_examples += mined_examples
    train_examples += api_examples
    print(f'{len(train_examples)} training instances', file=sys.stderr)
    print(f'{len(dev_examples)} dev instances', file=sys.stderr)

    print('process testing data...')
    test_examples = preprocess_dataset(test_file,
                                       name='test',
                                       tokenizer=tokenizer,
                                       transition_system=transition_system,
                                       rewritten=rewritten)
    print(f'{len(test_examples)} testing instances', file=sys.stderr)

    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=vocab_size,
                                       freq_cutoff=src_freq)
    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_examples]
    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=vocab_size, freq_cutoff=code_freq)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_examples]

    code_vocab = VocabEntry.from_corpus(code_tokens, size=vocab_size, freq_cutoff=code_freq)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_lens = [len(e.tgt_actions) for e in train_examples]
    print('Max action len: %d' % max(action_lens), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_lens), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_lens))), file=sys.stderr)

    pickle.dump(train_examples, open(os.path.join(out_dir, 'train.all_{}.bin'.format(num_mined)), 'wb'))
    pickle.dump(full_train_examples, open(os.path.join(out_dir, 'train.gold.full.bin'), 'wb'))
    pickle.dump(dev_examples, open(os.path.join(out_dir, 'dev.bin'), 'wb'))
    pickle.dump(test_examples, open(os.path.join(out_dir, 'test.bin'), 'wb'))
    if mined_examples and api_examples:
        vocab_name = 'vocab.src_freq%d.code_freq%d.mined_%s.%s.bin' % (src_freq, code_freq, num_mined, name)
    elif mined_examples:
        vocab_name = 'vocab.src_freq%d.code_freq%d.mined_%s.bin' % (src_freq, code_freq, num_mined)
    elif api_examples:
        vocab_name = 'vocab.src_freq%d.code_freq%d.%s.bin' % (src_freq, code_freq, name)
    else:
        vocab_name = 'vocab.src_freq%d.code_freq%d.bin' % (src_freq, code_freq)
    pickle.dump(vocab, open(os.path.join(out_dir, vocab_name), 'wb'))


def get_example_actions(example_dict: dict,
                       transition_system: CppTransitionSystem,
                       cpp_code: str,
                       file_path: str,
                       compile_command: str):
    # logger.debug("codagen/dataset get_example_actions")
    snippet = example_dict['canonical_snippet']

    cpp_ast_reconstructed = ast.parse(snippet)
    cpp_ast = cpplang.parse.parse(s=snippet, filepath=filepath,
                                    compile_command=compile_command)
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

    decanonicalized_code_from_hyp = decanonicalize_code(
        code_from_hyp, example_dict['slot_map'])
    cpp_ast = cpplang.parse.parse(s=cpp_code, filepath=filepath,
                                    compile_command=compile_command)
    decanonicalized_code_from_hyp_ast = cpplang.parse.parse(
        s=decanonicalized_code_from_hyp, filepath=filepath,
        compile_command=compile_command)

    assert compare_ast(cpp_ast, decanonicalized_code_from_hyp_ast)

    assert transition_system.compare_ast(
        transition_system.surface_code_to_ast(
            decanonicalized_code_from_hyp),
        transition_system.surface_code_to_ast(example_json['snippet']))

    tgt_action_infos = get_action_infos(example_dict['intent_tokens'],
                                        tgt_actions)
    return tgt_action_infos


def preprocess_dataset(file_path: str,
                       transition_system: CppTransitionSystem,
                       tokenizer: str,
                       name: str = 'train',
                       num_examples: int = None,
                       debug: bool = False,
                       start_at: int = 0,
                       rewritten: bool = True):
    try:
        dataset = json.load(open(file_path))
    except Exception as e:
        # TODO handle opening errors
        dataset = [json.loads(jline) for jline in open(file_path).readlines()]
    if num_examples:
        dataset = dataset[:num_examples]
    examples = []
    evaluator = CodagenEvaluator(transition_system)
    f = open(file_path + '.debug', 'w')
    skipped_list = []
    for i, example_json in enumerate(dataset):
        if i < start_at:
            continue
        if debug:
            print(f"preprocess_dataset example n°{i+1}/{len(dataset)}",
                  end='\n', file=sys.stderr)
        else:
            print(f">>>>>>>> preprocess_dataset example n°{i+1}/{len(dataset)}",
                  end='\r', file=sys.stderr)
        try:
            example_dict = preprocess_example(example_json,
                                              tokenizer,
                                              rewritten=rewritten)
            tgt_action_infos = get_example_actions(example_dict,
                                                   transition_system,
                                                   example_json['snippet'],
                                                   file_path: str,
                                                   compile_command)
        except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
            skipped_list.append(example_json['question_id'])
            continue
        example = Example(idx=f'{i}-{example_json["question_id"]}',
                          src_sent=example_dict['intent_tokens'],
                          tgt_actions=tgt_action_infos,
                          tgt_code=canonical_code,
                          tgt_ast=asdl_ast,
                          meta=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))
        assert evaluator.is_hyp_correct(example, hyp)

        examples.append(example)

        # log!
        f.write(f'Example: {example.idx}\n')
        if rewritten and 'rewritten_intent' in example.meta['example_dict']:
            f.write(f"Original Utterance: {example.meta['example_dict']['rewritten_intent']}\n")
        else:
            f.write(f"Original Utterance: {example.meta['example_dict']['intent']}\n")
        f.write(f"Original Snippet: {example.meta['example_dict']['snippet']}\n")
        f.write(f"\n")
        f.write(f"Utterance: {' '.join([str(t) for t in example.src_sent])}\n")
        f.write(f"Snippet: {example.tgt_code}\n")
        f.write(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    f.close()
    print('Skipped due to exceptions: %d' % len(skipped_list), file=sys.stderr)
    return examples


def preprocess_example(example_json: str,
                       tokenizer: str,
                       rewritten: bool = True):
    intent = example_json['intent']
    if rewritten and 'rewritten_intent' in example_json:
        rewritten_intent = example_json['rewritten_intent']
    else:
        rewritten_intent = None

    if not rewritten or rewritten_intent is None:
        rewritten_intent = intent
    snippet = example_json['snippet']

    canonical_intent, slot_map = canonicalize_intent(rewritten_intent
                                                     if rewritten else intent)
    canonical_snippet = canonicalize_code(snippet, slot_map)
    intent_tokens = tokenize_intent(canonical_intent, tokenizer)
    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

    reconstructed_snippet = cppastor.to_source(ast.parse(snippet)).strip()
    reconstructed_decanonical_snippet = cppastor.to_source(ast.parse(decanonical_snippet)).strip()

    assert compare_ast(ast.parse(reconstructed_snippet), ast.parse(reconstructed_decanonical_snippet))

    return {'canonical_intent': canonical_intent,
            'intent_tokens': intent_tokens,
            'slot_map': slot_map,
            'canonical_snippet': canonical_snippet}


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    # ### General configuration ####
    arg_parser.add_argument('--train', type=str, help='Path to train file',
                            default='data/codagen/codagen-train.json')
    arg_parser.add_argument('--test', type=str, help='Path to test file',
                            default='data/codagen/codagen-test.json')
    arg_parser.add_argument('--mined', type=str, help='Path to mined file')
    arg_parser.add_argument('--grammar', type=str,
                            help='Path to language grammar',
                            default='src/asdl/lang/py3/py3_asdl.simplified.txt')
    arg_parser.add_argument('--out-dir', type=str, default='data/codagen',
                            help='Path to output file')
    arg_parser.add_argument('--freq', type=int, default=3,
                            help='minimum frequency of tokens')
    arg_parser.add_argument('--vocabsize', type=int, default=20000,
                            help='First k number from pretrain file')
    arg_parser.add_argument('--include-api', type=str,
                            help='Path to apidocs file')
    arg_parser.add_argument('-r', '--no_rewritten', action='store_false',
                            help='If set, will not use the manually rewritten '
                                 'intents.')
    arg_parser.add_argument('--tokenizer', type=str, required=True,
                            choices=['nltk', 'bert', 'spacy', 'lima'],
                            help='The tokenizer to use.')
    arg_parser.add_argument('--num-examples', type=int, default=0,
                            help='Max number of examples to use in any set')
    arg_parser.add_argument('--num-dev', type=int, default=200,
                            help='Max number of dev examples to use')
    arg_parser.add_argument('--num-mined', type=int, default=0,
                            help='First k number from mined file')
    arg_parser.add_argument('-d', '--debug', action='store_true',
                            help='Run in debug mode if set.')
    args = arg_parser.parse_args()

    print(f"tokenizer: {args.tokenizer}", file=sys.stderr)
    preprocess_codagen_dataset(train_file=args.train,
                              test_file=args.test,
                              mined_data_file=args.mined,
                              api_data_file=args.include_api,
                              grammar_file=args.grammar,
                              src_freq=args.freq, code_freq=args.freq,
                              vocab_size=args.vocabsize,
                              num_examples=args.num_examples,
                              num_mined=args.num_mined,
                              num_dev=args.num_dev,
                              out_dir=args.out_dir,
                              rewritten=args.no_rewritten,
                              tokenizer=args.tokenizer,
                              debug=args.debug)
