# coding=utf-8

from __future__ import print_function

import itertools
import re
import ast
import astor
import nltk
from transformers import BertTokenizer, AutoTokenizer


QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")
BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')


def compare_ast(node1, node2):
    if not isinstance(node1, str):
        if type(node1) is not type(node2):
            return False
    if isinstance(node1, ast.AST):
        for k, v in list(vars(node1).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, zip(node1, node2)))
    else:
        return node1 == node2


def tokenize_intent(intent: str, tokenizer: str):
    """
    Tokenize the input string.

    Parameters:
    intent: str: the text intent
    tokenizer: str: the kind of tokenizer

    Returns:
    """
    if tokenizer == 'nltk':
        lower_intent = intent.lower()
        tokens = nltk.word_tokenize(lower_intent)
    elif tokenizer == 'bert':
        encoded_input = BERT_TOKENIZER(
            text=intent,
            add_special_tokens=True,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_attention_mask=False
        )
        breakpoint()
        tokens = encoded_input["input_ids"]
    elif tokenizer == 'starcoder':  # Add this branch for the starcoder tokenizer
        starcoder_tokenizer = AutoTokenizer.from_pretrained('starcoder-path-or-model-name')
        encoded_input = starcoder_tokenizer(
            text=intent,
            add_special_tokens=True,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_attention_mask=False
        )
        tokens = encoded_input["input_ids"]
    elif tokenizer == 'spacy':
        raise NotImplementedError("Spacy tokenization not implemented")
    elif tokenizer == 'lima':
        raise NotImplementedError("Lima tokenization not implemented")
    else:
        raise RuntimeError(f"tokenize_intent: unknown tokenizer: {tokenizer}")
    # breakpoint()
    return tokens


def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'


def canonicalize_intent(intent):
    """
    Replace occurrences of strings and variable names with canonicalized values
    (var_1, var_1, …, str_1, str_2, …).

    Returns:
    The intent with canonicalized values and a map allowing to retrieve the
    original values.

    """
    # handle the following special case: quote is `''`
    marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

    slot_map = dict()
    var_id = 0
    str_id = 0
    for match in marked_token_matches:
        quote = match[0]
        value = match[1]
        quoted_value = quote + value + quote

        # try:
        #     # if it's a number, then keep it and leave it to the copy mechanism
        #     float(value)
        #     intent = intent.replace(quoted_value, value)
        #     continue
        # except:
        #     pass

        slot_type = infer_slot_type(quote, value)

        if slot_type == 'var':
            slot_name = f'var_{var_id}'
            var_id += 1
            slot_type = 'var'
        else:
            slot_name = f'str_{str_id}'
            str_id += 1
            slot_type = 'str'

        # slot_id = len(slot_map)
        # slot_name = 'slot_%d' % slot_id
        # # make sure slot_name is also unicode
        # slot_name = unicode(slot_name)

        intent = intent.replace(quoted_value, slot_name)
        slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                               'quote': quote,
                               'type': slot_type}

    return intent, slot_map


def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    # Python 3
                    # if isinstance(slot_name, unicode):
                    #     try: slot_name = slot_name.encode('ascii')
                    #     except: pass

                    setattr(node, k, slot_name)


def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """

    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in ('}', ']', ')')


def canonicalize_code(code, slot_map):
    string2slot = {x['value']: slot_name for slot_name, x in list(slot_map.items())}

    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast).strip()

    # the following code handles the special case that
    # a list/dict/set mentioned in the intent, like
    # Intent: zip two lists `[1, 2]` and `[3, 4]` into a list of two tuples containing elements at the same index in each list
    # Code: zip([1, 2], [3, 4])

    entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val['value'])]
    if entries_that_are_lists:
        for slot_name in entries_that_are_lists:
            list_repr = slot_map[slot_name]['value']
            #if list_repr[0] == '[' and list_repr[-1] == ']':
            first_token = list_repr[0]  # e.g. `[`
            last_token = list_repr[-1]  # e.g., `]`
            fake_list = first_token + slot_name + last_token
            slot_map[fake_list] = slot_map[slot_name]
            # else:
            #     fake_list = slot_name

            canonical_code = canonical_code.replace(list_repr, fake_list)

    return canonical_code


def decanonicalize_code(code, slot_map):
    for slot_name, slot_val in slot_map.items():
        if is_enumerable_str(slot_name):
            code = code.replace(slot_name, slot_val['value'])

    slot2string = {x[0]: x[1]['value'] for x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, slot2string)
    raw_code = astor.to_source(py_ast).strip()
    # for slot_name, slot_info in slot_map.items():
    #     raw_code = raw_code.replace(slot_name, slot_info['value'])

    return raw_code
