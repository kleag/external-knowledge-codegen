# coding=utf-8

import sys
from asdl.actions import (ApplyRuleAction, GenTokenAction, ReduceAction)
from asdl.hypothesis import Hypothesis

debug = True

class TransitionSystem(object):
    def __init__(self, grammar, debug=False):
        self.grammar = grammar
        self.debug = debug

    def get_actions(self, asdl_ast):
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions = []

        # if asdl_ast is None:
        # return actions
        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            # is a composite field
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions = self.get_actions(field.value)
                    # if not field_actions:
                    # field_actions.append(ReduceAction())
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions = self.get_actions(val)
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions = self.get_actions(field.value)

                    # if an optional field is filled, then do not need Reduce
                    # action
                    if (field.cardinality == 'multiple'
                            or field.cardinality == 'optional'
                            and not field_actions):
                        field_actions.append(ReduceAction())
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)

                # if an optional field is filled, then do not need Reduce
                # action
                if (field.cardinality == 'multiple'
                        or field.cardinality == 'optional'
                        and not field_actions):
                    # reduce action
                    field_actions.append(ReduceAction())

            actions.extend(field_actions)

        return actions

    def get_hypothesis(self, actions):
        hyp = Hypothesis()
        for t, action in enumerate(actions):
            valid_continuating_types = self.get_valid_continuation_types(hyp)
            if action.__class__ not in valid_continuating_types:
                print(f"Error: Valid continuation types are "
                      f"{valid_continuating_types} "
                      f"but current action class is {action.__class__}",
                      file=sys.stderr)
                assert action.__class__ in valid_continuating_types
            if isinstance(action, ApplyRuleAction):
                valid_continuating_productions = self.get_valid_continuating_productions(hyp)
                if (action.production not in valid_continuating_productions
                        and hyp.frontier_node):
                    raise Exception(
                        f"{bcolors.BLUE}{action.production}"
                        f"{bcolors.ENDC} should be in {bcolors.OK}"
                        f"{self.grammar[hyp.frontier_field.type] if hyp.frontier_field else ''}"
                        f"{bcolors.ENDC}")
                assert action.production in valid_continuating_productions
            p_t = -1
            f_t = None
            if hyp.frontier_node:
                p_t = hyp.frontier_node.created_time
                f_t = hyp.frontier_field.field.__repr__(plain=True)
            if self.debug:
                print(f'\t[{t}] {action}, frontier field: {f_t}, '
                        f'parent: {p_t}')
            hyp = hyp.clone_and_apply_action(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None
        return hyp

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]

    @staticmethod
    def get_class_by_lang(lang):
        if lang == 'python':
            from .lang.py.py_transition_system import PythonTransitionSystem
            return PythonTransitionSystem
        elif lang == 'python3':
            from .lang.py3.py3_transition_system import Python3TransitionSystem
            return Python3TransitionSystem
        elif lang == 'lambda_dcs':
            from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem
            return LambdaCalculusTransitionSystem
        elif lang == 'prolog':
            from .lang.prolog.prolog_transition_system import PrologTransitionSystem
            return PrologTransitionSystem
        elif lang == 'java':
            from .lang.java.java_transition_system import JavaTransitionSystem
            return JavaTransitionSystem

        raise ValueError('unknown language %s' % lang)
