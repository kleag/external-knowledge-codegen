# coding=utf-8
from asdl.hypothesis import Hypothesis
from asdl.actions import ApplyRuleAction, GenTokenAction


class ActionInfo(object):
    """sufficient statistics for making a prediction of an action at a time step"""

    def __init__(self, action=None):
        self.t = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None

        # for GenToken actions only
        self.copy_from_src = False
        self.src_token_position = -1

    def __repr__(self, verbose=False):
        ffr = (self.frontier_field.__repr__(True)
               if self.frontier_field else "None")
        repr_str = (f'{repr(self.action)} (t={self.t}, p_t={self.parent_t}, '
                    f'frontier_field={ffr})')

        if verbose:
            verbose_repr = 'action_prob=%.4f, ' % self.action_prob
            if isinstance(self.action, GenTokenAction):
                verbose_repr += (f'in_vocab={self.in_vocab}, '
                                f'gen_copy_switch={self.gen_copy_switch}, '
                                f'p(gen)={self.gen_token_prob}, '
                                f'p(copy)={self.copy_token_prob}, '
                                f'has_copy={self.copy_from_src}, '
                                f'copy_pos={self.src_token_position}')

            repr_str += '\n' + verbose_repr

        return repr_str


def get_action_infos(src_query, tgt_actions, force_copy=False):
    action_infos = []
    hyp = Hypothesis()
    for t, action in enumerate(tgt_actions):
        action_info = ActionInfo(action)
        action_info.t = t
        if hyp.frontier_node:
            action_info.parent_t = hyp.frontier_node.created_time
            action_info.frontier_prod = hyp.frontier_node.production
            action_info.frontier_field = hyp.frontier_field.field

        if isinstance(action, GenTokenAction):
            try:
                tok_src_idx = src_query.index(str(action.token))
                action_info.copy_from_src = True
                action_info.src_token_position = tok_src_idx
            except ValueError:
                if force_copy: raise ValueError(
                    f'cannot copy primitive token {action.token} from source')

        hyp.apply_action(action)
        action_infos.append(action_info)

    return action_infos
