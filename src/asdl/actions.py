# coding=utf-8

class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return (isinstance(other, ApplyRuleAction)
                and self.production == other.production)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f'ApplyRule[{self.production.__repr__()}]'


class GenTokenAction(Action):
    def __init__(self, token):
        assert(type(token) == str)
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return f'GenToken[{self.token}]'


class ReduceAction(Action):
    def __repr__(self):
        return 'Reduce'
