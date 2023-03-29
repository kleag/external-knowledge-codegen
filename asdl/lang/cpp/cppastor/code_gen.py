# -*- coding: utf-8 -*-
"""
Part of the astor library for Python AST manipulation.

License: 3-clause BSD

Copyright (c) 2008      Armin Ronacher
Copyright (c) 2012-2017 Patrick Maupin
Copyright (c) 2013-2017 Berker Peksag

This module converts an AST into Python source code.

Before being version-controlled as part of astor,
this code came from here (in 2012):

    https://gist.github.com/1250562

"""

import gc
import re
from cpplang.ast import Node
from cpplang import tree
import math
import re
import sys

from .op_util import get_op_symbol, get_op_precedence, Precedence
from .node_util import ExplicitNodeVisitor
from .string_repr import pretty_string, string_triplequote_repr
from .source_repr import pretty_source


def to_source(node, indent_with=" " * 4, add_line_information=False,
              pretty_string=pretty_string, pretty_source=pretty_source):
    """This function can convert a node tree back into python sourcecode.
    This is useful for debugging purposes, especially if you're dealing with
    custom asts not generated by python itself.

    It could be that the sourcecode is evaluable when the AST itself is not
    compilable / evaluable.  The reason for this is that the AST contains some
    more data than regular sourcecode does, which is dropped during
    conversion.

    Each level of indentation is replaced with `indent_with`.  Per default this
    parameter is equal to four spaces as suggested by PEP 8, but it might be
    adjusted to match the application's styleguide.

    If `add_line_information` is set to `True` comments for the line numbers
    of the nodes are added to the output.  This can be used to spot wrong line
    number information of statement nodes.

    """
    generator = SourceGenerator(indent_with, add_line_information,
                                pretty_string)
    generator.visit(node)
    generator.result.append("\n")
    if set(generator.result[0]) == set("\n"):
        generator.result[0] = ""
    return pretty_source(generator.result)


def precedence_setter(AST=Node, get_op_precedence=get_op_precedence,
                      isinstance=isinstance, list=list):
    """ This only uses a closure for performance reasons,
        to reduce the number of attribute lookups.  (set_precedence
        is called a lot of times.)
    """

    def set_precedence(value, *nodes):
        """Set the precedence (of the parent) into the children.
        """
        if isinstance(value, Node):
            value = get_op_precedence(value)
        for node in nodes:
            if isinstance(node, Node):
                node._pp = value
            elif isinstance(node, list):
                set_precedence(value, *node)
            else:
                assert node is None, node

    return set_precedence


set_precedence = precedence_setter()


#class Delimit(object):
    #"""A context manager that can add enclosing
       #delimiters around the output of a
       #SourceGenerator method.  By default, the
       #parentheses are added, but the enclosed code
       #may set discard=True to get rid of them.
    #"""

    #discard = False

    #def __init__(self, tree, *args):
        #""" use write instead of using result directly
            #for initial data, because it may flush
            #preceding data into result.
        #"""
        #delimiters = "()"
        #node = None
        #op = None
        #for arg in args:
            #if isinstance(arg, Node):
                #if node is None:
                    #node = arg
                #else:
                    #op = arg
            #else:
                #delimiters = arg
        #tree.write(delimiters[0])
        #result = self.result = tree.result
        #self.index = len(result)
        #self.closing = delimiters[1]
        #if node is not None:
            #self.p = p = get_op_precedence(op or node)
            #self.pp = pp = tree.get__pp(node)
            #self.discard = p >= pp

    #def __enter__(self):
        #return self

    #def __exit__(self, *exc_info):
        #result = self.result
        #start = self.index - 1
        #if self.discard:
            #result[start] = ""
        #else:
            #result.append(self.closing)


class SourceGenerator(ExplicitNodeVisitor):
    """This visitor is able to transform a well formed syntax tree into Python
    sourcecode.

    For more details have a look at the docstring of the `node_to_source`
    function.

    """

    using_unicode_literals = False

    def __init__(self, indent_with, add_line_information=False,
                 pretty_string=pretty_string,
                 # constants
                 len=len, isinstance=isinstance, callable=callable):
        self.result = []
        self.indent_with = indent_with
        self.add_line_information = add_line_information
        self.indentation = 0  # Current indentation level
        self.new_lines = 0  # Number of lines to insert before next code
        self.colinfo = 0, 0  # index in result of string containing linefeed, and
                             # position of last linefeed in that string
        self.pretty_string = pretty_string

        visit = self.visit
        newline = self.newline
        result = self.result
        append = result.append

        def write(*params):
            """ self.write is a closure for performance (to reduce the number
                of attribute lookups).
            """
            for item in params:
                if isinstance(item, Node):
                    visit(item)
                elif callable(item):
                    item()
                elif item == "\n":
                    newline()
                else:
                    if self.new_lines:
                        append("\n" * self.new_lines)
                        self.colinfo = len(result), 0
                        if self.indentation > 0:
                            append(self.indent_with * self.indentation)
                        self.new_lines = 0
                    if item:
                        #if item == 'NoParameters':
                            #breakpoint()
                        append(item)

        self.write = write

    def __getattr__(self, name, defaults=dict(keywords=(),
                    _pp=Precedence.highest).get):
        """ Get an attribute of the node.
            like dict.get (returns None if doesn't exist)
        """
        if not name.startswith("get_"):
            raise AttributeError
        geta = getattr
        shortname = name[4:]
        default = defaults(shortname)

        def getter(node):
            return geta(node, shortname, default)

        setattr(self, name, getter)
        return getter

    #def delimit(self, *args):
        #return Delimit(self, *args)

    def conditional_write(self, *stuff):
        if stuff[-1] is not None:
            self.write(*stuff)
            # Inform the caller that we wrote
            return True

    def newline(self, node=None, extra=0):
        self.new_lines = max(self.new_lines, 1 + extra)
        if node is not None and self.add_line_information:
            self.write(f"// line: {node.lineno}")
            self.new_lines = 1

    def visit_arguments(self, node):
        want_comma = []

        def write_comma():
            if want_comma:
                self.write(", ")
            else:
                want_comma.append(True)

        def loop_args(args, defaults):
            set_precedence(Precedence.Comma, defaults)
            padding = [None] * (len(args) - len(defaults))
            for arg, default in zip(args, padding + defaults):
                self.write(write_comma, arg)
                self.conditional_write("=", default)

        loop_args(node.args, node.defaults)
        self.conditional_write(write_comma, "*", node.vararg)

        kwonlyargs = self.get_kwonlyargs(node)
        if kwonlyargs:
            if node.vararg is None:
                self.write(write_comma, "*")
            loop_args(kwonlyargs, node.kw_defaults)
        self.conditional_write(write_comma, "**", node.kwarg)

    def statement(self, node, *params, **kw):
        self.newline(node)
        self.write(*params)

    def decorators(self, node, extra):
        self.newline(extra=extra)
        for decorator in node.decorator_list:
            self.statement(decorator, "@", decorator)

    def comma_list(self, items, trailing=False):
        # set_precedence(Precedence.Comma, *items)
        if items:
            for idx, item in enumerate(items):
                self.write(", " if idx else "", item)
            self.write("," if trailing else "")

    # Statements

    def visit_TranslationUnit(self, node: tree.TranslationUnit):
        if node.subnodes is not None:
            for c in node.subnodes:
                self.write(c)

    def visit_InitListExpr(self, node: tree.InitListExpr):
        self.write("{")
        self.comma_list(node.subnodes)
        self.write("}")
        self.newline(extra=1)

        #if node.package:
            #self.write(node.package)
        #if node.imports:
            #for imp in node.imports:
                #self.write(imp)
        #if node.types:
            #for type in node.types:
                #self.write(type)

    def visit_CXXRecordDecl(self, node: tree.CXXRecordDecl):
        assert node.name is not None
        self.write(node.kind, " ", node.name)
        if node.bases:
            self.write(" : ", node.bases)
        if node.subnodes is not None:
            if len(node.subnodes) > 0:
                self.write(" {", "\n")
                for c in node.subnodes:
                    self.write(c)
                self.write("}")
            elif len(node.complete_definition) > 0:
                self.write(" {", "\n")
                self.write("}")
        elif node.complete_definition:
            self.write(" {", "\n")
            self.write("}")
        self.write(";")
        self.newline(extra=1)

    def visit_CXXConstructorDecl(self, node: tree.CXXConstructorDecl):
        parameters = []
        initializers = []
        statements = []
        if node.subnodes is not None:
            for c in node.subnodes:
                if c.__class__.__name__ == "ParmVarDecl":
                    parameters.append(c)
                elif c.__class__.__name__ == "CXXCtorInitializer":
                    if c.subnodes is not None and len(c.subnodes) > 0:
                        initializers.append(c)
                else:
                    statements.append(c)
        self.write(node.name)
        self.write("(")
        if parameters:
            self.comma_list(parameters)
        self.write(")")
        if len(node.noexcept) > 0:
            self.write(" ", node.noexcept)
        if len(initializers) > 0:
            self.write(" : ")
            self.comma_list(initializers)
        if len(statements) > 0:
            for c in statements:
                self.write(c)
        else:
            if len(node.default) > 0:
                self.write(" = ", node.default)
            self.write(";")
        self.newline(extra=1)

    def visit_CXXConstructExpr(self, node: tree.CXXConstructExpr):
        if node.subnodes is not None and len(node.subnodes) > 0:
            self.write(node.subnodes[0])

    def visit_CXXCtorInitializer(self, node: tree.CXXCtorInitializer):
        self.write(node.name)
        self.write('(')
        self.comma_list(node.subnodes)
        self.write(')')
        #if node.subnodes is not None and len(node.subnodes) > 0:
            #self.write(node.subnodes[0])

    def visit_CXXDestructorDecl(self, node: tree.CXXDestructorDecl):
        if node.virtual == 'virtual':
            self.write("virtual", " ")
        self.write(node.name)
        self.write("(")
        self.write(")")
        if len(node.noexcept) > 0:
            self.write(" ", node.noexcept)
        if len(node.default) > 0:
            self.write(" = ", node.default)
            self.write(";")
        elif node.subnodes is not None and len(node.subnodes) > 0:
            for c in node.subnodes:
                self.write(c)
        else:
            self.write(";")
        self.newline(extra=1)

    def visit_NamespaceDecl(self, node: tree.NamespaceDecl):
        assert node.name is not None
        self.write("namespace ")
        self.write(node.name, "\n")
        self.write(" {", "\n")
        if node.subnodes is not None:
            for c in node.subnodes:
                self.write(c)
        self.write("}")
        self.newline(extra=1)

    def visit_UsingDirectiveDecl(self, node: tree.UsingDirectiveDecl):
        assert node.name is not None
        self.write("using namespace ", node.name, ";\n")

    def visit_AccessSpecDecl(self, node: tree.AccessSpecDecl):
        self.write(node.access_spec, ":", "\n")

    def visit_ParmVarDecl(self, node: tree.ParmVarDecl):
        self.write(node.type, " ", node.name)
        if node.subnodes is not None and len(node.subnodes) > 0:
            self.write(" = ", node.subnodes[0])

    def visit_ExprWithCleanups(self, node: tree.ExprWithCleanups):
        self.write(node.subnodes[0])
        #breakpoint()
        #if len(self.stack) > 1 and self.stack[-2] == 'CompoundStmt':
            #self.write(";")

    def visit_DeclRefExpr(self, node: tree.DeclRefExpr):
        if node.name.startswith("operator"):
            self.write(node.name.replace("operator", "", 1))
            return
        self.write(node.name)
        if node.kind == "CXXMethodDecl":
            self.write("(")
            self.comma_list(node.subnodes)
            self.write(")")

    def visit_MaterializeTemporaryExpr(self, node: tree.MaterializeTemporaryExpr):
        self.write(node.subnodes[0])

    def visit_CXXBindTemporaryExpr(self, node: tree.CXXBindTemporaryExpr):
        self.write(node.subnodes[0])

    def visit_ImplicitCastExpr(self, node: tree.ImplicitCastExpr):
        if node.subnodes is not None and len(node.subnodes) > 0:
            self.write(node.subnodes[0])
        else:
        # if node.name is not None and len(node.name) > 0:
            self.write(node.name)

    def visit_VarDecl(self, node: tree.VarDecl):
        if len(node.storage_class) > 0:
            self.write(node.storage_class, " ")
        # breakpoint()
        self.write(node.type, " ", node.name, node.array)
        if node.subnodes is not None and len(node.subnodes) > 0:
            if node.init == 'call':
                self.write("(", node.subnodes[0], ")")
            else:
                self.write(" = ", node.subnodes[0])
        self.conditional_write(";")

    def visit_FieldDecl(self, node: tree.FieldDecl):
        self.write(node.type, " ", node.name)
        if node.subnodes is not None and len(node.subnodes) > 0:
            self.write(" = ", node.subnodes[0], ";")
        else:
            self.write(";")
        self.newline(extra=1)

    def visit_TypeRef(self, node: tree.TypeRef):
        self.write(node.name)

    def visit_CXXMethodDecl(self, node: tree.CXXMethodDecl):
        parameters = []
        statements = []
        comment = ""
        for c in node.subnodes:
            if c.__class__.__name__ == "ParmVarDecl":
                parameters.append(c)
            elif c.__class__.__name__ == "FullComment":
                comment = c.comment
            else:
                statements.append(c)
        if len(comment) > 0:
            self.write("/** ", comment, "*/\n")
        if len(node.virtual) > 0:
            self.write("virtual ")
        self.write(node.return_type, " ")
        self.write(node.name)
        self.write("(")
        if parameters:
            self.comma_list(parameters)
        self.write(")")
        if len(node.const) > 0:
            self.write(" ", node.const)
        if len(node.noexcept) > 0:
            self.write(" ", node.noexcept)
        if len(statements) > 0:
            for c in statements:
                self.write(c)
        else:
            if len(node.default) > 0:
                self.write(" = ", node.default)
            self.write(";")
        self.newline(extra=1)

    def visit_FunctionDecl(self, node: tree.FunctionDecl):
        parameters = []
        statements = []
        for c in node.subnodes:
            if c.__class__.__name__ == "ParmVarDecl":
                parameters.append(c)
            else:
                statements.append(c)
        self.write(node.return_type, " ")
        self.write(node.name)
        self.write("(")
        if parameters:
            self.comma_list(parameters)
        self.write(")")
        if len(statements) > 0:
            for c in statements:
                self.write(c)
        else:
            self.write(";")
        self.newline(extra=1)

    def visit_IntegerLiteral(self, node: tree.IntegerLiteral):
        self.write(node.value)

    def visit_FloatingLiteral(self, node: tree.FloatingLiteral):
        self.write(node.value)

    def visit_CharacterLiteral(self, node: tree.CharacterLiteral):
        self.write("'", node.value, "'")

    def visit_StringLiteral(self, node: tree.StringLiteral):
        self.write(node.value)

    def visit_BinaryOperator(self, node: tree.BinaryOperator):
        self.write(node.subnodes[0])
        self.write(" ", node.opcode, " ")
        self.write(node.subnodes[1])

    def visit_UnaryOperator(self, node: tree.UnaryOperator):
        if node.postfix == "True":
            self.write(node.subnodes[0], node.opcode)
        else:
            self.write(node.opcode, node.subnodes[0])

    def visit_DeclStmt(self, node: tree.DeclStmt):
        self.write(node.subnodes[0])

    # ReturnStmt(identifier* label, expression expression)
    def visit_ReturnStmt(self, node: tree.ReturnStmt):
        if node.label:
            self.write(node.label, ": ", "\n")
        self.write("return ", node.subnodes[0], ";\n")

    def visit_NullStmt(self, node: tree.NullStmt):
        self.write(";")

    def visit_IfStmt(self, node: tree.IfStmt):
        if node.label:
            self.write(node.label, ": ", "\n")
        self.write("if (", node.subnodes[0], ")\n")
        self.write(node.subnodes[1])
        #if not isinstance(node.subnodes[1], tree.CompoundStmt):
            #self.write(";\n")
        if len(node.subnodes) > 2:
            self.write("else ", node.subnodes[2])
            if node.subnodes[2].__class__.__name__ not in ["CompoundStmt", "IfStmt"]:
                self.conditional_write(";")
                self.newline(extra=1)

    # BlockStatement(identifier? label, statement* statements)
    def visit_CompoundStmt(self, node: tree.CompoundStmt):
        if node.label:
            self.write(node.label, ": ", "\n")
        self.write("{", "\n")
        if node.subnodes is not None:
            for statement in node.subnodes:
                self.write(statement)
                if (isinstance(statement, tree.BinaryOperator)
                        or isinstance(statement, tree.ExprWithCleanups)
                        or isinstance(statement, tree.CXXOperatorCallExpr)
                        or isinstance(statement, tree.CXXMemberCallExpr)):
                    self.write(";\n")
        self.write("}")
        self.newline(extra=1)

    # prefix_operators, "postfix_operators", "qualifier", "selectors
    def visit_ParenExpr(self, node: tree.ParenExpr):
        self.write("(", node.subnodes[0], ")")


    def visit_SwitchStmt(self, node: tree.SwitchStmt):
        if node.label:
            self.write(node.label, ": ", "\n")
        self.write("switch (", node.subnodes[0], ")\n", node.subnodes[1])

    def visit_CaseStmt(self, node: tree.CaseStmt):
        self.write("case ", node.subnodes[0], ":\n", node.subnodes[1])

    def visit_BreakStmt(self, node: tree.BreakStmt):
        self.write("break;\n")

    def visit_MemberExpr(self, node: tree.MemberExpr):
        if 'addSearchPathes' in node.name:
            breakpoint()
        if (node.subnodes is not None and len(node.subnodes) > 0
                and node.subnodes[0].__class__.__name__ in ['CallExpr',
                                                            'DeclRefExpr',
                                                            'ImplicitCastExpr',
                                                            'MaterializeTemporaryExpr',
                                                            'MemberExpr']):
            #if node.op == "->":
                #breakpoint()
            self.write(node.subnodes[0], node.op)
        self.write(node.name)

    def visit_ConstantExpr(self, node: tree.ConstantExpr):
        self.write(node.value)

    def visit_DefaultStmt(self, node: tree.DefaultStmt):
        self.write("default:\n", node.subnodes[0])

    def visit_ClassTemplateDecl(self, node: tree.ClassTemplateDecl):
        parameters = []
        statements = []
        for c in node.subnodes:
            if c.__class__.__name__ in ["TemplateTypeParmDecl",
                                        "NonTypeTemplateParmDecl"]:
                parameters.append(c)
            else:
                statements.append(c)
        self.write("template<")
        if parameters:
            self.comma_list(parameters)
        self.write(">")
        if len(statements) > 0:
            for c in statements:
                self.write(c)
        else:
            self.write(";")
        self.newline(extra=1)

    def visit_FunctionTemplateDecl(self, node: tree.FunctionTemplateDecl):
        parameters = []
        statements = []
        for c in node.subnodes:
            if c.__class__.__name__ in ["TemplateTypeParmDecl",
                                        "NonTypeTemplateParmDecl"]:
                parameters.append(c)
            else:
                statements.append(c)
        self.write("template<")
        if parameters:
            self.comma_list(parameters)
        self.write(">")
        for c in statements:
            self.write(c)
        self.newline(extra=1)

    def visit_TemplateTypeParmDecl(self, node: tree.TemplateTypeParmDecl):
        self.write("typename", " ", node.name)

    def visit_NonTypeTemplateParmDecl(self, node: tree.NonTypeTemplateParmDecl):
        self.write(node.type, " ", node.name)

    def visit_FullComment(self, node: tree.FullComment):
        self.write("/** ", node.comment, "*/")

    def visit_OverrideAttr(self, node: tree.OverrideAttr):
        pass
        #self.write(" ", "override", ";")
        #self.newline(extra=1)

    def visit_CXXMemberCallExpr(self, node: tree.CXXMemberCallExpr):
        self.write(node.subnodes[0], "(")
        self.comma_list(node.subnodes[1:])
        self.write(")")
        #if len(node.subnodes) > 1:
            #self.write(node.subnodes[0], "(", node.subnodes[1], ")")
        #else:
            #self.write(node.subnodes[0])

    def visit_CallExpr(self, node: tree.CallExpr):
        self.write(node.subnodes[0])
        if len(node.subnodes) > 1:
            self.write("(")
            self.comma_list(node.subnodes[1:])
            self.write(")")

    def visit_CXXOperatorCallExpr(self, node: tree.CXXOperatorCallExpr):
        self.write(node.left, " ", node.op, " ")
        if node.right is not None:
            self.write(node.right)

    def visit_CXXBoolLiteralExpr(self, node: tree.CXXBoolLiteralExpr):
        self.write("true" if node.value == "True" else "false")

    def visit_CXXFunctionalCastExpr(self, node: tree.CXXFunctionalCastExpr):
        self.write(node.type, "(", node.subnodes[0], ")")

    def visit_CXXTemporaryObjectExpr(self, node: tree.CXXTemporaryObjectExpr):
        self.write(node.type, "(")
        self.comma_list(node.subnodes)
        self.write(")")

    def visit_ForStmt(self, node: tree.ForStmt):
        self.write("for (", node.subnodes[0])
        if not self.result[-1].strip().endswith(";"):
            self.write("; ")
        self.write(node.subnodes[1], "; ",
                   node.subnodes[2], ")\n", node.subnodes[3])

    def visit_WhileStmt(self, node: tree.ForStmt):
        self.write("while (", node.subnodes[0], ")\n", node.subnodes[1])

    def visit_ContinueStmt(self, node: tree.ContinueStmt):
        self.write("continue;")
        self.newline(extra=1)

    def visit_EnumConstantDecl(self, node: tree.EnumConstantDecl):
        self.write(node.name)
        if node.subnodes is not None and len(node.subnodes) > 0:
            self.write(" = ", node.subnodes[0])

    def visit_EnumDecl(self, node: tree.EnumDecl):
        self.write("enum ", node.name, " {\n")
        self.comma_list(node.subnodes)
        self.write("};")
        self.newline(extra=1)

    def visit_ImplicitValueInitExpr(self, node: tree.ImplicitValueInitExpr):
        pass

    def visit_CXXConversionDecl(self, node: tree.CXXConversionDecl):
        self.write(node.name,  node.subnodes[0])

    def visit_EmptyDecl(self, node: tree.EmptyDecl):
        self.write(";")

    def visit_CStyleCastExpr(self, node: tree.CStyleCastExpr):
        def get_deeper_member_name(node):
            if node.subnodes is not None and len(node.subnodes) > 0:
                return get_deeper_member_name(node.subnodes[0])
            elif node.__class__.__name__ != "MemberExpr":
                return None
            else:
                return node.name
        self.write(node.type, get_deeper_member_name(node))

    def visit_CXXThisExpr(self, node: tree.CXXThisExpr):
        self.write("this")

    def visit_FriendDecl(self, node: tree.FriendDecl):
        self.write("friend ", node.type, ";")
        self.newline(extra=1)

    def visit_CXXStdInitializerListExpr(self, node: tree.CXXStdInitializerListExpr):
        self.write("{")
        self.comma_list(node.subnodes)
        self.write("}")