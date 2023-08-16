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

    def statement(self, node, *params, **kw):
        self.newline(node)
        self.write(*params)

    def comma_list(self, items, trailing=False):
        # set_precedence(Precedence.Comma, *items)
        if items:
            for idx, item in enumerate(items):
                self.write(", " if idx else "", item)
            self.write("," if trailing else "")

    # Statements

    def visit_TranslationUnit(self, node: tree.TranslationUnit):
        if node.stmts is not None:
            for c in node.stmts:
                self.write(c)

    def visit_InitListExpr(self, node: tree.InitListExpr):
        self.write("{")
        self.comma_list(node.values)
        self.write("}")
        self.newline(extra=1)

    def visit_CXXRecordDecl(self, node: tree.CXXRecordDecl):
        self.write(node.kind, " ")
        self.write(node.name)
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

    def visit_CXXConstructorDestructorDecl(self, node):
        if getattr(node, "virtual", None):
            self.write("virtual ")

        self.write(node.name)
        self.write("(")
        if getattr(node, "parameters", None):
            self.comma_list(node.parameters)
        self.write(")")

        if node.noexcept:
            self.write(" ", node.noexcept)

        if getattr(node, 'initializers', None):
            self.write(" : ")
            self.comma_list(node.initializers)

        if node.body is not None:
            self.write(node.body)
        else:
            if node.defaulted:
                self.write(" = ", node.defaulted)
            self.write(";")
        self.newline(extra=1)

    def visit_CXXConstructorDecl(self, node: tree.CXXConstructorDecl):
        self.visit_CXXConstructorDestructorDecl(node)

    def visit_CXXConstructExpr(self, node: tree.CXXConstructExpr):
        if node.subnodes is not None and len(node.subnodes) > 0:
            self.write(node.subnodes[0])

    def visit_CXXCtorInitializer(self, node: tree.CXXCtorInitializer):
        self.write(node.name)
        self.write('(')
        self.comma_list(node.args)
        self.write(')')

    def visit_CXXDestructorDecl(self, node: tree.CXXDestructorDecl):
        self.visit_CXXConstructorDestructorDecl(node)

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
        self.write(self.visit_type_helper(node.name or "", node.type))
        if node.default:
            self.write(" = ", node.default)

    def visit_ExprWithCleanups(self, node: tree.ExprWithCleanups):
        self.write(node.subnodes[0])

    def visit_DeclRefExpr(self, node: tree.DeclRefExpr):
        if node.name.startswith("operator"):
            self.write(node.name.replace("operator", "", 1))
        else:
            self.write(node.name)

    def visit_MaterializeTemporaryExpr(self, node: tree.MaterializeTemporaryExpr):
        self.write(node.expr)

    def visit_CXXBindTemporaryExpr(self, node: tree.CXXBindTemporaryExpr):
        self.write(node.subnodes[0])

    def visit_ImplicitCastExpr(self, node: tree.ImplicitCastExpr):
        # The cast is implicit, no need to pretty-print it.
        self.write(node.expr)

    def visit_AlignedAttr(self, node: tree.AlignedAttr):
        self.write("__attribute__((aligned")
        if node.size is not None:
            self.write("(", node.size, ")")
        self.write("))")

    def visit_AliasAttr(self, node: tree.AliasAttr):
        self.write("__attribute__((alias(\"", node.aliasee, "\")))")

    def visit_CleanupAttr(self, node: tree.CleanupAttr):
        self.write("__attribute__((cleanup(", node.func, ")))")

    def visit_DeprecatedAttr(self, node: tree.DeprecatedAttr):
        self.write("__attribute__((deprecated")
        if node.msg is not None:
            self.write("(\"", node.msg, "\")")
        self.write("))")

    def visit_UnavailableAttr(self, node: tree.UnavailableAttr):
        self.write("__attribute__((unavailable")
        if node.msg is not None:
            self.write("(\"", node.msg, "\")")
        self.write("))")

    def visit_PackedAttr(self, node: tree.PackedAttr):
        self.write("__attribute__((packed))")

    def visit_RetainAttr(self, node: tree.RetainAttr):
        self.write("__attribute__((retain)))")

    def visit_SectionAttr(self, node: tree.SectionAttr):
        self.write("__attribute__((section(\"", node.section, "\")))")

    def visit_TLSModelAttr(self, node: tree.TLSModelAttr):
        self.write("__attribute__((tls_model(\"", node.tls_model, "\")))")

    def visit_UnusedAttr(self, node: tree.UnusedAttr):
        self.write("__attribute__((unused)))")

    def visit_UsedAttr(self, node: tree.UsedAttr):
        self.write("__attribute__((used)))")

    def visit_UninitializedAttr(self, node: tree.UninitializedAttr):
        self.write("__attribute__((uninitialized)))")

    def visit_VisibilityAttr(self, node: tree.VisibilityAttr):
        self.write("__attribute__((visibility(\"", node.visibility, "\")))")

    def visit_WeakAttr(self, node: tree.WeakAttr):
        self.write("__attribute__((weak)))")

    def visit_FinalAttr(self, node: tree.WeakAttr):
        self.write("final")

    def visit_VarDecl(self, node: tree.VarDecl):
        if node.storage_class:
            self.write(node.storage_class, " ")

        if node.attributes:
            for attribute in node.attributes:
                self.write(attribute, " ")

        if node.tls:
            tls_mode = {'dynamic': 'thread_local',
                        'static': '__thread'}
            self.write(tls_mode[node.tls], " ")

        if node.implicit and node.referenced:
            self.write(node.subnodes[0])
        else:
            self.write(self.visit_type_helper(node.name, node.type))
            if node.init_mode:
                if node.init_mode == 'call':
                    self.write("(", node.init, ")")
                else:
                    self.write(" = ", node.init)

    def visit_ExprStmt(self, node: tree.ExprStmt):
        self.write(node.expr, ";")

    def visit_ConstrainedExpression(self, node: tree.ConstrainedExpression):
        self.write('"', node.constraint, '"', "(", node.expr, ")")

    def visit_GCCAsmStmt(self, node: tree.GCCAsmStmt):
        def escape(s):
            return s.replace('\n', r'\n').replace('\t', r'\t')
        command = "asm goto" if node.labels else "asm"
        self.write(command + "(", '"', escape(node.string), '"')
        if node.output_operands or node.input_operands or node.clobbers or node.labels:
            self.write(":")
            self.comma_list(node.output_operands)

        if node.input_operands or node.clobbers or node.labels:
            self.write(":")
            self.comma_list(node.input_operands)

        if node.clobbers or node.labels :
            self.write(": ")
            self.comma_list(map('"{}"'.format, node.clobbers))
        if node.labels:
            self.write(": ")
            self.comma_list(node.labels)
        self.write(");")

    def visit_FieldDecl(self, node: tree.FieldDecl):
        if node.attributes:
            for attribute in node.attributes:
                self.write(attribute, " ")

        self.write(self.visit_type_helper(node.name, node.type))
        if node.init:
            self.write(" = ", node.init)
        self.write(";")
        self.newline(extra=1)

    def visit_type_helper(self, current_expr, current_type):
        if isinstance(current_type, tree.BuiltinType):
            return "{} {}".format(current_type.name, current_expr)
        if isinstance(current_type, tree.ElaboratedType):
            if isinstance(current_type.type, tree.RecordType):
                return"struct " + self.visit_type_helper(current_expr, current_type.type)
            else:
                return self.visit_type_helper(current_expr, current_type.type)
        if isinstance(current_type, tree.FunctionProtoType):
            parameter_types = current_type.parameter_types or []
            argument_types = ', '.join(self.visit_type_helper("", ty)
                                       for ty in parameter_types)
            return self.visit_type_helper(
                    "{}({})".format(current_expr, argument_types),
                    current_type.return_type)
        if isinstance(current_type, tree.ParenType):
            return self.visit_type_helper("({})".format(current_expr),
                                             current_type.type)
        if isinstance(current_type, tree.DecayedType):
            return self.visit_type_helper(current_expr, current_type.type)
        if isinstance(current_type, tree.PointerType):
            return self.visit_type_helper("*{}".format(current_expr),
                                             current_type.type)
        if isinstance(current_type, tree.QualType):
            qualified_type = current_type.type
            if current_type.qualifiers is None:
                return self.visit_type_helper(current_expr, qualified_type)

            # west const
            if isinstance(qualified_type, (tree.BuiltinType, tree.RecordType)):
                return "{} {}".format(current_type.qualifiers,
                                      self.visit_type_helper(current_expr,
                                                                qualified_type))
            # east const
            else:
                return self.visit_type_helper(
                        "{} {}".format(current_type.qualifiers,
                                       current_expr),
                        qualified_type)

        if isinstance(current_type, tree.RecordType):
            return "{} {}".format(current_type.name, current_expr)

        if isinstance(current_type, tree.ConstantArrayType):
            return self.visit_type_helper("{} [{}]".format(current_expr,
                                                              current_type.size),
                                             current_type.type)
        if isinstance(current_type, tree.IncompleteArrayType):
            return self.visit_type_helper("{} []".format(current_expr),
                                          current_type.type)
        if isinstance(current_type, tree.VectorType):
            vector_type = self.visit_type_helper("", current_type.type)
            return "__attribute__((vector_size({}))) {} {}".format(current_type.size, vector_type, current_expr)

        if isinstance(current_type, tree.LValueReferenceType):
            return self.visit_type_helper("& " + current_expr, current_type.type)

        if isinstance(current_type, tree.RValueReferenceType):
            return self.visit_type_helper("&& " + current_expr, current_type.type)

        raise NotImplementedError(current_type)

    def visit_TypedefDecl(self, node: tree.TypedefDecl):
        expr = self.visit_type_helper(node.name, node.type)
        self.write("typedef ", expr, ";")

    def visit_BuiltinType(self, node: tree.BuiltinType):
        self.write(node.name)

    def visit_ElaboratedType(self, node: tree.BuiltinType):
        if node.qualifiers:
            raise NotImplementedError()

        # FIXME: we probably don't want to support this.
        if isinstance(node.type, tree.RecordType):
            self.write("struct ")
        self.write(node.type)

    def visit_FunctionProtoType(self, node: tree.FunctionProtoType):
        self.write(node.return_type, "(")
        self.comma_list(node.parameter_types)
        self.write(")")

    def visit_ParenType(self, node: tree.ParenType):
        self.write("(", node.type, ")")

    def visit_PointerType(self, node: tree.PointerType):
        self.write(node.type, "*")

    def visit_RecordType(self, node: tree.RecordType):
        self.write(node.name)

    def visit_EnumType(self, node: tree.EnumType):
        self.write(node.name)

    def visit_LValueReferenceType(self, node: tree.LValueReferenceType):
        self.write(node.type)

    def visit_RValueReferenceType(self, node: tree.RValueReferenceType):
        self.write(node.type)

    def visit_ConstantArrayType(self, node: tree.ConstantArrayType):
        self.write(node.type, "[", node.size, "]")

    def visit_IncompleteArrayType(self, node: tree.IncompleteArrayType):
        self.write(node.type, "[]")

    def visit_DecayedType(self, node: tree.DecayedType):
        self.write(node.type)

    def visit_TypeRef(self, node: tree.TypeRef):
        self.write(node.name)

    def visit_QualType(self, node: tree.QualType):
        # west const
        if isinstance(node.type, tree.BuiltinType):
            if node.qualifiers:
                self.write(node.qualifiers, " ")
            self.write(node.type)
        # east const
        else:
            self.write(node.type)
            if node.qualifiers:
                self.write(" ", node.qualifiers)

    def visit_CXXMethodDecl(self, node: tree.CXXMethodDecl):
        if node.virtual:
            self.write("virtual ")

        self.write(node.return_type, " ")
        self.write(node.name)
        self.write("(")
        if node.parameters:
            self.comma_list(node.parameters)
        self.write(")")

        if node.const:
            self.write(" const")

        if node.ref_qualifier:
            self.write(" ", node.ref_qualifier)

        if node.noexcept:
            self.write(" ", node.noexcept)

        if node.final:
            self.write(" ", node.final)

        if node.body:
            self.write(node.body)
        else:
            if node.defaulted:
                self.write(" = ", node.defaulted)
            self.write(";")
        self.newline(extra=1)

    def visit_PureVirtual(self, node: tree.PureVirtual):
        self.write("0")

    def visit_Default(self, node: tree.Default):
        self.write("default")

    def visit_Delete(self, node: tree.Delete):
        self.write("delete")

    def visit_FunctionDecl(self, node: tree.FunctionDecl):
        if node.storage:
            self.write(node.storage, " ")

        if node.inline:
            self.write("inline ")

        self.write(node.return_type, " ")
        self.write(node.name)

        self.write("(")
        if node.parameters:
            self.comma_list(node.parameters)
        if node.variadic:
            if node.parameters:
                self.write(", ")
            self.write("...")
        self.write(")")

        if node.exception:
            self.write(" ", node.exception)

        if node.body:
            self.write(node.body)
        else:
            # forward declaration
            self.write(";")
        self.newline(extra=1)

    def visit_IntegerLiteral(self, node: tree.IntegerLiteral):
        suffixes = {
                'unsigned int': 'u',
                'long': 'l',
                'unsigned long': 'ul',
                'long long': 'll',
                'unsigned long long': 'ull'}
        self.write(node.value + suffixes.get(node.type.name, ''))

    def visit_FloatingLiteral(self, node: tree.FloatingLiteral):
        suffixes = {'float': 'f', 'long double': 'l'}
        # FIXME: we may loose precision by choosing this format
        self.write('{:g}{}'.format(float(node.value),
                                   suffixes.get(node.type.name, '')))

    def visit_CharacterLiteral(self, node: tree.CharacterLiteral):
        if node.value.isprintable():
            self.write("'{}'".format(node.value))
        else:
            c = ord(node.value)
            if c < 9:
                self.write("'\\{:o}'".format(c))
            else:
                self.write("'\\0x{:02X}'".format(c))

    def visit_StringLiteral(self, node: tree.StringLiteral):
        self.write(node.value)

    def visit_CXXNullPtrLiteralExpr(self, node: tree.CXXNullPtrLiteralExpr):
        self.write("nullptr")

    def visit_UnaryExprOrTypeTraitExpr(self, node: tree.UnaryExprOrTypeTraitExpr):
        self.write(node.name, "(")
        self.visit(node.type if node.type is not None else node.expr)
        self.write(")")

    def visit_BinaryOperator(self, node: tree.BinaryOperator):
        self.write(node.lhs)
        self.write(" ", node.opcode, " ")
        self.write(node.rhs)

    def visit_CompoundAssignOperator(self, node: tree.CompoundAssignOperator):
        self.write(node.lhs)
        self.write(" ", node.opcode, " ")
        self.write(node.rhs)

    def visit_UnaryOperator(self, node: tree.UnaryOperator):
        if node.postfix == "True":
            self.write(node.expr, node.opcode)
        else:
            self.write(node.opcode, node.expr)

    def visit_ConditionalOperator(self, node: tree.ConditionalOperator):
        self.write(node.cond, "?", node.true_expr, ":",
                   node.false_expr)

    def visit_ArraySubscriptExpr(self, node: tree.ArraySubscriptExpr):
        self.write(node.base, "[", node.index, "]")

    def visit_DeclStmt(self, node: tree.DeclStmt):
        # FIXME: could be improved to support multiple decl in one statement
        for decl in node.decls:
            self.write(decl, ";" if isinstance(decl, tree.VarDecl) else "")

    # ReturnStmt(expression? expression)
    def visit_ReturnStmt(self, node: tree.ReturnStmt):
        self.write("return ", node.value or "", ";\n")

    def visit_NullStmt(self, node: tree.NullStmt):
        self.write(";")

    def visit_IfStmt(self, node: tree.IfStmt):
        self.write("if (", node.cond, ")\n")
        self.write(node.true_body)
        if node.false_body is not None:
            self.write("else ", node.false_body)

    # BlockStatement(statement* statements)
    def visit_CompoundStmt(self, node: tree.CompoundStmt):
        self.write("{", "\n")
        if node.stmts is not None:
            for statement in node.stmts:
                self.write(statement)
                if isinstance(statement, tree.Expression):
                    self.write(";\n")
        self.write("}")
        self.newline(extra=1)

    # prefix_operators, "postfix_operators", "qualifier", "selectors
    def visit_ParenExpr(self, node: tree.ParenExpr):
        self.write("(", node.expr, ")")


    def visit_SwitchStmt(self, node: tree.SwitchStmt):
        self.write("switch (", node.cond, ")\n", node.body)

    def visit_CaseStmt(self, node: tree.CaseStmt):
        self.write("case ", node.pattern, ":\n", node.stmt)

    def visit_BreakStmt(self, node: tree.BreakStmt):
        self.write("break;\n")

    def visit_MemberExpr(self, node: tree.MemberExpr):
        if node.expr:
            self.write(node.expr, node.op, node.name)
        else:
            self.write(node.name)

    def visit_ConstantExpr(self, node: tree.ConstantExpr):
        self.write(node.value)

    def visit_DefaultStmt(self, node: tree.DefaultStmt):
        self.write("default:\n", node.stmt)

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
        self.write(node.callee)
        self.write("(")
        self.comma_list(node.args)
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

    def visit_DeclOrExpr(self, node: tree.DeclOrExpr):
        if node.decl:
            # FIXME: handle multiple declarations
            self.write(node.decl[0])
        else:
            self.write(node.expr)

    def visit_ForStmt(self, node: tree.ForStmt):
        self.write("for (",
                   node.init or '', "; ",
                   node.cond or '', "; ",
                   node.inc or '', ")\n",
                   node.body)

    def visit_LabelStmt(self, node: tree.LabelStmt):
        self.write(node.name, ":\n", node.stmt)

    def visit_GotoStmt(self, node: tree.GotoStmt):
        self.write("goto", node.target, ";")

    def visit_CXXForRangeStmt(self, node: tree.CXXForRangeStmt):
        assert len(node.subnodes) == 7
        self.write("for (", node.subnodes[-2])
        self.write(": ", node.subnodes[0], ")\n")
        self.write(node.subnodes[-1])

    def visit_WhileStmt(self, node: tree.WhileStmt):
        self.write("while (", node.cond, ")")
        self.newline()
        self.write(node.body)

    def visit_DoStmt(self, node: tree.DoStmt):
        self.write("do\n", node.body, "while (", node.cond, ");\n")

    def visit_ContinueStmt(self, node: tree.ContinueStmt):
        self.write("continue;")
        self.newline(extra=1)

    def visit_EnumConstantDecl(self, node: tree.EnumConstantDecl):
        if node.init is None:
            self.write(node.name)
        else:
            self.write(node.name, " = ", node.init)

    def visit_EnumDecl(self, node: tree.EnumDecl):
        self.write("enum ", node.name or "", " {\n")
        self.comma_list(node.fields)
        self.write("};")
        self.newline(extra=1)

    def visit_ImplicitValueInitExpr(self, node: tree.ImplicitValueInitExpr):
        pass

    def visit_CXXConversionDecl(self, node: tree.CXXConversionDecl):
        self.write(node.name,  node.subnodes[0])

    def visit_EmptyDecl(self, node: tree.EmptyDecl):
        self.write(";")

    def visit_CStyleCastExpr(self, node: tree.CStyleCastExpr):
        self.write("(", node.type, ")", node.expr)

    def visit_CXXThisExpr(self, node: tree.CXXThisExpr):
        self.write("this")

    def visit_FriendDecl(self, node: tree.FriendDecl):
        self.write("friend ", node.type, ";")
        self.newline(extra=1)

    def visit_CXXStdInitializerListExpr(self, node: tree.CXXStdInitializerListExpr):
        self.write("{")
        self.comma_list(node.subnodes)
        self.write("}")

    def visit_CXXNewExpr(self, node: tree.CXXNewExpr):
        markers = "[]" if node.is_array else "()"
        self.write("new ", node.type, markers[0])
        if node.is_array:
            size, *initializers = node.args
            self.write(size, markers[1])
            if initializers:
                initializer, = initializers
                self.write(initializer)
        else:
            self.comma_list(node.args)
            self.write(markers[1])

    def visit_Throw(self, node: tree.Throw):
        self.write("throw(")
        self.comma_list(node.args)
        self.write(")")

    def visit_NoExcept(self, node: tree.NoExcept):
        self.write("noexcept")
        if node.repr:
            self.write("(", node.repr, ")")

