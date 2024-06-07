import copy
import json
import re
import shutil
import subprocess
import sys
import os
from typing import (Dict, List, Set, Tuple)



from . import util
from . import tree

ENABLE_DEBUG_SUPPORT = False

LLVM_VERSION = "14"

clang = shutil.which(f"clang-{LLVM_VERSION}")
if not clang:
    print(f"clang-{LLVM_VERSION} not found.", file=sys.stderr)
    raise RuntimeError(f"clang-{LLVM_VERSION} not found.")

llvm_config = shutil.which(f"llvm-config-{LLVM_VERSION}")
if not llvm_config:
    print(f"llvm-config-{LLVM_VERSION} not found.", file=sys.stderr)
    raise RuntimeError(f"llvm-config-{LLVM_VERSION} not found.")

preprocess_command = [
    clang, "-x", "c++", "-std=c++14",
    "-E", "-"]

module_dir = os.path.dirname(__file__)
plugin_path = os.path.join(module_dir, "JSONTypeDumper.so")



def rebuild_clang_plugin():
    plugin_src = os.path.join(module_dir, "JSONDumpTypes.cpp")

    src_mtime = os.path.getmtime(plugin_src)
    try:
        plugin_mtime = os.path.getmtime(plugin_path)
    except Exception:
        plugin_mtime = 0

    if src_mtime > plugin_mtime:
        print(f"(Re)building JSONDumpTypes clang plugin", file=sys.stderr)
        raw_llvm_compile_flags = subprocess.check_output(
                [llvm_config, '--cxxflags', '--ldflags'])
        llvm_compile_flags = raw_llvm_compile_flags.split()
        subprocess.check_call([
            clang,
            plugin_src,
            # "-O0", "-g", "-fno-omit-frame-pointer", "-fno-optimize-sibling-calls",
            # "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-fno-optimize-sibling-calls",
            "-O2",
            "-fPIC",
            "-shared",
            "-o", plugin_path] + llvm_compile_flags)
    else:
        print(f"No need to (Re)build JSONDumpTypes clang plugin",
              file=sys.stderr)


rebuild_clang_plugin()


def parse_debug(method):
    global ENABLE_DEBUG_SUPPORT

    if ENABLE_DEBUG_SUPPORT:
        def _method(self, node):
            if not hasattr(self, 'recursion_depth'):
                self.recursion_depth = 0

            if self.debug:
                depth = "%02d" % (self.recursion_depth,)
                token = self.get_node_source_code(node)
                start_value = self.get_node_source_code(node)
                name = method.__name__
                sep = ("-" * self.recursion_depth)
                e_message = ""

                print("%s %s> %s(%s)" % (depth, sep, name, token))

                self.recursion_depth += 1

                try:
                    r = method(self, node)

                except CppSyntaxError as e:
                    e_message = e.description
                    raise

                except Exception as e:
                    e_message = e.__repr__()
                    raise

                finally:
                    if self.stack:
                        token = self.get_node_source_code(self.stack[-1])
                    else:
                        token = ""
                    print(f"{depth} <{sep} {name}({start_value}, {token}) {e_message}")
                    self.recursion_depth -= 1
            else:
                self.recursion_depth += 1
                try:
                    r = method(self, node)
                finally:
                    self.recursion_depth -= 1

            return r

        return _method

    else:
        return method

# ------------------------------------------------------------------------------
# ---- Parsing exception ----

class CppParserBaseException(Exception):
    def __init__(self, message=''):
        super(CppParserBaseException, self).__init__(message)

class CppSyntaxError(CppParserBaseException):
    def __init__(self, description, at=None):
        super(CppSyntaxError, self).__init__()

        self.description = description
        self.at = at

class CppParserError(CppParserBaseException):
    pass

# ------------------------------------------------------------------------------
# ---- Parser class ----

class Parser(object):
    operator_precedence = [ set(('||',)),
                            set(('&&',)),
                            set(('|',)),
                            set(('^',)),
                            set(('&',)),
                            set(('==', '!=')),
                            set(('<', '>', '>=', '<=', 'instanceof')),
                            set(('<<', '>>', '>>>')),
                            set(('+', '-')),
                            set(('*', '/', '%'))]

    def __init__(self, cpp_code: str = None, filepath: str = None,
                 compile_command: Dict[str, str] = None):
        """
        Initialize the parser. Either cpp_code or compile_command must be not None.
        Depending on which one, a different init function is called
        """
        assert ((compile_command is not None and cpp_code is None and filepath is None)or
                (compile_command is None and cpp_code is not None and filepath is not None))
        self.type_informations = {}
        self.decl_informations = {}
        self.expr_informations = {}
        self.asm_informations = {}
        self.template_instances = {}
        self.attr_informations = {}
        self.stack = []
        self.debug = False
        self.anonymous_types = {}
        self.tu = {}
        if cpp_code is not None:
            self._init_direct(cpp_code, filepath)
        elif compile_command is not None:
            self._init_compile_commands(compile_command)

    def _init_compile_commands(self, compile_command: Dict[str, str]):
        """
        Initialize the parser using the command given by the external
        compile_commands.json database
        """
        # print(f"Parser._init_compile_commands", file=sys.stderr)
        workdir = compile_command["directory"]
        if "command" in compile_command:
            # split he command into its arguments
            compile_command["arguments"] = compile_command["command"].split(" ")
        arguments = copy.copy(compile_command["arguments"])

        if arguments[0].endswith("cc"):
            arguments[1:1] =["-x", "c"]
        elif arguments[0].endswith("c++"):
            arguments[1:1] =["-x", "c++"]
        arguments[0] = clang
        # remove -o /foo/bar.o from command. We will read the standard output stream
        if "-o" in arguments:
            i = arguments.index("-o")
            arguments = arguments[:i] + arguments[i+2:]

        arguments.remove("-c")
        filepath = arguments[-1]
        preprocess_command = copy.copy(arguments)
        preprocess_command.insert(-1, "-E")
        if ENABLE_DEBUG_SUPPORT:
            print(f"\npreprocess_command:\n{' '.join(preprocess_command)}\n",
                  file=sys.stderr)
        try:
            self.filepath = filepath
            preprocess = subprocess.run(
                preprocess_command,
                capture_output=True,
                check=True,
                cwd=workdir)
        except subprocess.CalledProcessError as e:
            print(f"While handling {self.filepath},\n", file=sys.stderr)
            print(f"Preprocessing error {e.returncode}:\n{e.stderr.decode()}",
                  file=sys.stderr)
            raise
        preprocess_stdout_data = preprocess.stdout
        # if ENABLE_DEBUG_SUPPORT:
        #     print(f"\npreprocess_stdout_data:\n{preprocess_stdout_data.decode()}\n",
        #           file=sys.stderr)
        preprocess_stderr_data = preprocess.stderr
        arguments.pop()
        arguments.extend([f"-fplugin={plugin_path}", "-fsyntax-only", "-"])
        if ENABLE_DEBUG_SUPPORT:
            print(f"process_command:\n{' '.join(arguments)}\n\n", file=sys.stderr)
        try:
            p = subprocess.run(
                arguments,
                capture_output=True,
                input=preprocess_stdout_data,
                check=True,
                cwd=workdir)
        except subprocess.CalledProcessError as e:
            print(f"While handling {self.filepath},\n", file=sys.stderr)
            print(f"Parsing error {e.returncode}:\n{e.stderr.decode()}",
                  file=sys.stderr)
            print(f"    command was:\n{' '.join(arguments)}", file=sys.stderr)
            raise
        stdout_data = p.stdout
        if ENABLE_DEBUG_SUPPORT:
            print(f"stdout_data:\n{stdout_data.decode()}\n\n", file=sys.stderr)
        stderr_data = p.stdout
        # print(stderr_data.decode(), file=sys.stderr)
        try:
            self.tu = json.loads(stdout_data.decode())
        except json.decoder.JSONDecodeError as e:
            print(f"\npreprocess_stdout_data:\n{preprocess_stdout_data.decode()}\n",
                  file=sys.stderr)
            print(f"json decode error on output of:\n{arguments}\n\n", file=sys.stderr)
            raise
        self.source_code = preprocess_stdout_data.decode()

    def _init_direct(self, cpp_code: str, filepath: str = None):
        """
        Initialize the parser by using compiled commands  forged internally
        """
        try:
            self.filepath = filepath
            # TODO replace in commands below the include path by those given by the
            # C++ project build system
            preprocess = subprocess.run(
                preprocess_command,
                capture_output=True,
                input=cpp_code.encode(),
                check=True)
        except subprocess.CalledProcessError as e:
            if self.filepath is not None:
                print(f"While handling {self.filepath},\n", file=sys.stderr)
            print(f"Preprocessing error {e.returncode}:\n{e.stderr.decode()}", file=sys.stderr)
            raise
        preprocess_stdout_data = preprocess.stdout
        # if ENABLE_DEBUG_SUPPORT:
        #     print(f"\npreprocess_stdout_data:\n{preprocess_stdout_data.decode()}\n",
        #           file=sys.stderr)
        preprocess_stderr_data = preprocess.stderr
        process_command = [
            shutil.which("clang"), "-x", "c++", "-std=c++14",
            "-fplugin={}".format(plugin_path),
            "-fsyntax-only", "-"]
        # if ENABLE_DEBUG_SUPPORT:
        #     print(f"process_command:\n{' '.join(process_command)}\n\n", file=sys.stderr)
        try:
            p = subprocess.run(
                process_command,
                capture_output=True,
                input=preprocess_stdout_data,
                check=True)
        except subprocess.CalledProcessError as e:
            if self.filepath is not None:
                print(f"While handling {self.filepath},\n", file=sys.stderr)
            print(f"Parsing error {e.returncode}:\n{e.stderr.decode()}", file=sys.stderr)
            print(f"    command was:\n{' '.join(process_command)}", file=sys.stderr)
            raise
        stdout_data = p.stdout
        # if ENABLE_DEBUG_SUPPORT:
        #     print(f"stdout_data:\n{stdout_data.decode()}\n\n", file=sys.stderr)
        stderr_data = p.stdout
        # print(stderr_data.decode(), file=sys.stderr)
        self.tu = json.loads(stdout_data.decode())
        self.source_code = preprocess_stdout_data.decode()

# ------------------------------------------------------------------------------
# ---- Debug control ----

    def set_debug(self, debug=True):
        global ENABLE_DEBUG_SUPPORT
        self.debug = debug
        ENABLE_DEBUG_SUPPORT = self.debug

# ------------------------------------------------------------------------------
# ---- Parsing entry point ----

    def parse(self) -> tree.TranslationUnit:
        self.parse_type_summary(self.tu["TypeSummary"])
        return self.parse_TranslationUnit(self.tu["Content"])

# ------------------------------------------------------------------------------
# ---- Helper methods ----

    #def illegal(self, description, at=None):
        #if not at:
            #at = self.tokens.look()

        #raise CppSyntaxError(description, at)

    #def accept(self, *accepts):
        #last = None

        #if len(accepts) == 0:
            #raise CppParserError("Missing acceptable values")

        #for accept in accepts:
            #token = next(self.tokens)
            #if isinstance(accept, six.string_types) and (
                    #not token.value == accept):
                #self.illegal("Expected '%s'" % (accept,))
            #elif isinstance(accept, type) and not isinstance(token, accept):
                #self.illegal("Expected %s" % (accept.__name__,))

            #last = token

        #return last.value

    #def would_accept(self, *accepts):
        #if len(accepts) == 0:
            #raise CppParserError("Missing acceptable values")

        #for i, accept in enumerate(accepts):
            #token = self.tokens.look(i)

            #if isinstance(accept, six.string_types) and (
                    #not token.value == accept):
                #return False
            #elif isinstance(accept, type) and not isinstance(token, accept):
                #return False

        #return True

    #def try_accept(self, *accepts):
        #if len(accepts) == 0:
            #raise CppParserError("Missing acceptable values")

        #for i, accept in enumerate(accepts):
            #token = self.tokens.look(i)

            #if isinstance(accept, six.string_types) and (
                    #not token.value == accept):
                #return False
            #elif isinstance(accept, type) and not isinstance(token, accept):
                #return False

        #for i in range(0, len(accepts)):
            #next(self.tokens)

        #return True

    #def build_binary_operation(self, parts, start_level=0) -> tree.BinaryOperation:
        #if len(parts) == 1:
            #return parts[0]

        #operands = list()
        #operators = list()

        #i = 0

        #for level in range(start_level, len(self.operator_precedence)):
            #for j in range(1, len(parts) - 1, 2):
                #if parts[j].operator in self.operator_precedence[level]:
                    #operand = self.build_binary_operation(parts[i:j], level + 1)
                    #operator = parts[j]
                    #i = j + 1

                    #operands.append(operand)
                    #operators.append(operator)

            #if operands:
                #break

        #operand = self.build_binary_operation(parts[i:], level + 1)
        #operands.append(operand)

        #operation = operands[0]

        #for operator, operandr in zip(operators, operands[1:]):
            #operation = tree.BinaryOperation(operandl=operation)
            #operation.operator = operator
            #operation.operandr = operandr

        #return operation

    #def is_annotation(self, i=0):
        #""" Returns true if the position is the start of an annotation application
        #(as opposed to an annotation declaration)

        #"""

        #return (isinstance(self.tokens.look(i), Annotation)
                #and not self.tokens.look(i + 1).value == 'interface')

    #def is_annotation_declaration(self, i=0):
        #""" Returns true if the position is the start of an annotation application
        #(as opposed to an annotation declaration)

        #"""

        #return (isinstance(self.tokens.look(i), Annotation)
                #and self.tokens.look(i + 1).value == 'interface')

    def parse_subnodes(self, node, *, keep_empty=False, key='inner'):
        subnodes = node.get(key)
        if subnodes is None:
            return []
        if ENABLE_DEBUG_SUPPORT:
            assert len(subnodes) > 0
        result = [self.parse_node(c) for c in subnodes]
        result = [el for el in result if el is not None]
        if keep_empty:
            return result
        else:
            return [c for c in result if c is not None]

    def collect_comment(self, node) -> str:
        if node['kind'] == 'TextComment':
            return node['text']
        return " ".join([self.collect_comment(subnode) for subnode in node['inner']])

    def get_node_source_code(self, node) -> str:
        begin = node.get('range', {}).get('begin', {}).get('offset')
        end = node.get('range', {}).get('end', {}).get('offset')
        if begin is None or end is None:
            return ''
        end_offset = node['range']['end']['tokLen']
        return self.source_code[begin:end + end_offset]
# ------------------------------------------------------------------------------
# ---- Parsing methods ----

# ------------------------------------------------------------------------------
# -- Identifiers --

    #@parse_debug
    #def parse_identifier(self):
        #return self.accept(Identifier)

    #@parse_debug
    #def parse_qualified_identifier(self):
        #qualified_identifier = list()

        #while True:
            #identifier = self.parse_identifier()
            #qualified_identifier.append(identifier)

            #if not self.try_accept('.'):
                #break

        #return '.'.join(qualified_identifier)

    #@parse_debug
    #def parse_qualified_identifier_list(self):
        #qualified_identifiers = list()

        #while True:
            #qualified_identifier = self.parse_qualified_identifier()
            #qualified_identifiers.append(qualified_identifier)

            #if not self.try_accept(','):
                #break

        #return qualified_identifiers

    def abort_visit(node):  # XXX: self?
        msg = (f"No defined parse handler for clang node of type `{node['kind']}`.\n"
               f"please define `parse_{node['kind']}` in cpplang/parser.py.")
        raise NotImplementedError(msg)

    @parse_debug
    def parse_node(self, node, abort=abort_visit) -> tree.Node:
        if node is None or 'kind' not in node:
            return None
        if ENABLE_DEBUG_SUPPORT:
            print(f"parse_node {node['kind']}", file=sys.stderr)
        if ('isImplicit' in node and node['isImplicit'] and not ('isReferenced' in node and node['isReferenced'])):
            return None
        elif 'isImplicit' in node and node['isImplicit'] and 'isReferenced' in node and node['isReferenced'] and node['kind'] == 'TypedefDecl':
            return None
        elif (('loc' in node and 'includedFrom' in node['loc'])
            or ('range' in node and 'includedFrom' in node['range']['begin'])
            or ('loc' in node and 'range' in node['loc']
                and 'includedFrom' in node['loc']['range']['begin'])
            or ('loc' in node and 'spellingLoc' in node['loc'] and 'includedFrom' in node['loc']['spellingLoc'])
            or ('range' in node and 'spellingLoc' in node['range']['begin'] and 'includedFrom' in node['range']['begin']['spellingLoc'])
            ):
            return None
        elif ((len(self.stack) == 0 and node.get('loc') and 'file' in node['loc']
             and node['loc']['file'] == "<stdin>")
                or len(self.stack) > 0):
            self.stack.append(node)
            parse_method = getattr(self, "parse_"+node['kind'], abort)
            try:
                result = parse_method(node)
            except NotImplementedError as e:
                print(f"Catching NotImplementedError in parse_node: {e}",
                      file=sys.stderr)
                result = None
                if ENABLE_DEBUG_SUPPORT:
                    raise
            except ValueError as e:
                print(f"Catching ValueError in parse_node: {e}",
                      file=sys.stderr)
                result = None
                if ENABLE_DEBUG_SUPPORT:
                    raise
            self.stack.pop()
            return result
        elif ((len(self.stack) == 0 and 'loc' in node
               and 'file' not in node['loc']
               and 'includedFrom' not in node['loc']
               and ('range' not in node['loc']
                    or 'includedFrom' not in node['loc']['range']['begin'])
               and ('spellingLoc' not in node['loc']
                    or 'includedFrom' not in node['loc']['spellingLoc']))
                or len(self.stack) > 0):
            self.stack.append(node)
            parse_method = getattr(self, "parse_"+node['kind'], abort)
            try:
                result = parse_method(node)
            except NotImplementedError as e:
                print(f"Catching NotImplementedError in parse_node: {e}",
                      file=sys.stderr)
                result = None
                if ENABLE_DEBUG_SUPPORT:
                    raise
            except ValueError as e:
                print(f"Catching ValueError in parse_node: {e}",
                      file=sys.stderr)
                result = None
                if ENABLE_DEBUG_SUPPORT:
                    raise
            self.stack.pop()
            return result
        else:
            return None

# ------------------------------------------------------------------------------
# -- Top level units --

    def parse_type_summary(self, node):
        for child in node:
            if 'node_inner' in child:
                inner_child, = child['node_inner']
                self.type_informations[child['node_id']] = inner_child
            elif 'node_id' in child:
                self.type_informations[child['node_id']] = child

            if 'isExplicit' in child:
                decl_info = self.decl_informations.setdefault(child['node_id'], {})
                decl_info["isExplicit"] = child['isExplicit']

            if 'inner' in child:
                self.parse_type_summary(child['inner'])

            if 'expr_inner' in child:
                assert 'node_inner' not in child
                self.stack.append(child)
                self.expr_informations[child['node_id']] = self.parse_subnodes(child, key='expr_inner')
                self.stack.pop()
            if 'asm_string' in child:
                asm_infos = {'asm_string': child['asm_string']}

                self.asm_informations[child['node_id']] = asm_infos

                if "output_constraints" in child:
                    output_constraints = {n['id']: n['constraint'] for n in
                                        child['output_constraints']}
                else:
                    output_constraints = {}

                if "input_constraints" in child:
                    input_constraints = {n['id']: n['constraint'] for n in
                                         child['input_constraints']}
                else:
                    input_constraints = {}

                asm_infos['output_constraints'] = output_constraints
                asm_infos['input_constraints'] = input_constraints

                asm_infos['clobbers'] = [x['clobber'] for x in
                                         child.get('clobbers', [])]
                asm_infos['labels'] = [x['label'] for x in child.get('labels',
                                                                     [])]

            # FIXME: skipping here means we're losing some attributes.
            if 'node_id' not in child:
                continue

            for field in ('aliasee', 'cleanup_function', 'deprecation_message',
                          'section_name', 'visibility', 'tls_model', 'name',
                          'source_index', "size_index", 'nmemb_index',
                          'priority', 'message', 'options', 'indices',
                          'archetype', 'fmt_index', 'vargs_index', 'count',
                          'offset', 'value',):
                if field not in child:
                    continue

                self.attr_informations.setdefault(child['node_id'], {})[field] = child[field]


    @parse_debug
    def parse_TranslationUnit(self, node) -> tree.TranslationUnit:
        assert node['kind'] == "TranslationUnitDecl"
        # print(f"parse_TranslationUnit {node}", file=sys.stderr)
        subnodes = self.parse_subnodes(node)
        return tree.TranslationUnit(stmts=self.as_statements(subnodes))

    def parse_class_inner(self, node):
        name = node.get('name')
        if name is None:
            loc = node['loc']
            where = '(unnamed struct at {}:{}:{})'.format(
                    loc.get('file', '<stdin>'),
                    loc.get('presumedLine', loc['line']),
                    loc['col'])
            name = '$_{}'.format(len(self.anonymous_types))
            self.anonymous_types[where] = name

        kind = node.get('tagUsed')
        complete = node.get('completeDefinition', '') and 'complete'

        bases = []
        for base in node.get('bases', ()):
            written_access = base['writtenAccess']
            if written_access == 'none':
                access_type = type(None)
            else:
                access_type = getattr(tree, written_access.capitalize())
            virtual = base.get('isVirtual') and tree.Virtual()
            bases.append(tree.Base(access_spec=access_type(),
                                   virtual=virtual,
                                   name=base['type']['qualType']))

        inner_nodes = self.parse_subnodes(node)

        # specific support for anonymous record through indirect field
        if inner_nodes:
            indirect_field_names = {n['name'] for n in node['inner']
                                    if n['kind'] == 'IndirectFieldDecl'}
            anonymous_records = [n for n in node['inner']
                                 if n['kind'] == 'CXXRecordDecl'
                                 if 'name' not in n]

            for inner_node in inner_nodes:
                if not isinstance(inner_node, tree.CXXRecordDecl):
                    continue
                if inner_node.name not in self.anonymous_types.values():
                    continue
                field_names = {field.name for field in inner_node.decls
                               if isinstance(field, tree.FieldDecl)}

                # Force the record name to empty to correctly represent indirect
                # fields.
                if field_names.issubset(indirect_field_names):
                    inner_node.name = ""
        return name, kind, bases, complete, inner_nodes

    @parse_debug
    def parse_CXXRecordDecl(self, node) -> tree.CXXRecordDecl:
        assert node['kind'] == "CXXRecordDecl"
        if 'isImplicit' in node and node['isImplicit']:
            return None

        name, kind, bases, complete, decls = self.parse_class_inner(node)

        return tree.CXXRecordDecl(name=name, kind=kind, bases=bases,
                                  complete=complete,
                                  decls=decls)

    @parse_debug
    def parse_RecordDecl(self, node) -> tree.RecordDecl:
        print(node)

    def parse_function_inner(self, node):
        inner_nodes = self.parse_subnodes(node)

        body, args, init, method_attrs, attrs = None, [], [], [], []

        type_info = self.type_informations[node['id']]
        noreturn = type_info.get('isNoReturn')
        if noreturn:
            attrs.append(tree.NoReturnAttr())

        for inner_node in inner_nodes:
            if isinstance(inner_node, tree.ParmVarDecl):
                args.append(inner_node)
            elif isinstance(inner_node, tree.TemplateArgument):
                continue
            elif isinstance(inner_node, tree.CXXCtorInitializer):
                init.append(inner_node)
            elif isinstance(inner_node, (tree.OverrideAttr, tree.FinalAttr)):
                method_attrs.append(inner_node)
            elif isinstance(inner_node, tree.CompoundStmt):
                assert body is None
                body = inner_node
            elif isinstance(inner_node, tree.RestrictAttr):
                # This is actually a MallocAttr, but clang represent it as a
                # RestrictAttr
                attrs.append(tree.MallocAttr())
            elif isinstance(inner_node, tree.Attr):
                attrs.append(inner_node)
            else:
                raise NotImplementedError(inner_node)

        exception = None
        exception_spec = type_info.get('exception_spec')
        if exception_spec:
            if exception_spec.get('isDynamic'):
                exception = tree.Throw(args=exception_spec.get('inner', []))
            elif exception_spec.get('isNoThrow'):
                exception = tree.NoThrow()
            elif exception_spec.get('isBasic'):
                # Unfortunately there might be implicit noexcept attributes
                # And I cannot find a way to check if it's implicit or not...
                if 'noexcept' in self.get_node_source_code(node):
                    repr_ = exception_spec.get('expr_repr')
                    exception = tree.NoExcept(repr=repr_)

        return body, args, init, method_attrs, attrs, exception

    @parse_debug
    def parse_CXXConstructorDecl(self, node) -> tree.CXXConstructorDecl:
        assert node['kind'] == "CXXConstructorDecl"
        if node.get('isImplicit'):
            return None

        name = node['name']
        # Get rid of template parameters in the name, they are usually just
        # ignored
        name = re.sub(r'(.*?)<.*',r'\1', name)

        body, args, inits, method_attrs, attrs, exception = self.parse_function_inner(node)

        assert not method_attrs

        constexpr = 'constexpr' if node.get('constexpr') else None

        if self.decl_informations.get(node['id'], {}).get('isExplicit'):
            explicit = 'explicit'
        else:
            explicit = None

        defaulted = self.parse_default(node)

        return tree.CXXConstructorDecl(name=name,
                                       constexpr=constexpr,
                                       exception=exception,
                                       attributes=attrs,
                                       explicit=explicit,
                                       defaulted=defaulted, body=body,
                                       parameters=args, initializers=inits)

    @parse_debug
    def parse_CXXCtorInitializer(self, node) -> tree.CXXCtorInitializer:
        assert node['kind'] == "CXXCtorInitializer"
        anyInit = node.get('anyInit')
        if anyInit:
            name = anyInit['name']
            args = self.parse_subnodes(node)

            # When the initializer list is not user-defined we need to not write anything.
            # Only if it have been written explicitly (user-defined) must we take it into account.
            if not args:
                return None

        baseInit = node.get('baseInit')
        if baseInit:
            name = baseInit["qualType"]
            args = self.parse_subnodes(node)

        if not anyInit and not baseInit:
            raise NotImplementedError("parse_CXXCtorInitializer not any nor base")

        return tree.CXXCtorInitializer(name=name, args=args)

    @parse_debug
    def parse_CXXDestructorDecl(self, node) -> tree.CXXDestructorDecl:
        assert node['kind'] == "CXXDestructorDecl"

        if node.get('isImplicit'):
            return None

        name = node['name']

        virtual = node.get('virtual') and tree.Virtual()

        body, args, inits, method_attrs, attrs, exception = self.parse_function_inner(node)
        assert not args
        assert not inits
        assert not method_attrs

        defaulted = self.parse_default(node)

        return tree.CXXDestructorDecl(name=name, exception=exception,
                                      attributes=attrs,
                                      virtual=virtual,
                                      defaulted=defaulted, body=body)

    def parse_default(self, node):
        if node.get('explicitlyDefaulted'):
            return tree.Default()
        elif node.get('explicitlyDeleted'):
            return tree.Delete()
        elif node.get('pure'):
            return tree.PureVirtual()
        else:
            return None

    @parse_debug
    def parse_AccessSpecDecl(self, node) -> tree.AccessSpecDecl:
        assert node['kind'] == "AccessSpecDecl"
        access = node['access']
        access_spec = getattr(tree, access.capitalize())()
        return tree.AccessSpecDecl(access_spec=access_spec)

    def is_variadic(self, node):
        qual_type = node['type']['qualType']
        if re.match(r".*, \.\.\.\)(( const.*)|( noexcept.*)|( ->.*)|( except.*))?$",
                    qual_type):
            return "..."
        else:
            return None

    @parse_debug
    def parse_CXXMethodDecl(self, node) -> tree.CXXMethodDecl:
        assert node['kind'] == "CXXMethodDecl"
        if node.get('isImplicit'):
            return None

        name = node['name']
        body, args, inits, method_attrs, attrs, exception = self.parse_function_inner(node)
        assert not inits

        type_info = self.type_informations[node['id']]
        return_type = self.parse_node(type_info).return_type
        variadic = self.is_variadic(node)
        inline = "inline" if node.get('inline') else None
        storage = node.get('storageClass')
        trailing_return = type_info.get("trailingReturn") and "trailing-return"
        constexpr = 'constexpr' if node.get('constexpr') else None

        const = type_info.get('isconst')
        if const:
            const = "const"

        ref_qualifier = type_info.get('ref_qualifier')
        if ref_qualifier == "LValue":
            ref_qualifier = "&"
        elif ref_qualifier == "RValue":
            ref_qualifier = "&&"
        else:
            assert ref_qualifier is None

        virtual = node.get('virtual') and tree.Virtual()

        defaulted = self.parse_default(node)

        return tree.CXXMethodDecl(name=name,
                                  constexpr=constexpr,
                                  return_type=return_type,
                                  variadic=variadic, parameters=args,
                                  inline=inline, storage=storage,
                                  virtual=virtual,
                                  trailing_return=trailing_return,
                                  body=body, exception=exception,
                                  attributes=attrs,
                                  # method specific keywords
                                  const=const, defaulted=defaulted,
                                  method_attributes=method_attrs,
                                  ref_qualifier=ref_qualifier)

    @parse_debug
    def parse_FunctionDecl(self, node) -> tree.FunctionDecl:
        assert node['kind'] == "FunctionDecl"

        name = node['name']
        type_info = self.type_informations[node['id']]
        return_type = getattr(self.parse_node(type_info), 'return_type', None)
        variadic = self.is_variadic(node)
        inline = "inline" if node.get('inline') else None
        storage = node.get('storageClass')
        trailing_return = type_info.get("trailingReturn") and "trailing-return"
        constexpr = 'constexpr' if node.get('constexpr') else None

        body, args, inits, method_attrs, attrs, exception = self.parse_function_inner(node)
        assert not inits
        assert not method_attrs

        return tree.FunctionDecl(name=name,
                                 constexpr=constexpr,
                                 return_type=return_type,
                                 trailing_return=trailing_return,
                                 attributes=attrs,
                                 variadic=variadic, parameters=args,
                                 inline=inline, storage=storage,
                                 body=body, exception=exception)

    @parse_debug
    def parse_GCCAsmStmt(self, node) -> tree.GCCAsmStmt:
        assert node['kind'] == "GCCAsmStmt"
        exprs = self.parse_subnodes(node)
        asm_infos = self.asm_informations[node['id']]

        asm_string = asm_infos['asm_string']
        input_constraints, input_exprs = [], []
        input_constraints_mapping = asm_infos.get('input_constraints', {})
        output_constraints, output_exprs = [], []
        output_constraints_mapping = asm_infos.get('output_constraints', {})
        for expr, subnode in zip(exprs, node.get('inner', ())):
            subnode_id = subnode['id']
            input_constraint = input_constraints_mapping.get(subnode_id, '')
            if input_constraint:
                input_constraints.append(input_constraint)
                input_exprs.append(expr)
            output_constraint = output_constraints_mapping.get(subnode_id, '')
            if output_constraint:
                output_constraints.append(output_constraint)
                output_exprs.append(expr)

        clobbers = asm_infos.get('clobbers', [])
        labels = asm_infos.get('labels', [])
        return tree.GCCAsmStmt(string=asm_string,
                               clobbers=clobbers,
                               labels=labels,
                               input_operands=[
                                   tree.ConstrainedExpression(expr=input_expr,
                                                              constraint=input_constraint)
                                   for input_expr, input_constraint in
                                   zip(input_exprs, input_constraints)],
                               output_operands=[
                                   tree.ConstrainedExpression(expr=output_expr,
                                                              constraint=output_constraint)
                                   for output_expr, output_constraint in
                                   zip(output_exprs, output_constraints)])

    @parse_debug
    def parse_TypedefDecl(self, node) -> tree.TypedefDecl:
        assert node['kind'] == "TypedefDecl"
        name = node['name']
        type_, = self.parse_subnodes(node)
        return tree.TypedefDecl(name=name, type=type_)

    @parse_debug
    def parse_ParmVarDecl(self, node) -> tree.ParmVarDecl:
        assert node['kind'] == "ParmVarDecl"
        name = node.get('name')
        var_type = self.parse_node(self.type_informations[node['id']])
        inner_nodes = self.parse_subnodes(node)

        default, attributes = None, []
        for inner_node in inner_nodes:
            if isinstance(inner_node, tree.Expression):
                assert not default
                default = inner_node
            elif isinstance(inner_node, tree.Attr):
                attributes.append(inner_node)
            else:
                raise NotImplementedError(inner_node)

        return tree.ParmVarDecl(name=name, type=var_type,
                                default=default, attributes=attributes)

    def as_statement(self, subnode):
        if isinstance(subnode, tree.Expression):
            return tree.ExprStmt(expr=subnode)
        if isinstance(subnode, tree.Declaration):
            return tree.DeclStmt(decls=[subnode])
        else:
            assert isinstance(subnode, tree.Statement), subnode
            return subnode

    def as_statements(self, subnodes):
        for i, subnode in enumerate(subnodes):
            subnodes[i] = self.as_statement(subnode)
        return subnodes

    @parse_debug
    def parse_AttributedStmt(self, node) -> tree.AttributedStmt:
        assert node['kind'] == "AttributedStmt"
        inner_nodes = self.parse_subnodes(node)
        attributes = inner_nodes[:-1]
        stmt = inner_nodes[-1]
        return tree.AttributedStmt(stmt=self.as_statement(stmt),
                                   attributes=attributes)


    @parse_debug
    def parse_CompoundStmt(self, node) -> tree.CompoundStmt:
        assert node['kind'] == "CompoundStmt"
        inner_nodes = self.parse_subnodes(node)
        return tree.CompoundStmt(stmts=self.as_statements(inner_nodes))

    @parse_debug
    def parse_CXXTryStmt(self, node) -> tree.CXXTryStmt:
        assert node['kind'] == "CXXTryStmt"
        body, *handlers = self.parse_subnodes(node)
        return tree.CXXTryStmt(body=self.as_statement(body),
                               handlers=self.as_statements(handlers))

    @parse_debug
    def parse_CXXCatchStmt(self, node) -> tree.CXXCatchStmt:
        assert node["kind"] == "CXXCatchStmt"
        subnodes = self.parse_subnodes(node, keep_empty=True)
        if len(subnodes) == 1:
            return tree.CXXCatchStmt(decl=None, body=subnodes[0])
        else:
            decl, body = subnodes
        return tree.CXXCatchStmt(decl= decl or None,
                                 body=body)

    @parse_debug
    def parse_IfStmt(self, node) -> tree.IfStmt:
        assert node['kind'] == "IfStmt"
        cond, *subnodes = self.parse_subnodes(node)
        if node.get('hasVar'):
            assert isinstance(cond, tree.DeclStmt)
            decl, = cond.decls
            cond = tree.DeclOrExpr(decl=decl, expr=None)
            subnodes = subnodes[1:]  # pop the implicit condition evaluation
        else:
            cond = tree.DeclOrExpr(decl=None, expr=cond)

        if len(subnodes) == 1:
            true_body, false_body = subnodes[0], None
        elif len(subnodes) == 2:
            true_body, false_body = subnodes
        else:
            print(f"IfStmt has more than 2 subnodes: {subnodes}",
                  file=sys.stderr)
            true_body, false_body, *others = subnodes
        return tree.IfStmt(cond=cond,
                           true_body=self.as_statement(true_body),
                           false_body=false_body and self.as_statement(false_body))

    @parse_debug
    def parse_LabelStmt(self, node) -> tree.LabelStmt:
        assert node['kind'] == "LabelStmt"
        name = node['name']
        child, = self.parse_subnodes(node)
        return tree.LabelStmt(name=name,
                              stmt=self.as_statement(child))

    @parse_debug
    def parse_GotoStmt(self, node) -> tree.LabelStmt:
        assert node['kind'] == "GotoStmt"
        target = self.get_node_source_code(node).replace('goto', '', 1).strip()
        return tree.GotoStmt(target=target)

    @parse_debug
    def parse_ForStmt(self, node) -> tree.ForStmt:
        assert node['kind'] == "ForStmt"
        subnodes = self.parse_subnodes(node, keep_empty=True)
        if len(subnodes) == 5:
            init, cond_decl, cond, inc, body = subnodes

            if isinstance(init, tree.Expression):
                init = tree.DeclsOrExpr(expr=init, decls=None)
            elif isinstance(init, tree.DeclStmt):
                init = tree.DeclsOrExpr(expr=None, decls=init.decls)
            if cond_decl:
                assert isinstance(cond_decl, tree.DeclStmt)
                decl, = cond_decl.decls
                cond = tree.DeclOrExpr(expr=None, decl=decl)
            elif cond:
                assert isinstance(cond, tree.Expression)
                cond = tree.DeclOrExpr(expr=cond, decl=None)
        else:
            print(f"ForStmt with number of subnodes != 5: {subnodes}",
                  file=sys.stderr)
            raise NotImplementedError(f"ForStmt with number of subnodes != 5: {subnodes}")
        return tree.ForStmt(
                init=init,
                cond=cond,
                inc=inc,
                body=self.as_statement(body))

    @parse_debug
    def parse_WhileStmt(self, node) -> tree.WhileStmt:
        assert node['kind'] == "WhileStmt"
        if node.get('hasVar'):
            var, cond, body = self.parse_subnodes(node)
            decl, = var.decls
            cond = tree.DeclOrExpr(decl=decl, expr=None)
        else:
            cond, body = self.parse_subnodes(node)
            cond =  tree.DeclOrExpr(decl=None, expr=cond)
        return tree.WhileStmt(cond=cond,
                              body=self.as_statement(body))

    @parse_debug
    def parse_DoStmt(self, node) -> tree.DoStmt:
        assert node['kind'] == "DoStmt"
        body, cond = self.parse_subnodes(node)
        return tree.DoStmt(cond=cond,
                           body=self.as_statement(body))

    @parse_debug
    def parse_ContinueStmt(self, node) -> tree.ContinueStmt:
        assert node['kind'] == "ContinueStmt"
        return tree.ContinueStmt()

    @parse_debug
    def parse_ReturnStmt(self, node) -> tree.ReturnStmt:
        assert node['kind'] == "ReturnStmt"
        subnodes = self.parse_subnodes(node)
        assert len(subnodes) <= 1
        value = subnodes[0] if len(subnodes) == 1 else None
        return tree.ReturnStmt(value=value)

    def parse_SwitchStmt(self, node) -> tree.SwitchStmt:
        assert node['kind'] == "SwitchStmt"
        cond, body = self.parse_subnodes(node)
        return tree.SwitchStmt(cond=cond,
                               body=self.as_statement(body))

    def parse_CaseStmt(self, node) -> tree.CaseStmt:
        assert node['kind'] == "CaseStmt"
        inner_nodes = self.parse_subnodes(node)
        if len(inner_nodes) == 3:
            pattern, pattern_end, child = inner_nodes
        else:
            pattern, child = inner_nodes
            pattern_end = None
        return tree.CaseStmt(pattern=pattern, pattern_end=pattern_end,
                             stmt=self.as_statement(child))

    def parse_BreakStmt(self, node) -> tree.BreakStmt:
        assert node['kind'] == "BreakStmt"
        return tree.BreakStmt()

    def parse_DefaultStmt(self, node) -> tree.DefaultStmt:
        assert node['kind'] == "DefaultStmt"
        child, = self.parse_subnodes(node)
        return tree.DefaultStmt(stmt=self.as_statement(child))

    def parse_CXXUnresolvedConstructExpr(self, node) -> tree.CXXUnresolvedConstructExpr:
        assert node['kind'] == 'CXXUnresolvedConstructExpr'
        expr, = self.parse_subnodes(node)
        type_ = self.parse_node(self.type_informations[node['id']])
        return tree.CXXUnresolvedConstructExpr(type=type_, expr=expr)

    def parse_SizeOfPackExpr(self, node) -> tree.SizeOfPackExpr:
        assert node['kind'] == 'SizeOfPackExpr'
        name = node['name']
        return tree.SizeOfPackExpr(name=name)

    def parse_UnresolvedLookupExpr(self, node) -> tree.UnresolvedLookupExpr:
        assert node['kind'] == 'UnresolvedLookupExpr'
        name = node['name']
        return tree.UnresolvedLookupExpr(name=name)

    def parse_ParenListExpr(self, node) -> tree.ParenListExpr:
        assert node["kind"] == "ParenListExpr"
        exprs = self.parse_subnodes(node)
        return tree.ParenListExpr(exprs=exprs)

    def parse_AddrLabelExpr(self, node) -> tree.AddrLabelExpr:
        assert node['kind'] == 'AddrLabelExpr'
        name = node['name']
        return tree.AddrLabelExpr(name=name)

    def parse_PackExpansionExpr(self, node) -> tree.PackExpansionExpr:
        assert node['kind'] == 'PackExpansionExpr'
        expr, = self.parse_subnodes(node)
        return tree.PackExpansionExpr(expr=expr)

    def parse_VAArgExpr(self, node) -> tree.VAArgExpr:
        assert node['kind'] == 'VAArgExpr'
        expr, = self.parse_subnodes(node)
        type_ = self.parse_node(self.type_informations[node['id']])
        return tree.VAArgExpr(expr=expr, type=type_)

    def parse_OffsetOfExpr(self, node) -> tree.OffsetOfExpr:
        assert node['kind'] == 'OffsetOfExpr'
        expr_infos = self.expr_informations[node['id']]
        type_, *kinds = expr_infos
        inner_nodes = iter(self.parse_subnodes(node))
        for kind in kinds:
            if isinstance(kind, tree.OffsetOfArray):
                kind.index = next(inner_nodes)
        return tree.OffsetOfExpr(type=type_, kinds=kinds)

    def parse_OffsetOfField(self, node) -> tree.OffsetOfField:
        return tree.OffsetOfField(name=node['field'])

    def parse_OffsetOfArray(self, node) -> tree.OffsetOfArray:
        # the actual index is set in the caller
        return tree.OffsetOfArray(index=None)

    def parse_IndirectGotoStmt(self, node) -> tree.IndirectGotoStmt:
        assert node['kind'] == 'IndirectGotoStmt'
        expr, = self.parse_subnodes(node)
        return tree.IndirectGotoStmt(expr=expr)

    def parse_PredefinedExpr(self, node) -> tree.PredefinedExpr:
        assert node['kind'] == 'PredefinedExpr'
        return tree.PredefinedExpr(name=node['name'])

    def parse_CXXThrowExpr(self, node) -> tree.CXXThrowExpr:
        assert node['kind'] == "CXXThrowExpr"
        inner_nodes = self.parse_subnodes(node)
        if inner_nodes:
            expr, = inner_nodes
        else:
            expr = None
        return tree.CXXThrowExpr(expr=expr)

    def parse_CXXThisExpr(self, node) -> tree.CXXThisExpr:
        assert node['kind'] == "CXXThisExpr"
        if node.get('implicit'):
            return None
        return tree.CXXThisExpr()

    def parse_CXXTypeidExpr(self, node) -> tree.CXXTypeidExpr:
        assert node['kind'] == "CXXTypeidExpr"
        if 'typeArg' in node:
            expr = None
            type_ = self.parse_node(self.type_informations[node['id']])
        else:
            expr, = self.parse_subnodes(node)
            type_ = None
        return tree.CXXTypeidExpr(expr=expr, type=type_)

    def parse_MemberExpr(self, node) -> tree.MemberExpr:
        assert node['kind'] == "MemberExpr"
        name = node['name']
        op = "->" if node.get('isArrow') else "."
        inner_nodes = self.parse_subnodes(node)
        if inner_nodes:
            expr, = inner_nodes
        else:
            expr = None
        return tree.MemberExpr(name=name, op=op, expr=expr)

    def parse_ConstantExpr(self, node) -> tree.ConstantExpr:
        assert node['kind'] == "ConstantExpr"
        result = node.get('value')
        expr, = self.parse_subnodes(node)
        return tree.ConstantExpr(expr=expr, result=result)

    @parse_debug
    def parse_DeclRefExpr(self, node) -> tree.DeclRefExpr:
        assert node['kind'] == "DeclRefExpr"
        name = self.get_node_source_code(node) #ref_decl['name']
        return tree.DeclRefExpr(name=name)

    @parse_debug
    def parse_DependentScopeDeclRefExpr(self, node) -> tree.DependentScopeDeclRefExpr:
        assert node['kind'] == "DependentScopeDeclRefExpr"
        name = self.get_node_source_code(node) #ref_decl['name']
        return tree.DependentScopeDeclRefExpr(name=name)

    @parse_debug
    def parse_IntegerLiteral(self, node) -> tree.IntegerLiteral:
        assert node['kind'] == "IntegerLiteral"
        value = node['value']
        if 'inner_type' in node:  # Added by our parser
            type_ = self.parse_node(node['inner_type'])
        else:
            type_ = self.parse_node(self.type_informations[node['id']])
        return tree.IntegerLiteral(type=type_, value=value)

    @parse_debug
    def parse_FloatingLiteral(self, node) -> tree.FloatingLiteral:
        assert node['kind'] == "FloatingLiteral"
        value = node['value'].lower()  # turns 'E' into 'e'
        if '.' not in value:
            value += '.'
        type_ = self.parse_node(self.type_informations[node['id']])
        return tree.FloatingLiteral(type=type_, value=value)

    @parse_debug
    def parse_ImaginaryLiteral(self, node) -> tree.ImaginaryLiteral:
        assert node['kind'] == "ImaginaryLiteral"
        type_ = self.parse_node(self.type_informations[node['id']])
        float_, = self.parse_subnodes(node)
        return tree.ImaginaryLiteral(type=type_, value=float_.value)

    def is_parameter_pack(self, expr, expr_node):
        decl = expr_node.get("referencedDecl")
        if decl is None:
            return False
        type_info = self.type_informations[decl["id"]]
        return type_info["kind"] == 'PackExpansionType'

    @parse_debug
    def parse_LambdaExpr(self, node) -> tree.LambdaExpr:
        assert node['kind'] == "LambdaExpr"
        # Lambda are parsed as an implicit class, dig into it to find the call operator
        cxx_record = node['inner'][0]
        cxx_methods = cxx_record['inner']
        for cxx_method in cxx_methods:
            if cxx_method['name'] == "operator()":
                break
        else:
            raise ValueError("expecting at least a call operator for a lambda")

        call_method = self.parse_node(cxx_method)

        extract_trailing_type = lambda d: d.trailing_return and d.return_type

        if isinstance(call_method, tree.FunctionTemplateDecl):
            parameters = call_method.decl.parameters
            trailing_type = extract_trailing_type(call_method.decl)
            variadic = call_method.decl.variadic
            exception = call_method.decl.exception
            attributes = call_method.decl.attributes
        else:
            parameters = call_method.parameters
            trailing_type = extract_trailing_type(call_method)
            variadic = call_method.variadic
            exception = call_method.exception
            attributes = call_method.attributes

        inner_nodes = self.parse_subnodes(node, keep_empty=True)
        capture_exprs = []
        body = None
        for inner_node, subnode in zip(inner_nodes, node["inner"]):
            if not inner_node:
                continue
            if isinstance(inner_node, tree.ParenListExpr):
                for expr, expr_node in zip(inner_node.exprs, subnode["inner"]):
                    if self.is_parameter_pack(expr, expr_node):
                        expr = tree.PackExpansionExpr(expr=expr)
                    capture_exprs.append(expr)
            elif isinstance(inner_node, tree.Expression):
                capture_exprs.append(inner_node)
            elif isinstance(inner_node, tree.Statement):
                assert not body
                body = inner_node
            else:
                raise NotImplementedError(inner_node)

        return tree.LambdaExpr(parameters=parameters, body=body,
                               trailing_type=trailing_type,
                               variadic=variadic, exception=exception,
                               attributes=attributes,
                               capture_exprs=capture_exprs)

    @parse_debug
    def parse_CharacterLiteral(self, node) -> tree.CharacterLiteral:
        assert node['kind'] == "CharacterLiteral"
        value = node['value']
        return tree.CharacterLiteral(value=chr(value))

    @parse_debug
    def parse_StringLiteral(self, node) -> tree.StringLiteral:
        assert node['kind'] == "StringLiteral"
        value = node['value']
        return tree.StringLiteral(value=value)

    @parse_debug
    def parse_UserDefinedLiteral(self, node) -> tree.UserDefinedLiteral:
        assert node['kind'] == "UserDefinedLiteral"
        func, expr = self.parse_subnodes(node)
        suffix = func.expr.name

        if isinstance(expr, (tree.FloatingLiteral, tree.IntegerLiteral)):
            expr.type.name = 'user-defined-literal'

        return tree.UserDefinedLiteral(suffix=suffix, expr=expr)

    @parse_debug
    def parse_CXXNullPtrLiteralExpr(self, node) -> tree.CXXNullPtrLiteralExpr:
        assert node['kind'] == "CXXNullPtrLiteralExpr"
        return tree.CXXNullPtrLiteralExpr()

    @parse_debug
    def parse_NamespaceDecl(self, node) -> tree.NamespaceDecl:
        assert node['kind'] == "NamespaceDecl"
        name = node.get('name') # Namespaces can be anonymous (no name given), in which case name is null
        inline = node.get('isInline') and "inline"
        decls = self.parse_subnodes(node)
        return tree.NamespaceDecl(name=name, inline=inline, decls=decls)

    @parse_debug
    def parse_UnresolvedUsingTypenameDecl(self, node) -> tree.UnresolvedUsingTypenameDecl:
        assert node['kind'] == "UnresolvedUsingTypenameDecl"
        desc = self.get_node_source_code(node)
        assert desc.startswith('using')
        name = desc[len('using'):].strip()
        assert name.endswith(node['name'])
        return tree.UnresolvedUsingTypenameDecl(name=name)

    @parse_debug
    def parse_UnresolvedUsingType(self, node) -> tree.UnresolvedUsingType:
        assert node['kind'] == "UnresolvedUsingType"
        name = node.get('name')
        if name is None:
            name = node['decl']['name']
        return tree.UnresolvedUsingType(name=name)

    #@parse_debug
    #def parse_Namespace(self, node) -> tree.Namespace:
        #assert node['kind'] == "Namespace"
        #name = node['name']
        #subnodes = self.parse_subnodes(node)
        #return tree.Namespace(name=name, subnodes=subnodes)

    @parse_debug
    def parse_StaticAssertDecl(self, node) -> tree.StaticAssertDecl:
        assert node['kind'] == "StaticAssertDecl"
        inner_nodes = self.parse_subnodes(node)
        if len(inner_nodes) == 1:
            cond, = inner_nodes
            message = None
        else:
            cond, message = inner_nodes
            message = message.value
        return tree.StaticAssertDecl(cond=cond, message=message)

    @parse_debug
    def parse_UsingDirectiveDecl(self, node) -> tree.UsingDirectiveDecl:
        assert node['kind'] == "UsingDirectiveDecl"
        name = node['nominatedNamespace']['name']
        assert name
        return tree.UsingDirectiveDecl(name=name)

    @parse_debug
    def parse_DeclStmt(self, node) -> tree.DeclStmt:
        assert node['kind'] == "DeclStmt"
        decls = self.parse_subnodes(node)
        return tree.DeclStmt(decls=decls)

    def mangle_anonymous_type(self, qual_type):
        if qual_type.startswith('struct (unnamed struct'):
            return self.anonymous_types[qual_type[7:]]
        else:
            return qual_type

    @parse_debug
    def parse_AutoType(self, node) -> tree.AutoType:
        assert node['kind'] == "AutoType"
        keyword = node['keyword']
        if keyword == "auto":
            keyword = tree.Auto()
        elif keyword == "decltype(auto)":
            keyword = tree.DecltypeAuto()
        elif keyword == "__auto_type":
            keyword = tree.GNUAutoType()

        return tree.AutoType(keyword=keyword)

    @parse_debug
    def parse_MemberPointerType(self, node) -> tree.MemberPointerType:
        assert node['kind'] == "MemberPointerType"
        cls, type_ = self.parse_subnodes(node)
        return tree.MemberPointerType(cls=cls, type=type_)

    @parse_debug
    def parse_PackExpansionType(self, node) -> tree.PackExpansionType:
        assert node['kind'] == "PackExpansionType"
        type_, = self.parse_subnodes(node)
        return tree.PackExpansionType(type=type_)

    @parse_debug
    def parse_BitIntType(self, node) -> tree.BitIntType:
        assert node['kind'] == "BitIntType"
        if 'size' in node:
            size = str(node['size'])
            sign = node['sign']
        else:
            qual_type = node['type']['qualType']
            match = re.match(r'^((?:(?:un)signed)?) ?_BitInt\(([0-9]+)\)$',
                             qual_type)
            sign, size = match.groups()

        return tree.BitIntType(size=size, sign=sign)

    @parse_debug
    def parse_QualType(self, node) -> tree.QualType:
        assert node['kind'] == "QualType"
        type_, = self.parse_subnodes(node)
        qualifiers = node['qualifiers']
        return tree.QualType(qualifiers=qualifiers, type=type_)

    @parse_debug
    def parse_VarDecl(self, node) -> tree.VarDecl:
        assert node['kind'] == "VarDecl"
        name = node.get('name', '') # Can be nonexistent in exception handlers
        implicit = node.get('isImplicit')
        referenced = node.get('isReferenced')
        if referenced:
            referenced = "referenced"
        storage_class = node.get('storageClass')
        tls = node.get('tls')

        type_ = self.parse_node(self.type_informations[node['id']])

        inner_nodes = self.parse_subnodes(node)

        if 'init' in node:
            init = inner_nodes.pop() if inner_nodes else None
            init_mode = node['init']
        else:
            init = None
            init_mode = ''

        constexpr = 'constexpr' if node.get('constexpr') else None
        attributes = inner_nodes

        return tree.VarDecl(name=name,
                            type=type_,
                            constexpr=constexpr,
                            storage_class=storage_class,
                            init_mode=init_mode,
                            implicit=implicit,
                            referenced=referenced,
                            init=init,
                            attributes=attributes,
                            tls=tls)

    @parse_debug
    def parse_AllocAlignAttr(self, node) -> tree.AllocAlignAttr:
        assert node['kind'] == "AllocAlignAttr"
        index = str(self.attr_informations[node['id']]['source_index'])
        return tree.AllocAlignAttr(index=index)

    @parse_debug
    def parse_AllocSizeAttr(self, node) -> tree.AllocSizeAttr:
        assert node['kind'] == "AllocSizeAttr"
        size = str(self.attr_informations[node['id']]['size_index'])
        nmemb = self.attr_informations[node['id']].get('nmemb_index')
        if nmemb is not None:
            nmemb = str(nmemb)
        return tree.AllocSizeAttr(size=size, nmemb=nmemb)

    @parse_debug
    def parse_AlignedAttr(self, node) -> tree.AlignedAttr:
        assert node['kind'] == "AlignedAttr"
        inner_nodes = self.parse_subnodes(node)
        if not inner_nodes:
            size = None
        else:
            size, = inner_nodes
        return tree.AlignedAttr(size=size)

    @parse_debug
    def parse_AliasAttr(self, node) -> tree.AliasAttr:
        assert node['kind'] == "AliasAttr"
        aliasee = self.attr_informations[node['id']]['aliasee']
        return tree.AliasAttr(aliasee=aliasee)

    @parse_debug
    def parse_AlwaysInlineAttr(self, node) -> tree.AlwaysInlineAttr:
        assert node['kind'] == "AlwaysInlineAttr"
        return tree.AlwaysInlineAttr()

    @parse_debug
    def parse_ColdAttr(self, node) -> tree.ColdAttr:
        assert node['kind'] == "ColdAttr"
        return tree.ColdAttr()

    @parse_debug
    def parse_FallThroughAttr(self, node) -> tree.FallThroughAttr:
        assert node['kind'] == "FallThroughAttr"
        return tree.FallThroughAttr()

    @parse_debug
    def parse_LikelyAttr(self, node) -> tree.LikelyAttr:
        assert node['kind'] == "LikelyAttr"
        return tree.LikelyAttr()

    @parse_debug
    def parse_UnlikelyAttr(self, node) -> tree.UnlikelyAttr:
        assert node['kind'] == "UnlikelyAttr"
        return tree.UnlikelyAttr()

    @parse_debug
    def parse_ConstAttr(self, node) -> tree.ConstAttr:
        assert node['kind'] == "ConstAttr"
        return tree.ConstAttr()

    @parse_debug
    def parse_ConstructorAttr(self, node) -> tree.ConstructorAttr:
        assert node['kind'] == "ConstructorAttr"
        priority = self.attr_informations.get(node['id'], {}).get('priority')
        if priority is not None:
            priority = str(priority)
        return tree.ConstructorAttr(priority=priority)

    @parse_debug
    def parse_DestructorAttr(self, node) -> tree.DestructorAttr:
        assert node['kind'] == "DestructorAttr"
        priority = self.attr_informations.get(node['id'], {}).get('priority')
        if priority is not None:
            priority = str(priority)
        return tree.DestructorAttr(priority=priority)

    @parse_debug
    def parse_ErrorAttr(self, node) -> tree.ErrorAttr:
        assert node['kind'] == "ErrorAttr"
        message = self.attr_informations[node['id']]['message']
        return tree.ErrorAttr(msg=message)

    @parse_debug
    def parse_FlattenAttr(self, node) -> tree.FlattenAttr:
        assert node['kind'] == "FlattenAttr"
        return tree.FlattenAttr()

    @parse_debug
    def parse_CleanupAttr(self, node) -> tree.CleanupAttr:
        assert node['kind'] == "CleanupAttr"
        func = self.attr_informations[node['id']]['cleanup_function']
        return tree.CleanupAttr(func=func)

    @parse_debug
    def parse_DeprecatedAttr(self, node) -> tree.DeprecatedAttr:
        assert node['kind'] == "DeprecatedAttr"
        msg = self.attr_informations[node['id']]['deprecation_message'] or None
        return tree.DeprecatedAttr(msg=msg)

    @parse_debug
    def parse_NoUniqueAddressAttr(self, node) -> tree.NoUniqueAddressAttr:
        assert node['kind'] == "NoUniqueAddressAttr"
        return tree.NoUniqueAddressAttr()

    @parse_debug
    def parse_CarriesDependencyAttr(self, node) -> tree.CarriesDependencyAttr:
        assert node['kind'] == "CarriesDependencyAttr"
        return tree.CarriesDependencyAttr()

    @parse_debug
    def parse_UnavailableAttr(self, node) -> tree.UnavailableAttr:
        assert node['kind'] == "UnavailableAttr"
        msg = self.attr_informations[node['id']]['deprecation_message'] or None
        return tree.UnavailableAttr(msg=msg)

    @parse_debug
    def parse_RetainAttr(self, node) -> tree.RetainAttr:
        assert node['kind'] == "RetainAttr"
        return tree.RetainAttr()

    @parse_debug
    def parse_SectionAttr(self, node) -> tree.SectionAttr:
        assert node['kind'] == "SectionAttr"
        section = self.attr_informations[node['id']]['section_name']
        return tree.SectionAttr(section=section)

    @parse_debug
    def parse_TLSModelAttr(self, node) -> tree.TLSModelAttr:
        assert node['kind'] == "TLSModelAttr"
        tls_model = self.attr_informations[node['id']]['tls_model']
        return tree.TLSModelAttr(tls_model=tls_model)

    @parse_debug
    def parse_UnusedAttr(self, node) -> tree.UnusedAttr:
        assert node['kind'] == "UnusedAttr"
        return tree.UnusedAttr()

    @parse_debug
    def parse_FinalAttr(self, node) -> tree.FinalAttr:
        assert node['kind'] == "FinalAttr"
        return tree.FinalAttr()

    @parse_debug
    def parse_FormatAttr(self, node) -> tree.FormatAttr:
        assert node['kind'] == "FormatAttr"
        archetype = self.attr_informations[node['id']]['archetype']
        fmt_index = str(self.attr_informations[node['id']]['fmt_index'])
        vargs_index = str(self.attr_informations[node['id']]['vargs_index'])
        return tree.FormatAttr(archetype=archetype, fmt_index=fmt_index,
                               vargs_index=vargs_index)

    @parse_debug
    def parse_FormatArgAttr(self, node) -> tree.FormatArgAttr:
        assert node['kind'] == "FormatArgAttr"
        fmt_index = str(self.attr_informations[node['id']]['fmt_index'])
        return tree.FormatArgAttr(fmt_index=fmt_index)

    @parse_debug
    def parse_GNUInlineAttr(self, node) -> tree.GNUInlineAttr:
        assert node['kind'] == "GNUInlineAttr"
        return tree.GNUInlineAttr()

    @parse_debug
    def parse_HotAttr(self, node) -> tree.HotAttr:
        assert node['kind'] == "HotAttr"
        return tree.HotAttr()

    @parse_debug
    def parse_IFuncAttr(self, node) -> tree.IFuncAttr:
        assert node['kind'] == "IFuncAttr"
        name = self.attr_informations[node['id']]['name']
        return tree.IFuncAttr(name=name)

    @parse_debug
    def parse_AnyX86InterruptAttr(self, node) -> tree.AnyX86InterruptAttr:
        assert node['kind'] == "AnyX86InterruptAttr"
        return tree.AnyX86InterruptAttr()

    @parse_debug
    def parse_LeafAttr(self, node) -> tree.LeafAttr:
        assert node['kind'] == "LeafAttr"
        return tree.LeafAttr()

    @parse_debug
    def parse_RestrictAttr(self, node) -> tree.RestrictAttr:
        assert node['kind'] == "RestrictAttr"
        return tree.RestrictAttr()

    @parse_debug
    def parse_NoInstrumentFunctionAttr(self, node) -> tree.NoInstrumentFunctionAttr:
        assert node['kind'] == "NoInstrumentFunctionAttr"
        return tree.NoInstrumentFunctionAttr()

    @parse_debug
    def parse_NoInlineAttr(self, node) -> tree.NoInlineAttr:
        assert node['kind'] == "NoInlineAttr"
        return tree.NoInlineAttr()

    @parse_debug
    def parse_NonNullAttr(self, node) -> tree.NonNullAttr:
        assert node['kind'] == "NonNullAttr"
        indices = list(map(str,self.attr_informations[node['id']]['indices']))
        return tree.NonNullAttr(indices=indices)

    @parse_debug
    def parse_NoSplitStackAttr(self, node) -> tree.NoSplitStackAttr:
        assert node['kind'] == "NoSplitStackAttr"
        return tree.NoSplitStackAttr()

    @parse_debug
    def parse_NoProfileFunctionAttr(self, node) -> tree.NoProfileFunctionAttr:
        assert node['kind'] == "NoProfileFunctionAttr"
        return tree.NoProfileFunctionAttr()

    @parse_debug
    def parse_NoSanitizeAttr(self, node) -> tree.NoSanitizeAttr:
        assert node['kind'] == "NoSanitizeAttr"
        options = self.attr_informations[node['id']]['options']
        return tree.NoSanitizeAttr(options=options)

    @parse_debug
    def parse_WarnUnusedResultAttr(self, node) -> tree.WarnUnusedResultAttr:
        assert node['kind'] == "WarnUnusedResultAttr"
        return tree.WarnUnusedResultAttr()

    @parse_debug
    def parse_NoStackProtectorAttr(self, node) -> tree.NoStackProtectorAttr:
        assert node['kind'] == "NoStackProtectorAttr"
        return tree.NoStackProtectorAttr()

    @parse_debug
    def parse_TargetAttr(self, node) -> tree.TargetAttr:
        assert node['kind'] == "TargetAttr"
        # There are so many ways to describe a target...
        desc = self.get_node_source_code(node)
        desc = re.sub(r'^\s*__target__\s*\((.*)\)$', r'\1', desc)
        return tree.TargetAttr(desc=desc)

    @parse_debug
    def parse_TargetClonesAttr(self, node) -> tree.TargetClonesAttr:
        assert node['kind'] == "TargetClonesAttr"
        # There are so many ways to describe a target...
        desc = self.get_node_source_code(node)
        desc = re.sub(r'^\s*target_clones\s*\((.*)\)$', r'\1', desc)
        return tree.TargetClonesAttr(desc=desc)

    @parse_debug
    def parse_PatchableFunctionEntryAttr(self, node) -> tree.PatchableFunctionEntryAttr:
        assert node['kind'] == "PatchableFunctionEntryAttr"
        count = str(self.attr_informations[node['id']]['count'])
        offset = self.attr_informations[node['id']].get('offset')
        if offset is not None:
            offset = str(offset)
        return tree.PatchableFunctionEntryAttr(count=count, offset=offset)

    @parse_debug
    def parse_SentinelAttr(self, node) -> tree.SentinelAttr:
        assert node['kind'] == "SentinelAttr"
        value = self.attr_informations.get(node['id'], {}).get('value')
        offset = self.attr_informations.get(node['id'], {}).get('offset')
        if value is not None or offset is not None:
            value = str(value or 0)
        if offset is not None:
            offset = str(offset)
        return tree.SentinelAttr(value=value, offset=offset)

    @parse_debug
    def parse_PureAttr(self, node) -> tree.PureAttr:
        assert node['kind'] == "PureAttr"
        return tree.PureAttr()

    @parse_debug
    def parse_ReturnsNonNullAttr(self, node) -> tree.ReturnsNonNullAttr:
        assert node['kind'] == "ReturnsNonNullAttr"
        return tree.ReturnsNonNullAttr()

    @parse_debug
    def parse_ReturnsTwiceAttr(self, node) -> tree.ReturnsTwiceAttr:
        assert node['kind'] == "ReturnsTwiceAttr"
        return tree.ReturnsTwiceAttr()

    @parse_debug
    def parse_UsedAttr(self, node) -> tree.UsedAttr:
        assert node['kind'] == "UsedAttr"
        if node.get('implicit'):
            return None
        return tree.UsedAttr()

    @parse_debug
    def parse_UninitializedAttr(self, node) -> tree.UninitializedAttr:
        assert node['kind'] == "UninitializedAttr"
        return tree.UninitializedAttr()

    @parse_debug
    def parse_VisibilityAttr(self, node) -> tree.VisibilityAttr:
        assert node['kind'] == "VisibilityAttr"
        visibility = self.attr_informations[node['id']]['visibility']
        return tree.VisibilityAttr(visibility=visibility)

    @parse_debug
    def parse_WeakAttr(self, node) -> tree.WeakAttr:
        assert node['kind'] == "WeakAttr"
        return tree.WeakAttr()

    @parse_debug
    def parse_WeakRefAttr(self, node) -> tree.WeakRefAttr:
        assert node['kind'] == "WeakRefAttr"
        name = self.attr_informations.get(node['id'], {}).get('name')
        return tree.WeakAttr()

    @parse_debug
    def parse_PackedAttr(self, node) -> tree.PackedAttr:
        assert node['kind'] == "PackedAttr"
        return tree.PackedAttr()

    @parse_debug
    def parse_InitListExpr(self, node) -> tree.InitListExpr:
        assert node['kind'] == "InitListExpr"
        values = self.parse_subnodes(node)
        return tree.InitListExpr(values=values)

    @parse_debug
    def parse_TypeRef(self, node) -> tree.TypeRef:
        assert node['kind'] == "TypeRef"
        name = node['name']
        subnodes = self.parse_subnodes(node)
        return tree.TypeRef(name=name, subnodes=subnodes)

    @parse_debug
    def parse_TypeAliasTemplateDecl(self, node) -> tree.TypeAliasTemplateDecl:
        assert node['kind'] == "TypeAliasTemplateDecl"
        inner_nodes = self.parse_subnodes(node)
        template_parameters = inner_nodes[:-1]
        decl = inner_nodes[-1]
        return tree.TypeAliasTemplateDecl(
                template_parameters=template_parameters,
                decl=decl)

    @parse_debug
    def parse_TypeAliasDecl(self, node) -> tree.TypeAliasDecl:
        assert node['kind'] == "TypeAliasDecl"
        name = node['name']
        type_, = self.parse_subnodes(node)
        return tree.TypeAliasDecl(name=name, type=type_)

    @parse_debug
    def parse_UsingDecl(self, node) -> tree.UsingDecl:
        assert node['kind'] == "UsingDecl"
        name = node['name']
        return tree.UsingDecl(name=name)

    #@parse_debug
    #def parse_NamespaceRef(self, node) -> tree.NamespaceRef:
        #assert node['kind'] == "NamespaceRef"
        #name = node['name']
        #subnodes = self.parse_subnodes(node)
        #return tree.NamespaceRef(name=name, subnodes=subnodes)

    @parse_debug
    def parse_ExprWithCleanups(self, node) -> tree.ExprWithCleanups:
        assert node['kind'] == "ExprWithCleanups"
        expr, = self.parse_subnodes(node)
        return tree.ExprWithCleanups(expr=expr)

    @parse_debug
    def parse_CXXConstructExpr(self, node) -> tree.CXXConstructExpr:
        assert node['kind'] == "CXXConstructExpr"

        args = self.parse_subnodes(node)
        return tree.CXXConstructExpr(args=args)

    @parse_debug
    def parse_MaterializeTemporaryExpr(self, node) -> tree.MaterializeTemporaryExpr:
        assert node['kind'] == "MaterializeTemporaryExpr"
        expr, = self.parse_subnodes(node)
        return tree.MaterializeTemporaryExpr(expr=expr)

    @parse_debug
    def parse_CXXBindTemporaryExpr(self, node) -> tree.CXXBindTemporaryExpr:
        assert node['kind'] == "CXXBindTemporaryExpr"
        expr, = self.parse_subnodes(node)
        return tree.CXXBindTemporaryExpr(expr=expr)

    @parse_debug
    def parse_ImplicitCastExpr(self, node) -> tree.ImplicitCastExpr:
        assert node['kind'] == "ImplicitCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        subnodes = self.parse_subnodes(node)
        if len(subnodes) > 0:
            expr, = self.parse_subnodes(node)
            return tree.ImplicitCastExpr(type=type_, expr=expr)
        else:
            raise NotImplementedError

    @parse_debug
    def parse_CXXDefaultArgExpr(self, node) -> None:
        assert node['kind'] == "CXXDefaultArgExpr"
        return None

    @parse_debug
    def parse_FieldDecl(self, node) -> tree.FieldDecl:
        assert node['kind'] == "FieldDecl"
        name = node['name']
        var_type = self.parse_node(self.type_informations[node['id']])
        inner_nodes = self.parse_subnodes(node)

        type_qualifier = "mutable" if node.get('mutable') else None  # TODO: add support for const and volatile

        if node.get('isBitfield'):
            bitwidth = inner_nodes.pop(0)
        else:
            bitwidth = None

        if node.get('hasInClassInitializer'):
            init = inner_nodes.pop(0)
        else:
            init = None

        attributes = inner_nodes

        return tree.FieldDecl(name=name, type=var_type, init=init,
                              type_qualifier=type_qualifier,
                              bitwidth=bitwidth, attributes=attributes)

    @parse_debug
    def parse_CXXDefaultInitExpr(self, node) -> tree.CXXDefaultInitExpr:
        assert node['kind'] == "CXXDefaultInitExpr"
        expr = node.get('expression')
        if expr:
            return tree.CXXDefaultInitExpr(expr=expr)
        else:
            return None

    @parse_debug
    def parse_BinaryOperator(self, node) -> tree.BinaryOperator:
        assert node['kind'] == "BinaryOperator"
        opcode = node['opcode']
        lhs, rhs = self.parse_subnodes(node)
        return tree.BinaryOperator(opcode=opcode, lhs=lhs, rhs=rhs)

    @parse_debug
    def parse_CompoundAssignOperator(self, node) -> tree.CompoundAssignOperator:
        assert node['kind'] == "CompoundAssignOperator"
        opcode = node['opcode']
        lhs, rhs = self.parse_subnodes(node)
        return tree.CompoundAssignOperator(opcode=opcode, lhs=lhs, rhs=rhs)

    @parse_debug
    def parse_UnaryOperator(self, node) -> tree.UnaryOperator:
        assert node['kind'] == "UnaryOperator"
        opcode = node['opcode']
        expr, = self.parse_subnodes(node)
        postfix = str(node['isPostfix'])
        return tree.UnaryOperator(opcode=opcode, expr=expr, postfix=postfix)

    @parse_debug
    def parse_BinaryConditionalOperator(self, node) -> tree.BinaryConditionalOperator:
        assert node['kind'] == "BinaryConditionalOperator"
        cond, _, _, false_expr = self.parse_subnodes(node)
        return tree.BinaryConditionalOperator(cond=cond, false_expr=false_expr)

    @parse_debug
    def parse_ConditionalOperator(self, node) -> tree.ConditionalOperator:
        assert node['kind'] == "ConditionalOperator"
        cond, true_expr, false_expr = self.parse_subnodes(node)
        return tree.ConditionalOperator(cond=cond, true_expr=true_expr,
                                        false_expr=false_expr)

    @parse_debug
    def parse_ChooseExpr(self, node) -> tree.ChooseExpr:
        assert node['kind'] == "ChooseExpr"
        cond, true_expr, false_expr = self.parse_subnodes(node)
        return tree.ChooseExpr(cond=cond, true_expr=true_expr,
                                        false_expr=false_expr)

    @parse_debug
    def parse_ArraySubscriptExpr(self, node) -> tree.ArraySubscriptExpr:
        assert node['kind'] == "ArraySubscriptExpr"
        base, index = self.parse_subnodes(node)
        return tree.ArraySubscriptExpr(base=base, index=index)

    @parse_debug
    def parse_OpaqueValueExpr(self, node) -> tree.OpaqueValueExpr:
        assert node['kind'] == "OpaqueValueExpr"
        expr, = self.parse_subnodes(node)
        return tree.OpaqueValueExpr(expr=expr)

    @parse_debug
    def parse_StmtExpr(self, node) -> tree.StmtExpr:
        assert node['kind'] == "StmtExpr"
        stmt, = self.parse_subnodes(node)
        return tree.StmtExpr(stmt=stmt)

    @parse_debug
    def parse_AtomicExpr(self, node) -> tree.AtomicExpr:
        assert node['kind'] == "AtomicExpr"
        name = self.attr_informations[node['id']]['name']
        args = self.parse_subnodes(node)
        # for some reason, inner args are not listed in the syntactic order
        if name.startswith("__atomic_compare_exchange"):
            args = args[:1] + args[2:3] + args[4:] + args[1:4:2]
        else:
            args = args[:1] + args[2:] + args[1:2]
        return tree.AtomicExpr(name=name, args=args)

    @parse_debug
    def parse_ParenExpr(self, node) -> tree.ParenExpr:
        assert node['kind'] == "ParenExpr"
        expr, = self.parse_subnodes(node)
        return tree.ParenExpr(expr=expr)

    @parse_debug
    def parse_UnaryExprOrTypeTraitExpr(self, node) -> tree.UnaryExprOrTypeTraitExpr:
        assert node['kind'] == "UnaryExprOrTypeTraitExpr"
        name = node['name']
        if 'argType' in node:
            expr = None
            type_ = self.parse_node(self.type_informations[node['id']])
        else:
            expr = self.parse_subnodes(node)[0]
            type_ = None
        return tree.UnaryExprOrTypeTraitExpr(name=name, expr=expr, type=type_)

    @parse_debug
    def parse_ClassTemplateDecl(self, node) -> tree.ClassTemplateDecl:
        assert node['kind'] == "ClassTemplateDecl"
        inner_nodes = self.parse_subnodes(node)
        template_parameters = []
        decl = None
        for inner_node, json_descr in zip(inner_nodes, node['inner']):
            if isinstance(inner_node, tree.TemplateParmDecl):
                template_parameters.append(inner_node)
            elif isinstance(inner_node, tree.CXXRecordDecl):
                # first instance is the generic definition, the others are the
                # one generated by instantiation, register them
                if not decl:
                    decl = inner_node
                else:
                    descr = self.parse_subnodes(json_descr)
                    self.template_instances[json_descr['id']] = [
                            d for d in descr if isinstance(d, tree.TemplateArgument)
                    ]
        return tree.ClassTemplateDecl(template_parameters=template_parameters,
                                      decl=decl)

    @parse_debug
    def parse_ClassTemplateSpecializationDecl(self, node) -> tree.ClassTemplateSpecializationDecl:
        assert node['kind'] == "ClassTemplateSpecializationDecl"

        name, kind, bases, complete, inner_nodes = self.parse_class_inner(node)
        if kind is None:
            return None

        decls = []
        template_parameters = []
        template_arguments = []

        for inner_node in inner_nodes:
            if isinstance(inner_node, tree.TemplateArgument):
                template_arguments.append(inner_node)
            elif isinstance(inner_node, tree.TemplateParmDecl):
                template_parameters.append(inner_node)
            else:
                decls.append(inner_node)

        return tree.ClassTemplateSpecializationDecl(
                template_arguments=template_arguments,
                template_parameters=template_parameters,
                name=name, kind=kind, bases=bases,
                complete=complete,
                decls=decls)


    @parse_debug
    def parse_ClassTemplatePartialSpecializationDecl(self, node) -> tree.ClassTemplatePartialSpecializationDecl:
        assert node['kind'] == "ClassTemplatePartialSpecializationDecl"


        name, kind, bases, complete, inner_nodes = self.parse_class_inner(node)
        if kind is None:
            return None

        decls = []
        template_parameters = []
        template_arguments = []

        for inner_node in inner_nodes:
            if isinstance(inner_node, tree.TemplateArgument):
                template_arguments.append(inner_node)
            elif isinstance(inner_node, tree.TemplateParmDecl):
                template_parameters.append(inner_node)
            else:
                decls.append(inner_node)

        return tree.ClassTemplatePartialSpecializationDecl(
                template_arguments=template_arguments,
                template_parameters=template_parameters,
                name=name, kind=kind, bases=bases,
                complete=complete,
                decls=decls)


    @parse_debug
    def parse_FunctionTemplateDecl(self, node) -> tree.FunctionTemplateDecl:
        assert node['kind'] == "FunctionTemplateDecl"
        inner_nodes = self.parse_subnodes(node)
        template_parameters = []
        decl = None
        for inner_node, json_descr in zip(inner_nodes, node['inner']):
            if isinstance(inner_node, tree.TemplateParmDecl):
                template_parameters.append(inner_node)
            elif isinstance(inner_node, (tree.FunctionDecl, tree.CXXMethodDecl)):
                # first instance is the generic definition, the others are the
                # one generated by instantiation, register them
                if not decl:
                    decl = inner_node
                else:
                    descr = self.parse_subnodes(json_descr)
                    self.template_instances[json_descr['id']] = [
                            d for d in descr if isinstance(d, tree.TemplateArgument)
                    ]
            else:
                raise NotImplementedError(inner_node)

        assert decl is not None
        return tree.FunctionTemplateDecl(template_parameters=template_parameters,
                                         decl=decl)

    @parse_debug
    def parse_TemplateArgument(self, node):
        assert node['kind'] == "TemplateArgument"
        inner_nodes = self.parse_subnodes(node)

        if 'value' in node:
            intty = tree.BuiltinType(name='int')
            intval = str(node['value'])
            value = tree.IntegerLiteral(type=intty, value=intval)
        elif len(inner_nodes) == 1:
            value, = inner_nodes
        elif len(inner_nodes) > 1:
            value = inner_nodes
        else:
            return tree.TemplateArgument(type=None, expr=None, pack=None)

        assert value is not None

        if isinstance(value, tree.Type):
            return tree.TemplateArgument(type=value, expr=None, pack=None)
        elif isinstance(value, tree.Expression):
            return tree.TemplateArgument(type=None, expr=value, pack=None)
        elif isinstance(value, list):
            assert all(isinstance(elt, tree.TemplateArgument) for elt in value)
            return tree.TemplateArgument(type=None, expr=None, pack=value)
        else:
            raise NotImplementedError

    @parse_debug
    def parse_TemplateArgumentPack(self, node):
        # A kind of node that only exists because we forge it in our custom json
        # generator
        assert node['kind'] == "TemplateArgumentPack"
        inner_nodes = self.parse_subnodes(node)
        return inner_nodes


    @parse_debug
    def parse_SubstTemplateTypeParmType(self, node) -> tree.SubstTemplateTypeParmType:
        assert node['kind'] == "SubstTemplateTypeParmType"
        type_, = self.parse_subnodes(node)
        return tree.SubstTemplateTypeParmType(type=type_)

    @parse_debug
    def parse_InjectedClassNameType(self, node) -> tree.InjectedClassNameType:
        assert node['kind'] == "InjectedClassNameType"
        type_, = self.parse_subnodes(node)
        return tree.InjectedClassNameType(type=type_)

    @parse_debug
    def parse_TemplateTypeParmDecl(self, node) -> tree.TemplateTypeParmDecl:
        assert node['kind'] == "TemplateTypeParmDecl"
        name = node.get('name')
        tag = getattr(tree, node['tagUsed'].capitalize() + 'Tag')()
        inner_nodes = self.parse_subnodes(node)
        if inner_nodes:
            default, = inner_nodes
        else:
            default = None
        parameter_pack = node.get("isParameterPack") and "pack"
        return tree.TemplateTypeParmDecl(name=name, tag=tag, default=default,
                                         parameter_pack=parameter_pack)

    @parse_debug
    def parse_TemplateTemplateParmDecl(self, node) -> tree.TemplateTemplateParmDecl:
        assert node['kind'] == "TemplateTemplateParmDecl"
        name = node.get('name')
        template_parameters = self.parse_subnodes(node)
        return tree.TemplateTemplateParmDecl(name=name,
                                             template_parameters=template_parameters)

    @parse_debug
    def parse_NonTypeTemplateParmDecl(self, node) -> tree.NonTypeTemplateParmDecl:
        assert node['kind'] == "NonTypeTemplateParmDecl"
        type_ = self.parse_node(self.type_informations[node['id']])
        name = node.get('name')
        inner_nodes = self.parse_subnodes(node)
        if inner_nodes:
            default, = inner_nodes
        else:
            default = None
        parameter_pack = node.get("isParameterPack") and "pack"
        return tree.NonTypeTemplateParmDecl(name=name, type=type_,
                                            default=default,
                                            parameter_pack=parameter_pack)

    @parse_debug
    def parse_FullComment(self, node) -> tree.FullComment:
        assert node['kind'] == "FullComment"
        comment = self.collect_comment(node)
        return tree.FullComment(comment=comment)

    @parse_debug
    def parse_OverrideAttr(self, node) -> tree.OverrideAttr:
        assert node['kind'] == "OverrideAttr"
        return tree.OverrideAttr()

    @parse_debug
    def parse_CXX11NoReturnAttr(self, node) -> tree.CXX11NoReturnAttr:
        assert node['kind'] == "CXX11NoReturnAttr"
        return tree.CXX11NoReturnAttr()

    @parse_debug
    def parse_CXXMemberCallExpr(self, node) -> tree.CXXMemberCallExpr:
        assert node['kind'] == "CXXMemberCallExpr"
        bound_method, *args = self.parse_subnodes(node)
        return tree.CXXMemberCallExpr(bound_method=bound_method, args=args)

    @parse_debug
    def parse_CallExpr(self, node) -> tree.CallExpr:
        assert node['kind'] == "CallExpr"
        callee, *args = self.parse_subnodes(node)
        return tree.CallExpr(callee=callee, args=args)

    @parse_debug
    def parse_CXXOperatorCallExpr(self, node) -> tree.CXXOperatorCallExpr:
        assert node['kind'] == "CXXOperatorCallExpr"
        subnodes = self.parse_subnodes(node)
        assert len(subnodes) >= 2
        op = subnodes[0]
        left = subnodes[1]
        right = subnodes[2] if len(subnodes) == 3 else None
        return tree.CXXOperatorCallExpr(left=left, op=op, right=right)

    @parse_debug
    def parse_CXXBoolLiteralExpr(self, node) -> tree.CXXBoolLiteralExpr:
        assert node['kind'] == "CXXBoolLiteralExpr"
        subnodes = self.parse_subnodes(node)
        return tree.CXXBoolLiteralExpr(value=str(node['value']))

    @parse_debug
    def parse_CXXTemporaryObjectExpr(self, node) -> tree.CXXTemporaryObjectExpr:
        assert node['kind'] == "CXXTemporaryObjectExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        args = self.parse_subnodes(node)
        return tree.CXXTemporaryObjectExpr(type=type_, args=args)

    @parse_debug
    def parse_CXXFunctionalCastExpr(self, node) -> tree.CXXFunctionalCastExpr:
        assert node['kind'] == "CXXFunctionalCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        expr, = self.parse_subnodes(node)
        return tree.CXXFunctionalCastExpr(type=type_, expr=expr)

    @parse_debug
    def parse_BuiltinBitCastExpr(self, node) -> tree.BuiltinBitCastExpr:
        assert node['kind'] == "BuiltinBitCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        expr, = self.parse_subnodes(node)
        return tree.BuiltinBitCastExpr(type=type_, expr=expr)

    @parse_debug
    def parse_CXXStaticCastExpr(self, node) -> tree.CXXStaticCastExpr:
        assert node['kind'] == "CXXStaticCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        expr, = self.parse_subnodes(node)
        value_category = node['valueCategory']
        return tree.CXXStaticCastExpr(type=type_, expr=expr, value_category=value_category)

    @parse_debug
    def parse_CXXConstCastExpr(self, node) -> tree.CXXConstCastExpr:
        assert node['kind'] == "CXXConstCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        expr, = self.parse_subnodes(node)
        value_category = node['valueCategory']
        return tree.CXXConstCastExpr(type=type_, expr=expr, value_category=value_category)

    @parse_debug
    def parse_CXXReinterpretCastExpr(self, node) -> tree.CXXReinterpretCastExpr:
        assert node['kind'] == "CXXReinterpretCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        expr, = self.parse_subnodes(node)
        return tree.CXXReinterpretCastExpr(type=type_, expr=expr)

    @parse_debug
    def parse_NullStmt(self, node) -> tree.NullStmt:
        assert node['kind'] == "NullStmt"
        return tree.NullStmt()

    @parse_debug
    def parse_EnumConstantDecl(self, node) -> tree.EnumConstantDecl:
        assert node['kind'] == "EnumConstantDecl"
        name = node['name']
        children = self.parse_subnodes(node)
        init = None if not children else children[0]
        return tree.EnumConstantDecl(name=name, init=init)

    @parse_debug
    def parse_EnumDecl(self, node) -> tree.EnumDecl:
        assert node['kind'] == "EnumDecl"
        name = node.get('name')
        fields = self.parse_subnodes(node)
        return tree.EnumDecl(name=name, fields=fields)

    @parse_debug
    def parse_ImplicitValueInitExpr(self, node) -> tree.ImplicitValueInitExpr:
        assert node['kind'] == "ImplicitValueInitExpr"
        return tree.ImplicitValueInitExpr()

    @parse_debug
    def parse_CXXConversionDecl(self, node) -> tree.CXXConversionDecl:
        assert node['kind'] == "CXXConversionDecl"
        name = node['name']
        body, args, inits, method_attrs, attrs, exception = self.parse_function_inner(node)
        assert not inits
        assert not args

        type_info = self.type_informations[node['id']]
        inline = "inline" if node.get('inline') else None

        const = type_info.get('isconst')
        if const:
            const = "const"

        return tree.CXXConversionDecl(name=name,
                                      inline=inline,
                                      attributes=attrs,
                                      body=body, exception=exception,
                                      # method specific keywords
                                      const=const)

    @parse_debug
    def parse_EmptyDecl(self, node) -> tree.EmptyDecl:
        assert node['kind'] == "EmptyDecl"
        return tree.EmptyDecl()

    @parse_debug
    def parse_CStyleCastExpr(self, node) -> tree.CStyleCastExpr:
        assert node['kind'] == "CStyleCastExpr"
        type_ = self.parse_node(self.type_informations[node['id']])
        expr, = self.parse_subnodes(node)
        return tree.CStyleCastExpr(type=type_, expr=expr)

    @parse_debug
    def parse_FriendDecl(self, node) -> tree.FriendDecl:
        assert node['kind'] == "FriendDecl"
        subnodes = self.parse_subnodes(node)
        if subnodes:
            decl, = subnodes
            raw_decl = None
        else:
            decl = None
            raw_decl = node['type']['qualType']

        return tree.FriendDecl(decl=decl, raw_decl=raw_decl)

    @parse_debug
    def parse_CXXStdInitializerListExpr(self, node) -> tree.CXXStdInitializerListExpr:
        assert node['kind'] == "CXXStdInitializerListExpr"
        subnodes = self.parse_subnodes(node)
        return tree.CXXStdInitializerListExpr(subnodes=subnodes)

    @parse_debug
    def parse_CXXNewExpr(self, node) -> tree.CXXNewExpr:
        assert node['kind'] == "CXXNewExpr"
        args = self.parse_subnodes(node)

        if node.get("isArray"):
            array_size, *args = args
        else:
            array_size = None

        if node.get("isPlacement"):
            args, placement = args[:-1], args[-1]
        else:
            placement = None

        type_ = self.parse_node(self.type_informations[node['id']])
        assert isinstance(type_, tree.PointerType)
        return tree.CXXNewExpr(type=type_.type, args=args,
                               array_size=array_size,
                               placement=placement)

    @parse_debug
    def parse_CXXDeleteExpr(self, node) -> tree.CXXDeleteExpr:
        assert node['kind'] == "CXXDeleteExpr"
        expr, = self.parse_subnodes(node)
        is_array = node.get('isArrayAsWritten')
        if is_array:
            is_array = 'array'
        return tree.CXXDeleteExpr(expr=expr, is_array=is_array)

    @parse_debug
    def parse_CXXForRangeStmt(self, node) -> tree.CXXForRangeStmt:
        assert node['kind'] == "CXXForRangeStmt"
        # range is not explicit, one need to dive through the implicitly generated begin statement
        range_ = self.parse_node(node['inner'][1]['inner'][0]['inner'][0])
        decl_stmt = self.parse_node(node['inner'][6])
        body =  self.parse_node(node['inner'][7])
        assert isinstance(decl_stmt, tree.DeclStmt)
        if len(decl_stmt.decls) != 1:
            raise NotImplementedError()
        decl, = decl_stmt.decls
        decl.init_mode = ''
        decl.init = None
        return tree.CXXForRangeStmt(decl=decl, range=range_, body=body)

    @parse_debug
    def parse_BuiltinType(self, node) -> tree.BuiltinType:
        assert node['kind'] == "BuiltinType"
        return tree.BuiltinType(name=node['type']['qualType'])

    @parse_debug
    def parse_ConstantArrayType(self, node) -> tree.ConstantArrayType:
        assert node['kind'] == "ConstantArrayType"
        size = str(node['size'])
        type_, = self.parse_subnodes(node)
        return tree.ConstantArrayType(type=type_, size=size)

    @parse_debug
    def parse_DependentSizedArrayType(self, node) -> tree.DependentSizedArrayType:
        assert node['kind'] == "DependentSizedArrayType"
        type_, *expr = self.parse_subnodes(node)
        size_repr = node.get('size_repr')  # FIXME: should be an expression
        if size_repr is None:
            # in some cases, the json file has full type info
            size_expr, = expr  # unfortunately we cannot use that yet
            size_repr = self.get_node_source_code(node['inner'][1])
        return tree.DependentSizedArrayType(type=type_, size_repr=size_repr)

    @parse_debug
    def parse_VariableArrayType(self, node) -> tree.VariableArrayType:
        assert node['kind'] == "VariableArrayType"
        type_, *expr = self.parse_subnodes(node)
        size_repr = node.get('size_repr')  # FIXME: should be an expression
        if size_repr is None:
            # in some cases, the json file has full type info
            size_expr, = expr  # unfortunately we cannot use that yet
            size_repr = self.get_node_source_code(node['inner'][1])
        return tree.VariableArrayType(type=type_, size_repr=size_repr)

    @parse_debug
    def parse_ComplexType(self, node) -> tree.ComplexType:
        assert node['kind'] == "ComplexType"
        type_, = self.parse_subnodes(node)
        return tree.ComplexType(type=type_)

    @parse_debug
    def parse_ElaboratedType(self, node) -> tree.ElaboratedType:
        assert node['kind'] == "ElaboratedType"
        type_, = self.parse_subnodes(node)
        qualifiers = node.get('qualifiers')
        return tree.ElaboratedType(qualifiers=qualifiers, type=type_)

    @parse_debug
    def parse_DecayedType(self, node) -> tree.DecayedType:
        assert node['kind'] == "DecayedType"
        type_, = self.parse_subnodes(node)
        return tree.DecayedType(type=type_)

    @parse_debug
    def parse_FunctionNoProtoType(self, node) -> tree.FunctionNoProtoType:
        assert node['kind'] == "FunctionNoProtoType"
        args = self.parse_subnodes(node)
        assert not args
        return tree.FunctionNoProtoType()

    @parse_debug
    def parse_FunctionProtoType(self, node) -> tree.FunctionProtoType:
        assert node['kind'] == "FunctionProtoType"
        trailing_return = node.get('trailingReturn') and "auto"
        return_type, *parameter_types = self.parse_subnodes(node)
        return tree.FunctionProtoType(return_type=return_type,
                                      trailing_return=trailing_return,
                                      parameter_types=parameter_types)

    @parse_debug
    def parse_TypedefType(self, node) -> tree.TypedefType:
        assert node['kind'] == "TypedefType"
        type_, = self.parse_subnodes(node)
        name = node.get('name')
        if name is None:
            name = node["decl"]["name"]
        return tree.TypedefType(name=name, type=type_)


    @parse_debug
    def parse_IncompleteArrayType(self, node) -> tree.IncompleteArrayType:
        assert node['kind'] == "IncompleteArrayType"
        type_, = self.parse_subnodes(node)
        return tree.IncompleteArrayType(type=type_)

    @parse_debug
    def parse_ParenType(self, node) -> tree.ParenType:
        assert node['kind'] == "ParenType"
        type_, = self.parse_subnodes(node)
        return tree.ParenType(type=type_)

    @parse_debug
    def parse_PointerType(self, node) -> tree.PointerType:
        assert node['kind'] == "PointerType"
        type_, = self.parse_subnodes(node)
        return tree.PointerType(type=type_)

    @parse_debug
    def parse_RecordType(self, node) -> tree.RecordType:
        assert node['kind'] == "RecordType"
        return tree.RecordType(name=node['decl']['name'])

    @parse_debug
    def parse_TemplateSpecializationType(self, node) -> tree.TemplateSpecializationType:
        assert node['kind'] == "TemplateSpecializationType"
        inner_nodes = self.parse_subnodes(node)
        if 'templateName' in node:
            if 'id' in node:
                type_info = self.type_informations[node['id']]
                template_arg_extra = iter(type_info["templateArgumentsExtra"])
            else:
                template_arg_extra = iter(())

            template_args = [
                    inner_node for inner_node in inner_nodes
                    if isinstance(inner_node, tree.TemplateArgument)
            ]

            for ta in template_args:
                if ta.expr == ta.type == ta.pack == None:
                    ta.expr = tree.DumpedExpr(value=next(template_arg_extra))
            name = node['templateName']
        elif 'name' in node:
            name = node['name']
            template_args = []
            def to_template_argument(obj):
                if isinstance(obj, tree.Type):
                    ta = tree.TemplateArgument(type=obj)
                elif isinstance(obj, tree.Expression):
                    ta = tree.TemplateArgument(expr=obj)
                elif isinstance(obj, list):
                    ta = tree.TemplateArgument(pack=[to_template_argument(elt)
                                                     for elt in obj])
                else:
                    raise NotImplementedError
                return ta

            for inner_node in inner_nodes:
                template_args.append(to_template_argument(inner_node))
        else:
            raise NotImplementedError
        return tree.TemplateSpecializationType(name=name,
                                               template_args=template_args)


    @parse_debug
    def parse_DumpedExpr(self, node) -> tree.DumpedExpr:
        # FIXME: this node shoudln't exist
        assert node['kind'] == "DumpedExpr"
        return tree.DumpedExpr(value=node["value"])

    @parse_debug
    def parse_SubstNonTypeTemplateParmExpr(self, node) -> tree.SubstNonTypeTemplateParmExpr:
        assert node['kind'] == "SubstNonTypeTemplateParmExpr"
        decl, expr = self.parse_subnodes(node)
        return tree.SubstNonTypeTemplateParmExpr(decl=decl, expr=expr)

    @parse_debug
    def parse_TypeOfExprType(self, node) -> tree.TypeOfExprType:
        assert node['kind'] == "TypeOfExprType"
        return tree.TypeOfExprType(repr=node["expr_repr"])

    @parse_debug
    def parse_DecltypeType(self, node) -> tree.DecltypeType:
        assert node['kind'] == "DecltypeType"
        return tree.DecltypeType(repr=node["expr_repr"])

    @parse_debug
    def parse_VectorType(self, node) -> tree.VectorType:
        assert node['kind'] == "VectorType"
        size = str(node['size'])
        type_, = self.parse_subnodes(node)
        return tree.VectorType(type=type_, size=size)

    @parse_debug
    def parse_EnumType(self, node) -> tree.EnumType:
        assert node['kind'] == "EnumType"
        return tree.EnumType(name=node['decl']['name'])

    @parse_debug
    def parse_LValueReferenceType(self, node) -> tree.LValueReferenceType:
        assert node['kind'] == "LValueReferenceType"
        type_, = self.parse_subnodes(node)
        return tree.LValueReferenceType(type=type_)

    @parse_debug
    def parse_DependentNameType(self, node) -> tree.DependentNameType:
        assert node['kind'] == "DependentNameType"
        if 'id' in node:
            type_info = self.type_informations[node['id']]
        else:
            type_info = node

        attr = type_info["attribute_name"]
        nested = type_info["nested_name"]
        return tree.DependentNameType(nested=nested, attr=attr)

    @parse_debug
    def parse_TemplateTypeParmType(self, node) -> tree.TemplateTypeParmType:
        assert node['kind'] == "TemplateTypeParmType"
        name = node.get('name')

        if name is None:
            index = node['index']
            depth = node['depth']
            for parent in self.stack:
                if parent['kind'] in ('ClassTemplateDecl',
                                      'ClassTemplatePartialSpecializationDecl',
                                      'TypeAliasTemplateDecl',):
                    if depth != 0:
                        depth -= 1
                    else:
                        break
            else:
                raise NotImplementedError
            parent_inner = parent['inner']
            parent_template_params = [
                    n for n in parent_inner
                    if n['kind'] in ('TemplateTypeParmDecl',
                                     'NonTypeTemplateParmDecl')
            ]
            name = parent_template_params[index]['name']
        # This happens for lambda becuase they are represented as templated function
        elif name.endswith(":auto"):
            return tree.AutoType(keyword=tree.Auto())

        return tree.TemplateTypeParmType(name=name)

    @parse_debug
    def parse_RValueReferenceType(self, node) -> tree.RValueReferenceType:
        assert node['kind'] == "RValueReferenceType"
        type_, = self.parse_subnodes(node)
        return tree.RValueReferenceType(type=type_)


def parse(tokens, debug=False, filepath=None):
    parser = Parser(tokens, filepath)
    parser.set_debug(debug)
    return parser.parse()

