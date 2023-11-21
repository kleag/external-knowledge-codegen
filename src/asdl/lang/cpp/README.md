This dir contains the tool allowing to convert C++ code to AST and back to the equivalent C++. This is used to generate
the sequence of grammar rule calls corresponding to input C++ code. This will be the training data of the model learning
to convert natural language into valid C++ code.

The demo.py program shoud accept any valid C++ code, convert it to ASDL AST and back to the equivalent (except for
spacing or other details) code. It allows to test the completeness of the grammar and accompanying code.

The equivalent exist (and is complete) for Java and Python3. It can be used as a reference while the present C++ version
is not complete.

To test the AST generator, one must first have `cpplang` in its PYTHONPATH. To do so, you can source the `setenv.sh`
file in the project root. Please use LLVM 15. LLVM 12 is known to fail.

Then, you can try:

```bash
python demo.py -D -f <file>
```

with `<file>` the path to a C++ file. If all goes well (unprobable while the work is not finished), it should print:
```
Succes: 1/1 (100%)
```

Add the `-c` switch to also test the hypothese parse tree. If the analysis is complete, it should be equivalent to the
initial parse tree.

Add the `-h` switch to see all options.


To generate the input used by the demo (preprocessor only):

```bash
clang -x c++ -E -std=c++17 -fPIC -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtCore -I/home/gael/Projets/Lima/lima/lima_common/src/ -I/home/gael/Projets/Lima/lima/lima_common/src/common/XMLConfigurationFiles <file>
```

Add `=json` to the `-ast-dump` option to see the complete json used in our python code. Note that there is sometimes subtle diffirences between the shaped output and the raw json. Look at the former to understand the organization of the information but refer to the later to produce the actual code.


```bash
clang -x c++ -std=c++17 -fPIC -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtCore -I/home/gael/Projets/Lima/lima/lima_common/src/ -I/home/gael/Projets/Lima/lima/lima_common/src/common/XMLConfigurationFiles -Xclang -ast-dump -fsyntax-only <file>
```
## How it works

  1. Convert the C++ code into an AST (using the cpplang library):

    ```python
    cpp_ast = cpplang.parse.parse(cpp_code, …)
    ```

  2. Convert the cpp AST into general-purpose ASDL AST used by tranX

    ```python
    asdl_ast = cpp_ast_to_asdl_ast(cpp_ast, grammar)
    ```

  3. Convert the ASDL AST back into Cpp AST

    ```python
    cpp_ast_reconstructed = asdl_ast_to_cpp_ast(asdl_ast, grammar)
    ```

  4. Convert back the C++ ASTs (the initial one and the reconstructed one) to source code

    ```python
    cppastor.to_source(cpp_ast) # or cpp_ast_reconstructed
    ```

    Both generated source codes should be the same except for comments and spaces.

  5. (optional) Check that the same source code can be generated directly from the ASDL AST.

  ```python
    code_from_hyp(asdl_ast, …)
  ```

## Important constructs

### `def parse_XYZ` in `cpplang/parser.py`

These `parse_XYZ`, with `XYZ` the name of a C++ AST node as defined by clang (e.g.: `CXXMethodDecl`), have all the same
structure:

```python
    @parse_debug
    def parse_XYZ(self, node) -> tree.XYZ:
        assert node['kind'] == "XYZ"
        x = … # get information specific to XYZ from node
        subnodes = self.parse_subnodes(node)
        return tree.XYZ(x=x, …, subnodes=subnodes)
```

First, the return type is a subclass of `Node` from `tree.py`. There is one such class for each node defined by the
clang C++ AST. These have attributes that must be initialized at instantiation time. Their values are retrieved from
current context. The `x` here. All nodes have a `subnodes` attribute that is initialized by calling `parse_subnodes`.
Finally, a new instance of `XYZ` is returned with all its attributes as parameters.

### All `Node` subclasses in `cpplang/tree.py`

There is on such class for each C++ AST node defined by clang. The parent class `Node` have a `subnodes` attribute and
child classes add other attributes as needed by the AST definition. All attributes are listed as strings in the `attrs`
member of the class. Python mechanisms are used to convert them into real attributes of the instantiated objects.


### The `Parser` class in `cpplang/parser.py`

Converts C++ code into a tree of AST nodes rooted on a `tree.TranslationUnit`. It runs clang on instantiation and then
recursively its `parse_*` methods on calling `parse`.

## Adding new construction to the incomplete grammar

  * Find an unsupported construct
    * Process C++ files and find one where the parsing fails
    * Try to create a minimum failing example For example, at time of writing the following code


```
namespace N {
template <typename T, typename S>
class Object
{
};
}
```

fails with

```
))))))) Original Cpp code      :
namespace N {
template <typename T, typename S>
class Object
{
};
}

(((((((

}}}}}}} Cpp AST                :
namespace N
 {
template<typename T, typename S>class Object;

}

{{{{{{{

Common prefix end: namespacen{template<typenamet,typenames>classobject (92%)
**Warn** Test failed for file: test/object.hpp

```

  * From the error, we can see that the code generated from the AST differs from after the name of the class. The
    class body is replaced by a `;` as in a forward declaration
  * By looking at the AST (see above how to generate it) of the same class with and without body (forward declaration or not), one can see that with a body, there is an attribute `definition` and subnodes to the `CXXRecordDecl` object that are absent without a body. When looking at the corresponding json (which is what is handled in our python code), we see that the actual attribute is `completeDefinition`  and it is set to `true`. This attribute is absent in the case of a forward declaration. The solution is to add the `completeDefinition` attribute.

### Adding support for a new C++ AST libclang class

Adding support for a new class from libclang C++ AST is made in 4 files:

  * The grammar (`cpp_asdl_simplified.txt`)
  * Node subclass to build the in-memory AST tree (`tree.py`)
  * The parsing method to generate the tree class instance during parsing from the json representation of the AST built by clang (`parser.py`)
  * The visitor method used to regenerate C++ tokens from the internal AST

To determine how to implement each of them, one can run clang in syntax only mode either with default output to get an idea of the organization or with json output to check the exact content.Then, after a first implementation, it is possible to run in debug mode and to stop in the parser method to verify the available attributes.

Sometimes, the libclang AST contains information that is not rendered in final code depending on the context and it can be not easy to identify such cases. For example, the CXXForRangeStmt has several subnodes whose content are not or only partially rendered as they are implicit. In this kind of case, you can be forced to trace down the execution. by setting breakpoint at strategic points. This can be in the `parse_XYZ` or `visit_XYZ` methods in question. or at `r = method(self, node)` in `parser.py` where the recursion is applied during parsing or at `visit(item)` in `code_gen.py`  where the recursion is applied during code generation. For CXXForRangeStmt we determined that its `VarDecl` subnodes had to be generated differently if they were present in a CXXForRangeStmt or in other context. This could be determined by looking at the `VarDecl` `isImplicit` and `isReferenced` members. So they had to be stored as attributes in the `tree.VarDecl` class and then used in `vist_VarDecl` to condition its outpout.


### Adding an attribute to an AST node

Let the node be called XYZ and the attribute x.

Add the new attribute to the grammar in `cpp_asdl_simplified.txt`

In `tree.py`, in the class `XYZ`, add `x` to the attributes:

```python
class XYZ(…):
    attrs = (…, "x",)
```

In `parser.py`, in the function `parse_XYZ)`, add the code to initialize `x`. To do this, study the json output from clang and determine how to retrieve this information. Have a look to other examples in parser.py. Add the new attribute to the call to tree.XYZ constructor at the end of `parse_XYZ`, before `subnodes=subnodes`.

In `lang/cpp/cppastor/code_gen.py`, in `visit_XYZ`, add the necessary code to write C++ tokens corresponding to the new
attribute.

### Understanding why something fails

  * Compare the "String representation of the ASDL AST" and the "String representation of the reconstructed CPP AST". If a token is missing in the second one or in the contrary if there is duplicated tokens,
    * it could be from a missing attribute in the corresponding rule in the grammar in `cpp_asdl_simplified.txt`
    * it could come from  a `visit_XYZ` method in `code_gen.py`;
  * If with the swich `-c` which check if the hypotheses ASDL parse tree can regenerate the initial code, you get something like
    `Error: Valid continuation types are (<class 'asdl.transition_system.GenTokenAction'>,) but current action class is
    <class 'asdl.transition_system.ApplyRuleAction'>`, it probably means that a `visit_XYZ` method in `code_gen.py` has
    not generated the correct tokens or that there is an incoherence between a class definition in `tree.py` and
    `cpp_asdl_simplified.txt`. Check the class in the right of the previous `ApplyRule` action (printed in debug mode).
  * If you get `AttributeError: No defined visitor for node of type XYZ`, `please define 'visit_XYZ' in cpp/cppastor/code_gen.py.`, the message explains clearly what to do, but check that `XYZ` is defined in `cpp_asdl_simplified.txt`, `parser.py` and `tree.py`.
  * If you get `AttributeError: module 'cpplang.tree' has no attribute 'XYZ'`, add this class to `tree.py`. Check also its definition in `code_gen.py`, `cpp_asdl_simplified.txt`, `parser.py`.
  * If you get `AttributeError: No defined parse handler for clang node of type "XYZ".`, `please define "parse_ImplicitCastExpr" in cpplang/parser.py.`, the message explains clearly what to do, but check that `XYZ` is defined in `cpp_asdl_simplified.txt`, `code_gen.py` and `tree.py`.
  * If you get `Exception: Error: there is no production named XYZ in the ASDL grammar`, then add `XYZ` in `cpp_asdl_simplified.txt`, but check that `XYZ` is defined in `parser.py`, `code_gen.py` and `tree.py`.
  * If you get

```
  File "…/external-knowledge-codegen/asdl/lang/cpp/cpp_asdl_helper.py", line 47, in cpp_ast_to_asdl_ast
    field_value = getattr(cpp_ast_node, field.name)
AttributeError: 'XYZ' object has no attribute 'xy'
```
it could be that there is a typo in the attribute name in `cpp_asdl_simplified.txt` or that the attribute is missing in `tree.py`.

