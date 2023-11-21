
from .ast import Node

# ------------------------------------------------------------------------------


class TranslationUnit(Node):
    attrs = ("stmts",)


class Import(Node):
    attrs = ("path", "static", "wildcard")

class Attr(Node):
    attrs = ()


class OverrideAttr(Attr):
    attrs = ()


class FinalAttr(Attr):
    attrs = ()


class AlignedAttr(Attr):
    attrs = ("size",)


class AllocAlignAttr(Attr):
    attrs = ("index",)


class AlwaysInlineAttr(Attr):
    attrs = ()


class ColdAttr(Attr):
    attrs = ()


class ConstAttr(Attr):
    attrs = ()


class ConstructorAttr(Attr):
    attrs = ("priority",)


class CXX11NoReturnAttr(Attr):
    attrs = ()


class DestructorAttr(Attr):
    attrs = ("priority",)


class ErrorAttr(Attr):
    attrs = ("msg",)


class FallThroughAttr(Attr):
    attrs = ()


class LikelyAttr(Attr):
    attrs = ()


class UnlikelyAttr(Attr):
    attrs = ()


class FlattenAttr(Attr):
    attrs = ()


class FormatAttr(Attr):
    attrs = ("archetype", "fmt_index", "vargs_index",)


class FormatArgAttr(Attr):
    attrs = ("fmt_index",)


class GNUInlineAttr(Attr):
    attrs = ()


class HotAttr(Attr):
    attrs = ()


class IFuncAttr(Attr):
    attrs = ("name",)


class AnyX86InterruptAttr(Attr):
    attrs = ()


class NoUniqueAddressAttr(Attr):
    attrs = ()


class CarriesDependencyAttr(Attr):
    attrs = ()


class PatchableFunctionEntryAttr(Attr):
    attrs = ("count", "offset",)


class PureAttr(Attr):
    attrs = ()


class ReturnsNonNullAttr(Attr):
    attrs = ()


class ReturnsTwiceAttr(Attr):
    attrs = ()


class NoStackProtectorAttr(Attr):
    attrs = ()


class TargetAttr(Attr):
    attrs = ("desc",)


class TargetClonesAttr(Attr):
    attrs = ("desc",)


class SentinelAttr(Attr):
    attrs = ("value", "offset",)


class LeafAttr(Attr):
    attrs = ()


class WarnUnusedResultAttr(Attr):
    attrs = ()


class MallocAttr(Attr):
    attrs = ()


class RestrictAttr(Attr):
    attrs = ()


class NoInstrumentFunctionAttr(Attr):
    attrs = ()


class NoInlineAttr(Attr):
    attrs = ()


class NoReturnAttr(Attr):
    attrs = ()


class NonNullAttr(Attr):
    attrs = ("indices",)


class NoProfileFunctionAttr(Attr):
    attrs = ()


class NoSanitizeAttr(Attr):
    attrs = ("options",)


class NoSplitStackAttr(Attr):
    attrs = ()


class AllocSizeAttr(Attr):
    attrs = ("size", "nmemb")


class AliasAttr(Attr):
    attrs = ("aliasee",)


class CleanupAttr(Attr):
    attrs = ("func",)


class DeprecatedAttr(Attr):
    attrs = ("msg",)


class UnavailableAttr(Attr):
    attrs = ("msg",)


class PackedAttr(Attr):
    attrs = ()


class RetainAttr(Attr):
    attrs = ()


class SectionAttr(Attr):
    attrs = ("section",)


class TLSModelAttr(Attr):
    attrs = ("tls_model",)


class UsedAttr(Attr):
    attrs = ()


class UnusedAttr(Attr):
    attrs = ()


class UninitializedAttr(Attr):
    attrs = ()


class VisibilityAttr(Attr):
    attrs = ("visibility",)


class WeakAttr(Attr):
    attrs = ()


class WeakRefAttr(Attr):
    attrs = ("name",)


class Documented(Node):
    attrs = ("documentation",)


class Delete(Node):
    attrs = ()


class Default(Node):
    attrs = ()


class PureVirtual(Node):
    attrs = ()


class Comment(Node):
    attrs = ("comment",)


class FullComment(Comment):
    attrs = ()


class BlockCommandComment(Comment):
    attrs = ()


class ParagraphComment(Comment):
    attrs = ()


class TextComment(Comment):
    attrs = ()


class Declaration(Documented):
    attrs = ()


class EmptyDeclaration(Declaration):
    attrs = ()


class NonEmptyDeclaration(Declaration):
    attrs = ("modifiers", "annotations")


class TypeDeclaration(NonEmptyDeclaration):
    attrs = ("name",)

    @property
    def fields(self):
        return [decl for decl in self.body if isinstance(decl,
                                                         FieldDecl)]

    @property
    def methods(self):
        return [decl for decl in self.body if isinstance(decl,
                                                         MethodDecl)]

    @property
    def constructors(self):
        return [decl for decl in self.body if isinstance(
          decl, ConstructorDeclaration)]


class PackageDeclaration(NonEmptyDeclaration):
    attrs = ("name",)


class CXXRecordDecl(TypeDeclaration):
    attrs = ("kind", "bases", "complete", "decls",)


class RecordDecl(TypeDeclaration):
    attrs = ()


class CXXConstructorDecl(Declaration):
    attrs = ("name", "constexpr", "exception", "defaulted", "body", "attributes", "parameters", "initializers",
             "explicit",)


class CXXCtorInitializer(Node):
    attrs = ("name", "args",)


class CXXDestructorDecl(Declaration):
    attrs = ("name", "exception", "virtual", "defaulted", "body", "attributes", )


class AccessSpecDecl(Declaration):
    attrs = ("access_spec",)


class Virtual(Node):
    attrs = ()


class Public(Node):
    attrs = ()


class Private(Node):
    attrs = ()


class Protected(Node):
    attrs = ()


class EnumDeclaration(TypeDeclaration):
    attrs = ("implements",)

    @property
    def fields(self):
        return [decl for decl in self.body.declarations if isinstance(
          decl, FieldDecl)]

    @property
    def methods(self):
        return [decl for decl in self.body.declarations if isinstance(
          decl, MethodDeclaration)]


class InterfaceDeclaration(TypeDeclaration):
    attrs = ("type_parameters", "extends",)


class AnnotationDeclaration(TypeDeclaration):
    attrs = ()


class StaticInitializer(NonEmptyDeclaration):
    attrs = ("block",)


class InstanceInitializer(NonEmptyDeclaration):
    attrs = ("block",)

# ------------------------------------------------------------------------------


class ArrayDimension(Node):
    attrs = ("dim",)


class Modifier(Node):
    attrs = ("value",)


class Operator(Node):
    attrs = ("operator",)

# ------------------------------------------------------------------------------


class Type(Node):
    attrs = ()

class AutoType(Type):
    attrs = ("keyword",)

class BitIntType(Type):
    attrs = ("size", "sign",)

class BuiltinType(Type):
    attrs = ("name",)

class ConstantArrayType(Type):
    attrs = ("type", "size", )

class DependentSizedArrayType(Type):
    attrs = ("type", "size_repr", )

class VariableArrayType(Type):
    attrs = ("type", "size_repr",)

class DependentNameType(Type):
    attrs = ("nested", "attr",)

class DecayedType(Type):
    attrs = ("type",)

class TypedefType(Type):
    attrs = ("name", "type",)

class ElaboratedType(Type):
    attrs = ("type", "qualifiers", )

class ComplexType(Type):
    attrs = ("type",)

class FunctionNoProtoType(Type):
    attrs = ()

class FunctionProtoType(Type):
    attrs = ("return_type", "parameter_types", "trailing_return")

class LValueReferenceType(Type):
    attrs = ("type",)

class RValueReferenceType(Type):
    attrs = ("type",)

class IncompleteArrayType(Type):
    attrs = ("type",)

class InjectedClassNameType(Type):
    attrs = ("type",)

class MemberPointerType(Type):
    attrs = ("cls", "type",)

class ParenType(Type):
    attrs = ("type",)

class PackExpansionType(Type):
    attrs = ("type",)

class PointerType(Type):
    attrs = ("type",)

class QualType(Type):
    attrs = ("qualifiers", "type",)

class RecordType(Type):
    attrs = ("name",)

class UnresolvedUsingType(Type):
    attrs = "name",

class VectorType(Type):
    attrs = ("type", "size")

class EnumType(Type):
    attrs = ("name",)

class DiamondType(Type):
    attrs = ("sub_type",)

class SubstTemplateTypeParmType(Type):
    attrs = ("type",)

class TemplateTypeParmType(Type):
    attrs = ("name",)

class TemplateSpecializationType(Type):
    attrs = ("name", "template_args",)

class ReferenceType(Type):
    attrs = ("arguments", "sub_type",)


class TypeOfExprType(Type):
    attrs = ("repr",)


class DecltypeType(Type):
    attrs = ("repr",)


class TypeArgument(Node):
    attrs = ("type", "pattern_type",)

# ------------------------------------------------------------------------------

class Auto(Node):
    attrs = ()

class DecltypeAuto(Node):
    attrs = ()

class GNUAutoType(Node):
    attrs = ()

# ------------------------------------------------------------------------------

class TypeParameter(Node):
    attrs = ("name", "extends",)

# ------------------------------------------------------------------------------


class Annotation(Node):
    attrs = ("name", "element",)


class NormalAnnotation(Annotation):
    attrs = ()


class MarkerAnnotation(Annotation):
    attrs = ()


class SingleElementAnnotation(Annotation):
    attrs = ()


class ElementValuePair(Node):
    attrs = ("name", "value",)


class ElementArrayValue(Node):
    attrs = ("values",)


class InitListExpr(Node):
    attrs = ("values",)


# ------------------------------------------------------------------------------


class Member(NonEmptyDeclaration):
    attrs = ()


class CXXMethodDecl(Declaration):
    attrs = ("name", "constexpr", "return_type", "storage", "variadic", "inline",
             "trailing_return", "exception",
             "virtual", "const", "defaulted", "method_attributes", "ref_qualifier",
             "body", "attributes", "parameters",)


class FunctionDecl(Declaration):
    attrs = ("name", "constexpr", "return_type", "storage", "variadic", "inline",
             "trailing_return", "exception",
             "body", "attributes", "parameters")

class Throw(Node):
    attrs = ("args",)

class NoExcept(Node):
    attrs = ("repr",)

class NoThrow(Node):
    attrs = ()


class ClassTemplateDecl(Declaration):
    attrs = ("template_parameters", "decl",)


class ClassTemplateSpecializationDecl(CXXRecordDecl):
    attrs = ("template_arguments", "template_parameters",)


class ClassTemplatePartialSpecializationDecl(CXXRecordDecl):
    attrs = ("template_arguments", "template_parameters",)


class FunctionTemplateDecl(Declaration):
    attrs = ("template_parameters", "decl",)


class TemplateArgument(Node):
    attrs = ("type", "expr", "pack",)


class TemplateParmDecl(Declaration):
    attrs = ()

class TemplateTemplateParmDecl(TemplateParmDecl):
    attrs = ("name", "template_parameters",)


class TemplateTypeParmDecl(TemplateParmDecl):
    attrs = ("name", "tag", "default", "parameter_pack",)


class NonTypeTemplateParmDecl(TemplateParmDecl):
    attrs = ("type", "name", "default", "parameter_pack",)


class ParmVarDecl(Declaration):
    attrs = ("type", "name", "default", "attributes",)


class FieldDecl(Declaration):
    attrs = ("type", "name", "init", "bitwidth", "attributes", "type_qualifier",)

class CXXDefaultInitExpr(Declaration):
    attrs = ("expression",)

class ClassTag(Node):
    attrs = ()


class TypenameTag(Node):
    attrs = ()


# ------------------------------------------------------------------------------


class ConstantDeclaration(FieldDecl):
    attrs = ()


class VariableInitializer(Node):
    """
    A VariableInitializer is either an expression or an array initializer
    https://docs.oracle.com/javase/specs/jls/se8/html/jls-8.html#jls-8.3
    TODO This is not C++ !!!
    """
    attrs = ("expression", "array",)


class ArrayInitializer(Node):
    attrs = ("initializers", "comma",)


class VariableDeclaration(NonEmptyDeclaration):
    attrs = ("type", "declarators",)


class LocalVariableDeclaration(VariableDeclaration):
    attrs = ()


class VariableDeclarator(Declaration):
    attrs = ("name", "dimensions", "initializer")


class FormalParameter(NonEmptyDeclaration):
    attrs = ("type", "name", "dimensions", "varargs")


class InferredFormalParameter(Node):
    attrs = ('expression',)

# ------------------------------------------------------------------------------


class Statement(Node):
    attrs = ()


class DoStmt(Statement):
    attrs = ("cond", "body",)


class LocalVariableDeclarationStmt(Statement):
    attrs = ("variable",)


class TypeDeclarationStmt(Statement):
    attrs = ("declaration",)


class AttributedStmt(Statement):
    attrs = ("stmt", "attributes",)


class IfStmt(Statement):
    attrs = ("cond", "true_body", "false_body")


class ForStmt(Statement):
    attrs = ("init", "cond", "inc", "body")


class WhileStmt(Statement):
    attrs = ("cond", "body",)


class ContinueStmt(Statement):
    attrs = ()


class AssertStmt(Statement):
    attrs = ("condition", "value")


class LabelStmt(Statement):
    attrs = ("name", "stmt",)


class GotoStmt(Statement):
    attrs = ("target",)


class ContinueStmt(Statement):
    attrs = ("goto",)


class ThrowStmt(Statement):
    attrs = ("expression",)


class SynchronizedStmt(Statement):
    attrs = ("lock", "block")


class CXXTryStmt(Statement):
    attrs = ("body", "handlers",)


class CXXCatchStmt(Statement):
    attrs = ("decl", "body",)


class SwitchStmt(Statement):
    attrs = ("cond", "body",)


class BreakStmt(Statement):
    attrs = ()


class DefaultStmt(Statement):
    attrs = ("stmt",)


class BlockStmt(Statement):
    attrs = ("statements",)


# statemenents are the subnodes (from Node)
class CompoundStmt(Statement):
    attrs = ("stmts",)


class ReturnStmt(Statement):
    attrs = ("value",)


class NamespaceDecl(Declaration):
    attrs = ("name", "inline", "decls",)


class UsingDirectiveDecl(Declaration):
    attrs = ("name",)


class DeclStmt(Statement):
    attrs = ("decls",)


class StaticAssertDecl(Declaration):
    attrs = ("cond", "message",)


class VarDecl(Declaration):
    attrs = ("name", "constexpr", "type", "storage_class", "init_mode", "implicit",
             "referenced", "init", "attributes", "tls")


class TypedefDecl(Declaration):
    attrs = ("name", "type")


class TypeAliasTemplateDecl(Declaration):
    attrs = ("template_parameters", "decl",)


class TypeAliasDecl(Declaration):
    attrs = ("name", "type")


class UsingDecl(Declaration):
    attrs = ("name",)


class TypeRef(Node):
    attrs = ("name",)


#class NamespaceRef(Node):
    #attrs = ("name",)


class ExpressionStmt(Statement):
    attrs = ("expression",)


class GCCAsmStmt(Statement):
    attrs = ("string", "output_operands", "input_operands", "clobbers", "labels",)

class ConstrainedExpression(Node):
    attrs = ("expr", "constraint")

class CXXForRangeStmt(Statement):
    attrs = ("decl", "range", "body",)

class DeclsOrExpr(Node):
    attrs = ("decls", "expr",)

class DeclOrExpr(Node):
    attrs = ("decl", "expr",)


# ------------------------------------------------------------------------------


class TryResource(NonEmptyDeclaration):
    attrs = ("type", "name", "value")


class CatchClause(Statement):
    attrs = ("parameter", "block")


class CatchClauseParameter(NonEmptyDeclaration):
    attrs = ("types", "name")

# ------------------------------------------------------------------------------


class CaseStmt(Statement):
    attrs = ("pattern", "pattern_end", "stmt",)


class ForControl(Node):
    attrs = ("init", "condition", "update")


class EnhancedForControl(Node):
    attrs = ("var", "iterable")


class ExprStmt(Node):
    attrs = ("expr",)

# ------------------------------------------------------------------------------


class Expression(Node):
    attrs = ()


class DeclRefExpr(Expression):
    attrs = ("name",)


class DependentScopeDeclRefExpr(Expression):
    attrs = ("name",)


class CXXUnresolvedConstructExpr(Expression):
    attrs = ("type", "expr",)


class ParenListExpr(Expression):
    attrs = ("exprs",)


class SizeOfPackExpr(Expression):
    attrs = ("name",)


class PackExpansionExpr(Expression):
    attrs = ("expr",)


class UnresolvedLookupExpr(Expression):
    attrs = ('name',)


class PredefinedExpr(Expression):
    attrs = ("name",)


class OffsetOfExpr(Expression):
    attrs = ("type", "kinds",)


class VAArgExpr(Expression):
    attrs = ("expr", "type",)


class DumpedExpr(Expression):
    attrs = ("value",)


class SubstNonTypeTemplateParmExpr(Expression):
    attrs = ("decl", "expr",)

class OffsetOfField(Node):
    attrs = ("name",)


class OffsetOfArray(Node):
    attrs = ("index",)


class AddrLabelExpr(Expression):
    attrs = ("name",)


class ElementValueArrayInitializer(Expression):
    attrs = ("initializer",)


class UserDefinedLiteral(Expression):
    attrs = ("suffix", "expr",)


class LambdaExpr(Expression):
    attrs = ("parameters", "capture_exprs", "trailing_type", "exception",
             "variadic", "body", "attributes",)


class ReferenceTypeExpression(Expression):
    attrs = ("type",)


class BlockExpression(Expression):
    attrs = ("block",)


class NoExpression(Expression):
    attrs = ()


class Primary(Expression):
    attrs = ()
    #attrs = ("prefix_operators", "postfix_operators", "qualifier", "selectors")


class ParenExpr(Primary):
    attrs = ("expr",)


class Assignment(Primary):
    attrs = ("expressionl", "value", "type")


class BinaryOperator(Expression):
    attrs = ("opcode", "lhs", "rhs",)

class CompoundAssignOperator(Expression):
    attrs = ("opcode", "lhs", "rhs",)


class UnaryOperator(Expression):
    attrs = ("opcode", "expr", "postfix",)


class BinaryConditionalOperator(Expression):
    attrs = ("cond", "false_expr")


class OpaqueValueExpr(Expression):
    attrs = ("expr",)


class ConditionalOperator(Expression):
    attrs = ("cond", "true_expr", "false_expr")


class ChooseExpr(Expression):
    attrs = ("cond", "true_expr", "false_expr")


class ArraySubscriptExpr(Expression):
    attrs = ("base", "index",)


class AtomicExpr(Expression):
    attrs = ("name", "args")


class StmtExpr(Expression):
    attrs = ("stmt",)


class MethodReference(Primary):
    attrs = ("expression", "method", "type_arguments")


class LambdaExpression(Primary):
    attrs = ('parameter', 'parameters', 'body')


class CXXConstructExpr(Expression):
    attrs = ("args",)


class MaterializeTemporaryExpr(Expression):
    attrs = ("expr",)


class CXXBindTemporaryExpr(Expression):
    attrs = ("expr",)


class CXXNewExpr(Expression):
    attrs = ("type", "args", "placement", "array_size")


class CXXDeleteExpr(Expression):
    attrs = ("expr", "is_array",)


class UnaryExprOrTypeTraitExpr(Expression):
    attrs = ("name", "expr", "type")


class ImplicitCastExpr(Expression):
    attrs = ("type", "expr",)


class ExprWithCleanups(Expression):
    attrs = ("expr",)


# ------------------------------------------------------------------------------


class Identifier(Primary):
    attrs = ("id",)


class Literal(Primary):
    attrs = ("value",)


class CharacterLiteral(Literal):
    attrs = ()


class IntegerLiteral(Literal):
    attrs = ("type",)


class FloatingLiteral(Literal):
    attrs = ("type",)


class ImaginaryLiteral(Literal):
    attrs = ("type",)


class StringLiteral(Literal):
    attrs = ()


class CXXNullPtrLiteralExpr(Literal):
    attrs = ()


class CXXThisExpr(Primary):
    attrs = ("subnodes",)


class CXXThrowExpr(Expression):
    attrs = ("expr",)


class CXXTypeidExpr(Primary):
    attrs = ("expr", "type",)


class MemberExpr(Primary):
    attrs = ("name", "op", "expr")


class ConstantExpr(Primary):
    attrs = ("expr", "result",)


class CXXMemberCallExpr(Primary):
    attrs = ("bound_method", "args")


class CallExpr(Primary):
    attrs = ("callee", "args",)


class CXXOperatorCallExpr(Primary):
    attrs = ("left", "op", "right",)


class CXXBoolLiteralExpr(Primary):
    attrs = ("value",)


class CXXTemporaryObjectExpr(Node):
    attrs = ("type", "args",)


class CXXFunctionalCastExpr(Expression):
    attrs = ("type", "expr",)


class BuiltinBitCastExpr(Expression):
    attrs = ("type", "expr",)


class CXXStaticCastExpr(Expression):
    attrs = ("type", "expr", "value_category")

class CXXConstCastExpr(Expression):
    attrs = ("type", "expr", "value_category")


class CXXReinterpretCastExpr(Expression):
    attrs = ("type", "expr",)


class NullStmt(Statement):
    attrs = ()


class IndirectGotoStmt(Statement):
    attrs = ("expr",)


class Cast(Primary):
    attrs = ("type", "expression")


class FieldReference(Primary):
    attrs = ("field",)


class MemberReference(Primary):
    attrs = ("member",)


class Invocation(Primary):
    attrs = ("type_arguments", "arguments")


class ExplicitConstructorInvocation(Invocation):
    attrs = ()


class SuperConstructorInvocation(Invocation):
    attrs = ()


class MethodInvocation(Invocation):
    attrs = ("member",)


class SuperMethodInvocation(Invocation):
    attrs = ("member",)


class SuperMemberReference(Primary):
    attrs = ("member",)


class ArraySelector(Expression):
    attrs = ("index",)


class ClassReference(Primary):
    attrs = ("type",)


class VoidClassReference(ClassReference):
    attrs = ()

# ------------------------------------------------------------------------------


class Creator(Primary):
    attrs = ("type",)


class ArrayCreator(Creator):
    attrs = ("dimensions", "initializer")


class ClassCreator(Creator):
    attrs = ("constructor_type_arguments", "arguments", "body")


class InnerClassCreator(Creator):
    attrs = ("constructor_type_arguments", "arguments", "body")


class ClassBody(Node):
    attrs = ("declarations",)


class EmptyClassBody(Node):
    attrs = ()


class ImplicitValueInitExpr(Node):
    attrs = ()


class CXXConversionDecl(Declaration):
    attrs = ("name", "inline", "exception", "const", "body", "attributes", )

# ------------------------------------------------------------------------------

class UnresolvedUsingTypenameDecl(Declaration):
    attrs = "name",


class EnumConstantDecl(Declaration):
    attrs = ("name", "init",)


class EnumDecl(Declaration):
    attrs = ("name", "fields",)


class AnnotationMethod(NonEmptyDeclaration):
    attrs = ("name", "return_type", "dimensions", "default")


class EmptyDecl(Declaration):
    attrs = ()


class CStyleCastExpr(Expression):
    attrs = ("type", "expr",)


class FriendDecl(Declaration):
    attrs = ("decl", "raw_decl",)


class Base(Node):
    attrs = ("name", "access_spec", "virtual",)


class CXXStdInitializerListExpr(Node):
    attrs = ("subnodes",)

