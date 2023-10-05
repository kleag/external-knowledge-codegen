#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LexDiagnostic.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/ASTNodeTraverser.h"
#include "clang/AST/AttrVisitor.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/JSONNodeDumper.h"
#include "clang/AST/Type.h"
#include "llvm/Support/JSON.h"

#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

static llvm::StringRef AtomicOpToStr(clang::AtomicExpr::AtomicOp op) {
  switch(op) {
#define BUILTIN(ID, TYPE, ATTRS)
#define ATOMIC_BUILTIN(ID, TYPE, ATTRS) case clang::AtomicExpr:: AO ## ID: return #ID;
#include "clang/Basic/Builtins.def"
  }
}

namespace clang {

// Custom, manual type visitor. this is producing the bare minimal information
// description of each type, as needed by the cpplang/parser.py.
// The type description is a strict subset of what's produced by the generic
// JSONDumper.
// This could/should rely on the visitor pattern provided by Clang, but I find
// it easier to implement this way.

static llvm::json::Object fullType(const ASTContext &Ctx, QualType T);

static llvm::json::Object templateArgument(const ASTContext &Ctx, TemplateArgument TA) {
  switch(TA.getKind()) {
    case TemplateArgument::ArgKind::Type:
      return fullType(Ctx, TA.getAsType());
    case TemplateArgument::ArgKind::Integral: {
        llvm::json::Object InnerValue;
        InnerValue["kind"] = "IntegerLiteral";
        InnerValue["inner_type"] = fullType(Ctx, TA.getIntegralType());
        SmallVector<char> Buffer;
        InnerValue["value"] = (TA.getAsIntegral().toString(Buffer), Buffer);
        return InnerValue;
      }
    case TemplateArgument::ArgKind::Template: {
        llvm::json::Object InnerExpr;
        std::string pretty_buffer;
        llvm::raw_string_ostream pretty_stream(pretty_buffer);
        TA.getAsTemplate().dump(pretty_stream);
        InnerExpr["kind"] = "DumpedExpr"; // template dumped as a string
        InnerExpr["value"] = pretty_buffer;
        return InnerExpr;
      }
    case TemplateArgument::ArgKind::Expression: {
        llvm::json::Object InnerExpr;
        std::string pretty_buffer;
        llvm::raw_string_ostream pretty_stream(pretty_buffer);
        TA.getAsExpr()->printPretty(pretty_stream, nullptr, PrintingPolicy(Ctx.getLangOpts()));
        InnerExpr["kind"] = "DumpedExpr"; // expression dumped as a string
        InnerExpr["value"] = pretty_buffer;
        return InnerExpr;
      }
    case TemplateArgument::ArgKind::Pack: {
        llvm::json::Object InnerPack;
        llvm::json::Array InnerPackElt;
        for(auto TAElt : TA.pack_elements()) {
          InnerPackElt.push_back(templateArgument(Ctx, TAElt));
        }
        InnerPack["kind"] = "TemplateArgumentPack";
        InnerPack["inner"] = std::move(InnerPackElt);
        return InnerPack;
      }
    default:
      assert(false && "unsupported template argument kind");
      return {};
  }
}

static llvm::json::Object fullType(const ASTContext &Ctx, const Type * Ty) {
  llvm::json::Object Ret;
  if(!Ty)
    return Ret;
  Ret["kind"] = (llvm::Twine(Ty->getTypeClassName()) + "Type").str();
  auto const& PP = Ctx.getPrintingPolicy();
  if(auto * BuiltinTy = dyn_cast<BuiltinType>(Ty)) {
    Ret["type"] = llvm::json::Object{{"qualType", BuiltinTy->getName(PP)}};
  }
  else if(auto * BitIntTy = dyn_cast<BitIntType>(Ty)) {
    Ret["size"] = BitIntTy->getNumBits();
    if(BitIntTy->isUnsigned())
      Ret["sign"] = "unsigned";
    else
      Ret["sign"] = "signed";
  }
  else if(auto * ConstantArrayTy = dyn_cast<ConstantArrayType>(Ty)) {
    Ret["size"] = ConstantArrayTy->getSize().getZExtValue();
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, ConstantArrayTy->getElementType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * DependentSizedArrayTy = dyn_cast<DependentSizedArrayType>(Ty)) {
    if(auto const* Size = DependentSizedArrayTy->getSizeExpr())
    {
      std::string pretty_buffer;
      llvm::raw_string_ostream pretty_stream(pretty_buffer);
      DependentSizedArrayTy->getSizeExpr()->printPretty(pretty_stream, nullptr, PrintingPolicy(Ctx.getLangOpts()));
      Ret["size_repr"] = pretty_buffer;
    }
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, DependentSizedArrayTy->getElementType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * VariableArrayTy = dyn_cast<VariableArrayType>(Ty)) {
    {
    std::string pretty_buffer;
    llvm::raw_string_ostream pretty_stream(pretty_buffer);
    VariableArrayTy->getSizeExpr()->printPretty(pretty_stream, nullptr, PrintingPolicy(Ctx.getLangOpts()));
    Ret["size_repr"] = pretty_buffer;
    }
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, VariableArrayTy->getElementType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * FunctionNoProtoTy = dyn_cast<FunctionNoProtoType>(Ty)) {
    if(FunctionNoProtoTy->isConst())
      Ret["isconst"] = true;
    if(FunctionNoProtoTy->getExtInfo().getNoReturn())
      Ret["isNoReturn"] = true;
  }
  else if(auto * FunctionProtoTy = dyn_cast<FunctionProtoType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, FunctionProtoTy->getReturnType()));
    for(auto ParamTy : FunctionProtoTy->param_types ())
      Inner.push_back(fullType(Ctx, ParamTy));
    Ret["inner"] = llvm::json::Value(std::move(Inner));

    if(FunctionProtoTy->isConst())
      Ret["isconst"] = true;
    if(FunctionProtoTy->getExtInfo().getNoReturn())
      Ret["isNoReturn"] = true;
    if(FunctionProtoTy->hasTrailingReturn())
      Ret["trailingReturn"] = true;
    switch(FunctionProtoTy->getRefQualifier()) {
      case RefQualifierKind::RQ_None:
        break;
      case RefQualifierKind::RQ_LValue:
        Ret["ref_qualifier"] = "LValue";
        break;
      case RefQualifierKind::RQ_RValue:
        Ret["ref_qualifier"] = "RValue";
        break;
    }

    if(FunctionProtoTy->hasExceptionSpec()) {
      auto ESI = FunctionProtoTy->getExceptionSpecInfo();
      llvm::json::Object ExceptionSpec;
      switch(ESI.Type) {
        case ExceptionSpecificationType::EST_None:
          assert(false && "should not happen");
          break;
        case ExceptionSpecificationType::EST_DynamicNone:
          ExceptionSpec["isDynamic"] = true;
          break;
        case ExceptionSpecificationType::EST_Dynamic:
          ExceptionSpec["isDynamic"] = true;
          {
            llvm::json::Array Inner;
            for(QualType QT : ESI.Exceptions)
              Inner.push_back(QT.getAsString());
            ExceptionSpec["inner"] = llvm::json::Value(std::move(Inner));
          }
          break;
        case ExceptionSpecificationType::EST_NoThrow:
          ExceptionSpec["isNoThrow"] = true;
          break;
        case ExceptionSpecificationType::EST_NoexceptFalse:
        case ExceptionSpecificationType::EST_NoexceptTrue:
        case ExceptionSpecificationType::EST_DependentNoexcept:
          // FIXME: parsing the expression at this level would be great
          // but this is rather complex for a feature that's scarcely used, so
          // rely on string representation as of now.
          {
          std::string pretty_buffer;
          llvm::raw_string_ostream pretty_stream(pretty_buffer);
          ESI.NoexceptExpr->printPretty(pretty_stream, nullptr, PrintingPolicy(Ctx.getLangOpts()));
          ExceptionSpec["expr_repr"] = pretty_buffer;
          }
        case ExceptionSpecificationType::EST_BasicNoexcept:
          ExceptionSpec["isBasic"] = true;
          break;
        case ExceptionSpecificationType::EST_Unevaluated:
        case ExceptionSpecificationType::EST_Uninstantiated:
          break;
        default:
          llvm::errs() << ESI.Type << "\n";
          assert(0 && "Not implemented Yet");
      }

      Ret["exception_spec"] = llvm::json::Value(std::move(ExceptionSpec));
    }
  }
  else if(auto * ReferenceTy = dyn_cast<ReferenceType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, ReferenceTy->getPointeeType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto* TemplateTypeParmTy = dyn_cast<TemplateTypeParmType>(Ty)) {
    if(auto * Identifier = TemplateTypeParmTy->getIdentifier())
      Ret["name"] = Identifier->getName();
    Ret["depth"] = TemplateTypeParmTy->getDepth();
    Ret["index"] = TemplateTypeParmTy->getIndex();
  }
  else if(auto * ParenTy = dyn_cast<ParenType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, ParenTy->getInnerType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * PointerTy = dyn_cast<PointerType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, PointerTy->getPointeeType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * ComplexTy = dyn_cast<ComplexType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, ComplexTy->getElementType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * TypedefTy = dyn_cast<TypedefType>(Ty)) {
    Ret["name"] = TypedefTy->getDecl()->getName();
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, TypedefTy->getDecl()->getUnderlyingType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * UsingTy = dyn_cast<UsingType>(Ty)) {
    Ret["name"] = UsingTy->getFoundDecl()->getName();
  }
  else if(auto* TypeOfExprTy = dyn_cast<TypeOfExprType>(Ty)) {
    // FIXME: Just as for exception spec, we're hitting the limit of the approach
    // here as we would like to parse the expression instead of just getting its
    // raw representation, but we don't have support for that feature yet.
    std::string pretty_buffer;
    llvm::raw_string_ostream pretty_stream(pretty_buffer);
    TypeOfExprTy->getUnderlyingExpr()->printPretty(pretty_stream, nullptr, PrintingPolicy(Ctx.getLangOpts()));
    Ret["expr_repr"] = pretty_buffer;
  }
  else if(auto* DecltypeTy = dyn_cast<DecltypeType>(Ty)) {
    // FIXME: Just as for exception spec, we're hitting the limit of the approach
    // here as we would like to parse the expression instead of just getting its
    // raw representation, but we don't have support for that feature yet.
    std::string pretty_buffer;
    llvm::raw_string_ostream pretty_stream(pretty_buffer);
    DecltypeTy->getUnderlyingExpr()->printPretty(pretty_stream, nullptr, PrintingPolicy(Ctx.getLangOpts()));
    Ret["expr_repr"] = pretty_buffer;
  }
  else if(auto * AutoTy = dyn_cast<AutoType>(Ty)) {
    switch(AutoTy->getKeyword()) {
      case AutoTypeKeyword::Auto:
        Ret["keyword"] = "auto";
        break;
      case AutoTypeKeyword::DecltypeAuto:
        Ret["keyword"] = "decltype(auto)";
        break;
      case AutoTypeKeyword::GNUAutoType:
        Ret["keyword"] = "__auto_type";
        break;
    }
  }
  else if(auto * RecordTy = dyn_cast<RecordType>(Ty)) {
    Ret["decl"] = llvm::json::Object({{"name", RecordTy->getDecl()->getName()}});
  }
  else if(auto * EnumTy = dyn_cast<EnumType>(Ty)) {
    Ret["decl"] = llvm::json::Object({{"name", EnumTy->getDecl()->getName()}});
  }
  else if(auto * DecayedTy = dyn_cast<DecayedType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, DecayedTy->getOriginalType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * IncompleteArrayTy = dyn_cast<IncompleteArrayType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, IncompleteArrayTy->getElementType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto * ElaboratedTy = dyn_cast<ElaboratedType>(Ty)) {
    QualType QT = ElaboratedTy->getNamedType();
    SplitQualType SQT = QT.split();
    Ret["qualifiers"] = SQT.Quals.getAsString();
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, SQT.Ty));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto *DependentNameTy = dyn_cast<DependentNameType>(Ty)) {
    std::string qual_dump;
    llvm::raw_string_ostream qual_stream(qual_dump);
    Ret["nested_name"] = (DependentNameTy->getQualifier()->dump(qual_stream), qual_dump);
    Ret["attribute_name"] = DependentNameTy->getIdentifier()->getName();
  }
  else if(auto * VectorTy = dyn_cast<VectorType>(Ty)) {
    QualType QT = VectorTy->getElementType();
    Ret["size"] = VectorTy->getNumElements() * Ctx.getTypeSizeInChars(VectorTy->getElementType()).getQuantity();
    SplitQualType SQT = QT.split();
    Ret["qualifiers"] = SQT.Quals.getAsString();
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, SQT.Ty));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto* SubstTemplateTypeParmTy = dyn_cast<SubstTemplateTypeParmType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, SubstTemplateTypeParmTy->getReplacementType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto* PackExpansionTy = dyn_cast<PackExpansionType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, PackExpansionTy->getPattern()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto* TemplateSpecializationTy = dyn_cast<TemplateSpecializationType>(Ty)) {
    Ret["name"] = TemplateSpecializationTy->getTemplateName().getAsTemplateDecl()->getName();
    llvm::json::Array Inner;
    for(auto& TA : TemplateSpecializationTy->template_arguments()) {
      Inner.push_back(templateArgument(Ctx, TA));
    }
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto *InjectedClassNameTy = dyn_cast<InjectedClassNameType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, InjectedClassNameTy->getInjectedSpecializationType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto* MemberPointerTy = dyn_cast<MemberPointerType>(Ty)) {
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, MemberPointerTy->getClass()));
    Inner.push_back(fullType(Ctx, MemberPointerTy->getPointeeType()));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
  }
  else if(auto* UnresolvedUsingTy = dyn_cast<UnresolvedUsingType>(Ty)) {
    Ret["name"] = UnresolvedUsingTy->getDecl()->getName();
  }
  else {
    Ty->dump();
    assert(false && "unsupported type");
  }
  return Ret;
}

static llvm::json::Object fullType(const ASTContext &Ctx, QualType T) {
  SplitQualType SQT = T.split();
  if (SQT.Quals.empty()) {
    return fullType(Ctx, SQT.Ty);
  }
  else {
    llvm::json::Object Ret;
    Ret["qualifiers"] = SQT.Quals.getAsString();
    llvm::json::Array Inner;
    Inner.push_back(fullType(Ctx, SQT.Ty));
    Ret["inner"] = llvm::json::Value(std::move(Inner));
    Ret["kind"] =  "QualType";
    return Ret;
  }
}


class JSONNodeTypeDumper
    : public ConstStmtVisitor<JSONNodeTypeDumper>,
      public ConstAttrVisitor<JSONNodeTypeDumper>,
      public TypeVisitor<JSONNodeTypeDumper>,
      public ConstDeclVisitor<JSONNodeTypeDumper>,
      public NodeStreamer {
  friend class JSONTypeDumper;

  const SourceManager &SM;
  ASTContext& Ctx;
  PrintingPolicy PrintPolicy;

  using InnerStmtVisitor = ConstStmtVisitor<JSONNodeTypeDumper>;
  using InnerAttrVisitor = ConstAttrVisitor<JSONNodeTypeDumper>;
  using InnerTypeVisitor = TypeVisitor<JSONNodeTypeDumper>;
  using InnerDeclVisitor = ConstDeclVisitor<JSONNodeTypeDumper>;


  std::string createPointerRepresentation(const void *Ptr) {
    // Because JSON stores integer values as signed 64-bit integers, trying to
    // represent them as such makes for very ugly pointer values in the resulting
    // output. Instead, we convert the value to hex and treat it as a string.
    return "0x" + llvm::utohexstr(reinterpret_cast<uint64_t>(Ptr), true);
  }

public:
  JSONNodeTypeDumper(raw_ostream &OS, const SourceManager &SrcMgr, ASTContext &Ctx,
                 const PrintingPolicy &PrintPolicy)
      : NodeStreamer(OS), SM(SrcMgr), Ctx(Ctx),
        PrintPolicy(PrintPolicy)
{}

  void Visit(const Attr *A) {
    if(const auto * AA = dyn_cast<AliasAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(AA));
      JOS.attribute("aliasee", AA->getAliasee());
    }
    else if(const auto * CA = dyn_cast<CleanupAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(CA));
      JOS.attribute("cleanup_function", CA->getFunctionDecl()->getName());
    }
    else if(const auto * DA = dyn_cast<DeprecatedAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(DA));
      JOS.attribute("deprecation_message", DA->getMessage());
    }
    else if(const auto * UA = dyn_cast<UnavailableAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(UA));
      JOS.attribute("deprecation_message", UA->getMessage());
    }
    else if(const auto * SA = dyn_cast<SectionAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(SA));
      JOS.attribute("section_name", SA->getName());
    }
    else if(const auto * TA = dyn_cast<TLSModelAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(TA));
      JOS.attribute("tls_model", TA->getModel());
    }
    else if(const auto * VA = dyn_cast<VisibilityAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(VA));
      JOS.attribute("visibility", VisibilityAttr::ConvertVisibilityTypeToStr(VA->getVisibility()));
    }
    else if(const auto * AAA = dyn_cast<AllocAlignAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(AAA));
      JOS.attribute("source_index", AAA->getParamIndex().getSourceIndex());
    }
    else if(const auto * ASA = dyn_cast<AllocSizeAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(ASA));
      JOS.attribute("size_index", ASA->getElemSizeParam().getSourceIndex());
      if(ASA->getNumElemsParam().isValid())
        JOS.attribute("nmemb_index", ASA->getNumElemsParam().getSourceIndex());
    }
    else if(const auto * ConsA = dyn_cast<ConstructorAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(ConsA));
      if(ConsA->getPriority() != ConstructorAttr::DefaultPriority)
        JOS.attribute("priority", ConsA->getPriority());
    }
    else if(const auto * DesA = dyn_cast<DestructorAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(DesA));
      if(DesA->getPriority() != DestructorAttr::DefaultPriority)
        JOS.attribute("priority", DesA->getPriority());
    }
    else if(const auto * EA = dyn_cast<ErrorAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(EA));
      JOS.attribute("message", EA->getUserDiagnostic());
    }
    else if(const auto * FA = dyn_cast<FormatAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(FA));
      JOS.attribute("archetype", FA->getType()->getName());
      JOS.attribute("fmt_index", FA->getFormatIdx());
      JOS.attribute("vargs_index", FA->getFirstArg());
    }
    else if(const auto * FAA = dyn_cast<FormatArgAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(FAA));
      JOS.attribute("fmt_index", FAA->getFormatIdx().getSourceIndex());
    }
    else if(const auto * IFA = dyn_cast<IFuncAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(IFA));
      JOS.attribute("name", IFA->getResolver());
    }
    else if(const auto * NSA = dyn_cast<NoSanitizeAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(NSA));
      JOS.attribute("options", llvm::json::Array(NSA->sanitizers()));
    }
    else if(const auto * NNA = dyn_cast<NonNullAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(NNA));
      llvm::json::Array Indices;
      for(auto const& Arg : NNA->args())
        Indices.push_back(Arg.getSourceIndex());
      JOS.attribute("indices", std::move(Indices));
    }
    else if(const auto * PFEA = dyn_cast<PatchableFunctionEntryAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(PFEA));
      JOS.attribute("count", PFEA->getCount());
      if(PFEA->getOffset() != 0)
        JOS.attribute("offset", PFEA->getOffset());
    }
    else if(const auto * SentA = dyn_cast<SentinelAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(SentA));
      if(SentA->getSentinel() != 0)
        JOS.attribute("value", SentA->getSentinel());
      if(SentA->getNullPos() != 0)
        JOS.attribute("offset", SentA->getNullPos());
    }
    else if(const auto * WRA = dyn_cast<WeakRefAttr>(A)) {
      JOS.attribute("node_id", createPointerRepresentation(WRA));
      if(!WRA->getAliasee().empty())
        JOS.attribute("name", WRA->getAliasee());
    }
    InnerAttrVisitor::Visit(A);
  }

  void Visit(const Stmt *S) {
    if(!S) return;
    if (const auto * GAS = dyn_cast<GCCAsmStmt>(S)) {
      JOS.attribute("node_id", createPointerRepresentation(S));
      if(const auto * SL =  GAS->getAsmString()) {
        JOS.attribute("asm_string", SL->getString());

        if(unsigned numOutputs = GAS->getNumOutputs()) {
          JOS.attributeArray("output_constraints", [GAS,numOutputs,this] {
              for(unsigned i = 0; i != numOutputs; ++i) {
                llvm::json::Object Val{
                  {"id", createPointerRepresentation(GAS->getOutputExpr(i))},
                  {"constraint", GAS->getOutputConstraint(i)},
                };
                JOS.value(std::move(Val));
              }
          });
        }
        if(unsigned numInputs = GAS->getNumInputs()) {
          JOS.attributeArray("input_constraints", [GAS,numInputs,this] {
              for(unsigned i = 0; i != numInputs; ++i) {
                llvm::json::Object Val{
                  {"id", createPointerRepresentation(GAS->getInputExpr(i))},
                  {"constraint", GAS->getInputConstraint(i)},
                };
                JOS.value(std::move(Val));
              }
          });
        }
        if(unsigned numClobbers = GAS->getNumClobbers()) {
          JOS.attributeArray("clobbers", [GAS,numClobbers,this] {
              for(unsigned i = 0; i != numClobbers; ++i) {
                llvm::json::Object Val{
                  {"clobber", GAS->getClobber(i)},
                };
                JOS.value(std::move(Val));
              }
          });
        }
        if(unsigned numLabels = GAS->getNumLabels()) {
          JOS.attributeArray("labels", [GAS,numLabels,this] {
              for(unsigned i = 0; i != numLabels; ++i) {
                llvm::json::Object Val{
                  {"label", GAS->getLabelName(i)},
                };
                JOS.value(std::move(Val));
              }
          });
        }
      }
    }

    InnerStmtVisitor::Visit(S);
  }

  void Visit(const Type *T);
  void Visit(QualType T);
  void VisitExpr(const Expr *E) {
    if(E) {
      if(const auto * UEoTTE = dyn_cast<UnaryExprOrTypeTraitExpr>(E)) {
        if(UEoTTE->isArgumentType()) {
          JOS.attribute("node_id", createPointerRepresentation(E));
            JOS.attributeBegin("node_inner");
            JOS.arrayBegin();
          JOS.objectBegin();
          Visit(UEoTTE->getArgumentType());
          JOS.objectEnd();
            JOS.arrayEnd();
            JOS.attributeEnd();
        }
      }
      else if(const auto * AE = dyn_cast<AtomicExpr>(E)) {
        JOS.attribute("node_id", createPointerRepresentation(AE));
        JOS.attribute("name", AtomicOpToStr(AE->getOp()));
      }
      else if(const auto * OOE = dyn_cast<OffsetOfExpr>(E)) {
        JOS.attribute("node_id", createPointerRepresentation(OOE));

        JOS.attributeBegin("expr_inner");
        JOS.arrayBegin();

        JOS.objectBegin();
        Visit(OOE->getTypeSourceInfo()->getType());
        JOS.objectEnd();

        for(int i = 0; i < OOE->getNumComponents(); ++i) {
          JOS.objectBegin();
          OffsetOfNode ON = OOE->getComponent(i);
          switch(ON.getKind()) {
            case OffsetOfNode::Array:
              JOS.attributeBegin("kind");
              JOS.value(llvm::json::Value("OffsetOfArray"));
              break;
            case OffsetOfNode::Field:
              JOS.attributeBegin("field");
              JOS.value(llvm::json::Value(ON.getFieldName()->getName()));
              JOS.attributeEnd();
              JOS.attributeBegin("kind");
              JOS.value(llvm::json::Value("OffsetOfField"));
              break;
            default:
              OOE->dump();
              assert(false && "not implemented yet offsetof kind");
          }
          JOS.attributeEnd();
          JOS.objectEnd();
        }

        JOS.arrayEnd();

        JOS.attributeEnd();
      }
      else if(const auto * TypeidE = dyn_cast<CXXTypeidExpr>(E)) {
        if(TypeidE->isTypeOperand()) {
          JOS.attribute("node_id", createPointerRepresentation(E));
            JOS.attributeBegin("node_inner");
            JOS.arrayBegin();
          JOS.objectBegin();
          Visit(TypeidE-> getTypeOperand (Ctx));
          JOS.objectEnd();
            JOS.arrayEnd();
            JOS.attributeEnd();
        }
      }
      else {
        JOS.attribute("node_id", createPointerRepresentation(E));
          JOS.attributeBegin("node_inner");
          JOS.arrayBegin();
        JOS.objectBegin();
        Visit(E->getType());
        JOS.objectEnd();
          JOS.arrayEnd();
          JOS.attributeEnd();
      }
    }
  }

  void Visit(const Decl *D) {
    if(!D) return;
    if (const auto *VD = dyn_cast<ValueDecl>(D)) {
      JOS.attribute("node_id", createPointerRepresentation(D));
      JOS.attributeBegin("node_inner");
      JOS.arrayBegin();
      JOS.objectBegin();
      Visit(VD->getType());
      JOS.objectEnd();
      JOS.arrayEnd();
      if (const auto *CD = dyn_cast<CXXConstructorDecl>(D)) {
        JOS.attribute("isExplicit", CD->isExplicit());
      }
      JOS.attributeEnd();
    }
    InnerDeclVisitor::Visit(D);
  }

  void Visit(const GCCAsmStmt * S) {}

  void Visit(const comments::Comment *C, const comments::FullComment *FC) {}
  void Visit(const TemplateArgument &TA, SourceRange R = {},
             const Decl *From = nullptr, StringRef Label = {}) {}
  void Visit(const CXXCtorInitializer *Init) {}
  void Visit(const OMPClause *C) {}
  void Visit(const BlockDecl::Capture &C) {}
  void Visit(const GenericSelectionExpr::ConstAssociation &A) {}
  void Visit(const concepts::Requirement *R) {}
  void Visit(const APValue &Value, QualType Ty) {}


llvm::json::Object createQualType(QualType QT, bool Desugar = true) {
  SplitQualType SQT = QT.split();
  llvm::json::Object Ret{{"qualType", QualType::getAsString(SQT, PrintPolicy)}};

  if (Desugar && !QT.isNull()) {
    SplitQualType DSQT = QT.getSplitDesugaredType();
    if (DSQT != SQT)
      Ret["desugaredQualType"] = QualType::getAsString(DSQT, PrintPolicy);
    if (const auto *TT = QT->getAs<TypedefType>())
      Ret["typeAliasDeclId"] = createPointerRepresentation(TT->getDecl());
  }
  return Ret;
}


  void writeBareDeclRef(const Decl *D) {
  JOS.attribute("id", createPointerRepresentation(D));
  if (!D)
    return;

  JOS.attribute("kind", (llvm::Twine(D->getDeclKindName()) + "Decl").str());
  if (const auto *ND = dyn_cast<NamedDecl>(D))
    JOS.attribute("name", ND->getDeclName().getAsString());
  if (const auto *VD = dyn_cast<ValueDecl>(D))
    JOS.attribute("type", createQualType(VD->getType()));
  }

};

class JSONTypeDumper : public ASTNodeTraverser<JSONTypeDumper, JSONNodeTypeDumper> {
  JSONNodeTypeDumper NodeDumper;
  template <typename SpecializationDecl>
  void writeTemplateDeclSpecialization(const SpecializationDecl *SD,
                                       bool DumpExplicitInst,
                                       bool DumpRefOnly) {
    bool DumpedAny = false;
    for (const auto *RedeclWithBadType : SD->redecls()) {
      // FIXME: The redecls() range sometimes has elements of a less-specific
      // type. (In particular, ClassTemplateSpecializationDecl::redecls() gives
      // us TagDecls, and should give CXXRecordDecls).
      const auto *Redecl = dyn_cast<SpecializationDecl>(RedeclWithBadType);
      if (!Redecl) {
        // Found the injected-class-name for a class template. This will be
        // dumped as part of its surrounding class so we don't need to dump it
        // here.
        assert(isa<CXXRecordDecl>(RedeclWithBadType) &&
               "expected an injected-class-name");
        continue;
      }

      switch (Redecl->getTemplateSpecializationKind()) {
      case TSK_ExplicitInstantiationDeclaration:
      case TSK_ExplicitInstantiationDefinition:
        if (!DumpExplicitInst)
          break;
        [[fallthrough]];
      case TSK_Undeclared:
      case TSK_ImplicitInstantiation:
        if (DumpRefOnly)
          NodeDumper.AddChild([=] { NodeDumper.writeBareDeclRef(Redecl); });
        else
          Visit(Redecl);
        DumpedAny = true;
        break;
      case TSK_ExplicitSpecialization:
        break;
      }
    }

    // Ensure we dump at least one decl for each specialization.
    if (!DumpedAny)
      NodeDumper.AddChild([=] { NodeDumper.writeBareDeclRef(SD); });
  }

  template <typename TemplateDecl>
  void writeTemplateDecl(const TemplateDecl *TD, bool DumpExplicitInst) {
    // FIXME: it would be nice to dump template parameters and specializations
    // to their own named arrays rather than shoving them into the "inner"
    // array. However, template declarations are currently being handled at the
    // wrong "level" of the traversal hierarchy and so it is difficult to
    // achieve without losing information elsewhere.

    dumpTemplateParameters(TD->getTemplateParameters());

    Visit(TD->getTemplatedDecl());

    for (const auto *Child : TD->specializations())
      writeTemplateDeclSpecialization(Child, DumpExplicitInst,
                                      !TD->isCanonicalDecl());
  }

public:
  JSONTypeDumper(raw_ostream &OS, const SourceManager &SrcMgr, ASTContext &Ctx,
             const PrintingPolicy &PrintPolicy,
             const comments::CommandTraits *Traits)
      : NodeDumper(OS, SrcMgr, Ctx, PrintPolicy) {}

  JSONNodeTypeDumper &doGetNodeDelegate() { return NodeDumper; }
  void VisitFunctionTemplateDecl(const FunctionTemplateDecl *FTD) {
    writeTemplateDecl(FTD, true);
  }
  void VisitClassTemplateDecl(const ClassTemplateDecl *CTD) {
    writeTemplateDecl(CTD, false);
  }
  void VisitVarTemplateDecl(const VarTemplateDecl *VTD) {
    writeTemplateDecl(VTD, false);
  }

};

void JSONNodeTypeDumper::Visit(const Type *T) {
  if(!T) return;
  if(auto * DependentNameTy = dyn_cast<DependentNameType>(T)) {
    JOS.attribute("node_id", createPointerRepresentation(DependentNameTy));
    std::string qual_dump;
    llvm::raw_string_ostream qual_stream(qual_dump);
    JOS.attribute("nested_name", (DependentNameTy->getQualifier()->dump(qual_stream), qual_dump));
    JOS.attribute("attribute_name", DependentNameTy->getIdentifier()->getName());
  }
  else if(auto * TemplateSpecializationTy = dyn_cast<TemplateSpecializationType>(T)) {
    JOS.attribute("node_id", createPointerRepresentation(TemplateSpecializationTy));
    JOS.attributeBegin("templateArgumentsExtra");
    JOS.arrayBegin();
    for(auto TA : TemplateSpecializationTy->template_arguments()) {
      if(TA.getKind() == TemplateArgument::ArgKind::Template) {
        std::string qual_dump;
        llvm::raw_string_ostream qual_stream(qual_dump);
        JOS.value((TA.getAsTemplate().dump(qual_stream), qual_dump));
      }
    }
    JOS.arrayEnd();
    JOS.attributeEnd();
  }
  InnerTypeVisitor::Visit(T);
}

void JSONNodeTypeDumper::Visit(QualType T) {
  llvm::json::Value V = fullType(Ctx, T);
  for(auto KV : *V.getAsObject())
    JOS.attribute(KV.first, KV.second);
}


class JSONTypeDumperConsumer : public ASTConsumer {
  std::unique_ptr<JSONTypeDumper> type_dumper;
  std::string type_dump;
  llvm::raw_string_ostream type_stream;

  std::unique_ptr<JSONDumper> ast_dumper;
  std::string ast_dump;
  llvm::raw_string_ostream ast_stream;

public:
  JSONTypeDumperConsumer() : type_stream(type_dump), ast_stream(ast_dump) {}

  void Initialize (ASTContext &Ctx) override {
    type_dumper.reset(new JSONTypeDumper(
          type_stream,
          Ctx.getSourceManager(),
          Ctx,
          Ctx.getPrintingPolicy(),
          &Ctx.getCommentCommandTraits()));

    ast_dumper.reset(new JSONDumper(
          ast_stream,
          Ctx.getSourceManager(),
          Ctx,
          Ctx.getPrintingPolicy(),
          &Ctx.getCommentCommandTraits()));

  }

  void HandleTranslationUnit (ASTContext &Ctx) override {
    type_dumper->Visit(Ctx.getTranslationUnitDecl());
    ast_dumper->Visit(Ctx.getTranslationUnitDecl());

    // This create a dictionary with two entries, first one for the type summary
    // as built by `JSONTypeDumper`, second one is the AST dump as built by
    // `JSONDumper`. The link between the two entries is made through the id
    // fields.
    llvm::outs() << "{\n\"TypeSummary\":[\n"
                 << type_dump
                 << "],\n\"Content\":\n"
                 << ast_dump
                 << "}";
  }
};

class JSONTypeDumperAction : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<JSONTypeDumperConsumer>();
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

}

using namespace clang;

static FrontendPluginRegistry::Add<JSONTypeDumperAction>
X("dump-ast-types", "dump AST types in JSON format");
