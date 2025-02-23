/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "Analysis/Analyzer.h"
#include "CompilerOptions.h"
#include "Transform/DE/Kernel.h"
#include "Transform/AWB/Kernel.h"
#include "Transform/AWB/Host.h"
#include "Transform/AG/Kernel.h"
#include "Transform/AG/Host.h"
#include "Transform/DAG/Kernel.h"
#include "Transform/DAG/Host.h"
#include "Transform/SK/Kernel.h"
#include "Transform/TE/Kernel.h"
#include "Transform/TH/Kernel.h"
#include "Transform/TC/Kernel.h"
#include "Transform/TC/Host.h"


#include <map>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

class DivergenceEliminationASTConsumer : public ASTConsumer {

public:

    DivergenceEliminationASTConsumer(Rewriter &rewriter)
            : rewriter_(rewriter) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {

        TranslationUnitDecl* TU = Context.getTranslationUnitDecl();
        Analyzer analyzer(TU);

        DivergenceEliminationTransform deTransform(rewriter_, analyzer);
        deTransform.TraverseDecl(TU);

    }

private:

    Rewriter& rewriter_;

};

class KernelTransformASTConsumer : public ASTConsumer {

public:

    KernelTransformASTConsumer(Rewriter &rewriter, bool isCalledFromKernel)
            : rewriter_(rewriter), isCalledFromKernel_(isCalledFromKernel) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {

        TranslationUnitDecl* TU = Context.getTranslationUnitDecl();
        Analyzer analyzer(TU);

        switch(CompilerOptions::transformType()) {
            case CompilerOptions::AW:
            case CompilerOptions::AB:
            {   KernelAWBTransform kernelTransform(rewriter_, analyzer, isCalledFromKernel_);
                kernelTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::AG:
            {   KernelAGTransform kernelTransform(rewriter_, analyzer, isCalledFromKernel_);
                kernelTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::DAG:
            {   KernelDAGTransform kernelTransform(rewriter_, analyzer, isCalledFromKernel_);
                kernelTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::TE: {
                ThreadExtractionTransform kernelTransform(rewriter_, analyzer);
                kernelTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::SK: {
                KernelSerializationTransform kernelSerializationTransform(rewriter_, analyzer);
                kernelSerializationTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::TH: {
                KernelThresholdTransform kernelThresholdTransform(rewriter_, analyzer);
                kernelThresholdTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::TC_1:
            case CompilerOptions::TC_2:
            case CompilerOptions::TC_3: {
                ThreadCoarseningTransform threadCoarseningTransform(rewriter_, analyzer);
                threadCoarseningTransform.TraverseDecl(TU);
                break;
            }
            default: assert(0 && "Unreachable");
        }

    }

private:

    Rewriter& rewriter_;
    bool isCalledFromKernel_;

};

class HostTransformASTConsumer : public ASTConsumer {

public:

    HostTransformASTConsumer(Rewriter &rewriter)
            : rewriter_(rewriter) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {

        TranslationUnitDecl* TU = Context.getTranslationUnitDecl();
        Analyzer analyzer(TU);

        switch(CompilerOptions::transformType()) {
            case CompilerOptions::AW:
            case CompilerOptions::AB:
            {   HostAWBTransform hostTransform(rewriter_, analyzer);
                hostTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::AG:
            {   HostAGTransform hostTransform(rewriter_, analyzer);
                hostTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::DAG:
            {   HostDAGTransform hostTransform(rewriter_, analyzer);
                hostTransform.TraverseDecl(TU);
                break;
            }
            case CompilerOptions::TC_1:
            case CompilerOptions::TC_2:
            case CompilerOptions::TC_3: {
                ThreadCoarseningHostTransform hostTransform(rewriter_, analyzer);
                hostTransform.TraverseDecl(TU);
                break;
            }
            default: assert(0 && "Unreachable");
        }
    }

private:

    Rewriter& rewriter_;

};

class DivergenceEliminationFrontendAction : public ASTFrontendAction {
public:

    DivergenceEliminationFrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<DivergenceEliminationASTConsumer>(rewriter_);
    }

    void EndSourceFileAction() override { CompilerOptions::writeToOutputFile(rewriter_, CompilerOptions::OVERWRITE); }

private:
    Rewriter rewriter_;
};

class KernelFromKernelTransformFrontendAction : public ASTFrontendAction {
public:

    KernelFromKernelTransformFrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<KernelTransformASTConsumer>(rewriter_, true);
    }

    void EndSourceFileAction() override { CompilerOptions::writeToOutputFile(rewriter_, CompilerOptions::OVERWRITE); }

private:
    Rewriter rewriter_;
};

class KernelFromHostTransformFrontendAction : public ASTFrontendAction {
public:

    KernelFromHostTransformFrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<KernelTransformASTConsumer>(rewriter_, false);
    }

    void EndSourceFileAction() override { CompilerOptions::writeToOutputFile(rewriter_, CompilerOptions::APPEND); }

private:
    Rewriter rewriter_;
};

class HostTransformFrontendAction : public ASTFrontendAction {
public:

    HostTransformFrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<HostTransformASTConsumer>(rewriter_);
    }

    void EndSourceFileAction() override { CompilerOptions::writeToOutputFile(rewriter_, CompilerOptions::APPEND); }

private:
    Rewriter rewriter_;
};

int main(int argc, const char **argv) {
    auto op = CommonOptionsParser::create(argc, argv, CompilerOptions::KLAPCategory);
    ClangTool Tool(op->getCompilations(), op->getSourcePathList());
    int ret;
    if(CompilerOptions::transformType() == CompilerOptions::DE) {
        ret = Tool.run(newFrontendActionFactory<DivergenceEliminationFrontendAction>().get());
    } else if (
            CompilerOptions::transformType() == CompilerOptions::TE
            || CompilerOptions::transformType() == CompilerOptions::SK
            || CompilerOptions::transformType() == CompilerOptions::TH
            ) {
        ret = Tool.run(newFrontendActionFactory<KernelFromKernelTransformFrontendAction>().get());
    }  else if (CompilerOptions::transformType() == CompilerOptions::TC_1 || CompilerOptions::transformType() == CompilerOptions::TC_2 || CompilerOptions::transformType() == CompilerOptions::TC_3) {
        ret  = Tool.run(newFrontendActionFactory<KernelFromKernelTransformFrontendAction>().get());
        ret |= Tool.run(newFrontendActionFactory<  HostTransformFrontendAction>().get());
    }else {
        ret  = Tool.run(newFrontendActionFactory<KernelFromKernelTransformFrontendAction>().get());
        ret |= Tool.run(newFrontendActionFactory<  KernelFromHostTransformFrontendAction>().get());
        ret |= Tool.run(newFrontendActionFactory<            HostTransformFrontendAction>().get());
    }

    return ret;
}

