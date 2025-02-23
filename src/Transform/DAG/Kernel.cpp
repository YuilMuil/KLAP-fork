/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>

#include "Kernel.h"
#include "Analysis/KernelCallFinder.h"
#include "Driver/CompilerOptions.h"
#include "Utils/Utils.h"

using namespace clang;

KernelDAGTransform::KernelDAGTransform(Rewriter& rewriter, Analyzer& analyzer, bool isCalledFromKernel)
    : rewriter_(rewriter), analyzer_(analyzer), isCalledFromKernel_(isCalledFromKernel) {}

bool KernelDAGTransform::VisitFunctionDecl(FunctionDecl *f) {

    if(f->getAttr<CUDAGlobalAttr>()) {
        if(isCalledFromKernel_) {

            /*
             * Transform the signature of the original kernel:
             *
             *      __global__ void kernel_name(T1 p1, T2 p2)
             *
             * into the signature of the transformed kernel called from a kernel:
             *
             *      __global__ void kernel_name_kernel_k(T1* p1_array, T2* p2_array,
             *                      unsigned int* __gDim_array, unsigned int* __bDim_array,
             *                      unsigned int __parentSize, __GridMemPool __memPool)
             *
             */

            std::stringstream ssPrologue, ssEpilogue;
            std::vector<bool> isScalarParam(f->getNumParams());
            for(unsigned int p = 0; p < f->getNumParams(); ++p) {
                isScalarParam[p] = CompilerOptions::scalarizeInvariantParameters() && analyzer_.isParamInvariant(f, p);
            }

            // Declare and initialize commonly used variables
            std::stringstream ssParams;
            bool isScalarBlockDim = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(f);

            // Append _kernel_k to kernel name
            SourceLocation nameLoc = f->getNameInfo().getLoc();
            rewriter_.InsertTextAfterToken(nameLoc, "_kernel_k");

            // Change parameters to parameter arrays
            for(unsigned int p = 0; p < f->getNumParams(); ++p) {
                ParmVarDecl* param = f->getParamDecl(p);
                ssParams << param->getType().getAsString() << " ";

                if (!isScalarParam[p]) {
                    ssParams << "* " << param->getNameAsString() << "_array";
                } else {
                    ssParams <<  param->getNameAsString();
                }

                ssParams << ", ";
            }

            // Add numBlocks and numThreads parameter arrays
            ssParams << "unsigned int* __gDim_array, ";
//            if(!isScalarBlockDim) {
//                ssParams << "unsigned int* __bDim_array, ";
//            }

            // Add parent size (needed for searching __gDim_array)
            ssParams << "unsigned int __parentSize, ";

            // Add memory pool
            ssParams << "__GridMemPool __memPool";

            // Update parameters
            SourceLocation beginParamLoc = f->getParamDecl(0)->getSourceRange().getBegin();
            SourceLocation endParamLoc =
                f->getParamDecl(f->getNumParams() - 1)->getSourceRange().getEnd();
            rewriter_.ReplaceText(SourceRange(beginParamLoc, endParamLoc), ssParams.str());

            if(f->doesThisDeclarationHaveABody()) {

                /*
                 * Insert code around original kernel body to extract parameters:
                 *
                 *      unsigned int __parentIdx = __find_parent_idx_kernel(__gDim_array, blockIdx.x, __parentSize);
                 *
                 *      T1 p1 = p1_array[__parentIdx];
                 *      T2 p2 = p2_array[__parentIdx];
                 *
                 *      unsigned int __gridDim_x =
                 *                      __gDim_array[__parentIdx] - ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);
                 *      unsigned int __blockDim_x = __bDim_array[__parentIdx];
                 *      unsigned int __blockIdx_x = blockIdx.x -
                 *                          ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);
                 *
                 *      if(threadIdx.x < __blockDim_x) {
                 *          <original kernel body>
                 *      }
                 *
                 */

                // Declare and initialize commonly used variables


                // Find parent index
                ssPrologue << "unsigned int __parentIdx = __find_parent_idx_kernel(__gDim_array, blockIdx.x, __parentSize);\n";

                // Load parameters
                for(unsigned int p = 0; p < f->getNumParams(); ++p) {
                    ParmVarDecl* param = f->getParamDecl(p);
                    if(isScalarParam[p]) {
                        continue;
                    }
                    ssPrologue << param->getType().getAsString() << " " << param->getNameAsString() << " = ";
                    ssPrologue << param->getNameAsString() << "_array[__parentIdx];\n";
                }

                // Compute __gridDim_x, __blockDim_x, and __blockIdx_x
                if(analyzer_.usesGridDim(f)) {
                    ssPrologue << "unsigned int __gridDim_x = __gDim_array[__parentIdx] - ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);\n";
                }
                ssPrologue << "unsigned int __blockDim_x = ";
//                if(isScalarBlockDim) {
                    ssPrologue << "blockDim.x;\n";
//                } else {
//                    ssPrologue << "__bDim_array[__parentIdx];\n";
//                }
                ssPrologue << "unsigned int __blockIdx_x = blockIdx.x - ((__parentIdx == 0)?0:__gDim_array[__parentIdx - 1]);\n";

                // Guard kernel body with bounds check
                ssPrologue << "if(threadIdx.x < __blockDim_x) {\n";

                SourceLocation beginBody = cast<CompoundStmt>(f->getBody())->body_front()->getBeginLoc();
                rewriter_.InsertTextBefore(beginBody, ssPrologue.str());

                // Close guard
                ssEpilogue << ";\n}\n}\n";

                /*
                 *   void kernel_name_kernel_k_h(unsigned int __gDim, unsigned int __bDim, unsigned int __smem, T1* p1_array, T2* p2_array, unsigned int* __gDim_array, unsigned int* __bDim_array, unsigned int __parentSize, __GridMemPool __memPoolHost) {
                 *
                 *      if(__gDim > 0) {
                 *
                 *          __GridMemPool __memPoolDevice = __memPoolHost;
                 *
                 *          // The following code is generated for each child kernel of kernel_name, where the child kernel signature is:
                 *          //         __global__ void child_name(S1 q1, S2 q2)
                 *          S1* q1_array_child = __memPoolHost.grid_allocate<S1>(__gDim*__bDim);
                 *          S2* q2_array_child = __memPoolHost.grid_allocate<S2>(__gDim*__bDim);
                 *          unsigned int* gDim_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);
                 *          unsigned int* bDim_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);
                 *          unsigned int* smem_array_child = __memPoolHost.grid_allocate<unsigned int>(__gDim*__bDim);
                 *          cudaMemset(gDim_array_child, 0, 2*__gDim*__bDim*sizeof(unsigned int));
                 *
                 *          // The kernel is called
                 *          kernel_name_kernel_k <<< __gDim, __bDim, __smem >>> (p1_array, p2_array, __gDim_array, __bDim_array, __parentSize, __memPoolDevice);
                 *
                 *          // The following code is generated for each child kernel of kernel_name, where the child kernel signature is:
                 *          //         __global__ void child_name(S1 q1, S2 q2)
                 *          unsigned int __gDim_child = kernel_inclusive_scan(gDim_array_child, __gDim*__bDim);
                 *          unsigned int __bDim_child = kernel_max(bDim_array_child, __gDim*__bDim);
                 *          unsigned int __smem_child = kernel_max(smem_array_child, __gDim*__bDim);
                 *          child_name_kernel_k_h(__gDim_child, __bDim_child, __smem_child, q1_array_child, q2_array_child, gDim_array_child, bDim_array_child, __gDim*__bDim, __memPoolHost);
                 *
                 *      }
                 *  }
                 */

                // Find all the kernel calls in the kernel
                KernelCallFinder kcFinder(f);
                std::set<CUDAKernelCallExpr*> childKernelCalls = kcFinder.getKernelCalls();

                // Declare and initialize commonly used variables
                std::map<FunctionDecl*, std::vector<bool>> isScalarChildParam;
                std::map<FunctionDecl*, bool> isScalarChildBlockDim;
                std::map<FunctionDecl*, bool> isChildSmemConfigExplicit;
                std::map<FunctionDecl*, bool> isScalarChildSmemConfig;
                for(CUDAKernelCallExpr* childKernelCall : childKernelCalls) {
                    FunctionDecl* childKernel = childKernelCall->getDirectCallee();
                    isScalarChildParam[childKernel].resize(childKernel->getNumParams());
                    for(unsigned int p = 0; p < childKernel->getNumParams(); ++p) {
                        isScalarChildParam[childKernel][p] = CompilerOptions::scalarizeInvariantParameters() && analyzer_.isParamInvariant(childKernel, p);
                    }
                    isScalarChildBlockDim[childKernel] = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(childKernel);
                    isChildSmemConfigExplicit[childKernel] = !dyn_cast<CXXDefaultArgExpr>(childKernelCall->getConfig()->getArg(2));
                    isScalarChildSmemConfig[childKernel] = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isInvariant(childKernelCall->getConfig()->getArg(2));
                }

                // Launchpad function signature
                ssEpilogue << "__device__ void " << f->getNameAsString() << "_kernel_k_h(unsigned int __gDim, unsigned int __bDim, ";
                if(analyzer_.isCalledFromKernelWithSmem(f)) {
                    ssEpilogue << "unsigned int __smem, ";
                }
                for(unsigned int p = 0; p < f->getNumParams(); ++p) {
                    ParmVarDecl* param = f->getParamDecl(p);
                    if (isScalarParam[p]) {
                        ssEpilogue << param->getType().getAsString() << " " << param->getNameAsString() << ", ";
                    } else {
                        ssEpilogue << param->getType().getAsString() << "* " << param->getNameAsString() << ", ";
                    }

                }

                // Add numBlocks and numThreads parameter arrays
                ssEpilogue << "unsigned int* __gDim_array, ";
//                if(!isScalarBlockDim) {
//                    ssEpilogue << "unsigned int* __bDim_array, ";
//                }

                // Add size of the parent
                ssEpilogue << "unsigned int __parentSize, ";

                // Add memory pool
                ssEpilogue << "__GridMemPool __memPoolHost) {\n";

                // Check for NB = 0
                ssEpilogue << "if (__gDim > 0) {\n";

                // Hold the memory pool to be used in the launch
                ssEpilogue << "__GridMemPool __memPoolDevice = __memPoolHost;\n";

                ssEpilogue << "unsigned int __granularity = __GRANULARITY > gridDim.x ? gridDim.x : __GRANULARITY;\n"
                      "int __bucketIdx = blockIdx.x / __granularity;\n"
                      "unsigned int __granularitySize = blockDim.x * __granularity;\n"
                      "unsigned int __offset = __bucketIdx * __granularitySize * __granularity;\n"
                      "unsigned int __totalNumOfBuckets = (gridDim.x + __granularity - 1) / __granularity;";

                // Extract and initialize parameter/configuration arrays of all children
                for(CUDAKernelCallExpr* childKernelCall : childKernelCalls) {

                    FunctionDecl* childKernel = childKernelCall->getDirectCallee();

                    // Extract pointers to the arrays of each parameter
                    for(unsigned int p = 0; p < childKernel->getNumParams(); ++p) {
                        ParmVarDecl* param = childKernel->getParamDecl(p);
                        std::string paramType = param->getType().getAsString();
                        std::string paramName = param->getNameAsString();
                        if(isScalarChildParam[childKernel][p]) {
                            continue;
                        }

                        ssEpilogue << paramType << "* " << paramName << "_array_child = __memPoolHost.grid_allocate<" << paramType << ">(";
                        ssEpilogue << "blockDim.x * __granularity";
                        ssEpilogue << ") + __offset;\n";
                    }

                    // Extract pointers to the arrays of the configurations and initialize to 0
                    ssEpilogue << "unsigned int* gDim_array = __memPoolHost.grid_allocate<unsigned int>(blockDim.x * __granularity);\n";
                    std::string memsetLength = "(__gDim*__bDim";
                    ssEpilogue << "unsigned int* bDim_array_child = __memPoolHost.grid_allocate<unsigned int>(";
                    if(isScalarChildBlockDim[childKernel]) {
                        ssEpilogue << "2";
                        memsetLength += " + 2";
                    } else {
                        ssEpilogue << "__granularity*__bDim";
                        memsetLength += "+ __granularity*__bDim";
                    }
                    ssEpilogue << ");\n";
                    if(isChildSmemConfigExplicit[childKernel]) {
                        ssEpilogue << "unsigned int* smem_array_child = __memPoolHost.grid_allocate<unsigned int>(";
                        if(isScalarChildSmemConfig[childKernel]) {
                            ssEpilogue << "2";
                            memsetLength += " + 2";
                        } else {
                            ssEpilogue << "__granularity*__bDim";
                            memsetLength += " + __granularity*__bDim";
                        }
                        ssEpilogue << ");\n";
                    }
                    memsetLength += ")*sizeof(unsigned int)";
                    if(CompilerOptions::useAtomicsBasedScan()) {
                        ssEpilogue << "union scan_counter* sc = __memPoolHost.grid_allocate<union scan_counter>(1);\n";
                        memsetLength += " + sizeof(union scan_counter)";
                    }

                }

                // Kernel call
                ssEpilogue << f->getNameAsString() << "_kernel_k <<< __gDim, __bDim";
                if(analyzer_.isCalledFromKernelWithSmem(f)) {
                    ssEpilogue << ", __smem";
                }
                ssEpilogue << " >>> (";
                for(unsigned int p = 0; p < f->getNumParams(); ++p) {
                    ParmVarDecl* param = f->getParamDecl(p);
                    ssEpilogue << param->getNameAsString() << ", ";
                }
                ssEpilogue << "__gDim_array, ";
//                if(!isScalarBlockDim) {
//                    ssEpilogue << "__bDim_array, ";
//                }
                ssEpilogue << "__parentSize, __memPoolDevice);\n\n";

                // Close if
                ssEpilogue << "}\n";

                SourceLocation endBody = cast<CompoundStmt>(f->getBody())->body_back()->getEndLoc();
                rewriter_.InsertTextAfterToken(endBody, ssEpilogue.str());

            }

        } else {

            /*
             * Transform the signature of the original kernel:
             *
             *      __global__ void kernel_name(T1 p1, T2 p2)
             *
             * into the signature of the transformed kernel called from the host:
             *
             *      __global__ void kernel_name_kernel_h(T1 p1, T2 p2, __GridMemPool __memPool)
             *
             */

            // Append _kernel_h to kernel name
            SourceLocation nameLoc = f->getNameInfo().getLoc();
            rewriter_.InsertTextAfterToken(nameLoc, "_kernel_h");

            // Create mem pool and index parameters
            std::stringstream ssParams;
            ssParams << ", __GridMemPool __memPool";

            // Add new parameters to kernel declaration
            SourceLocation lastParamLoc =
                f->getParamDecl(f->getNumParams() - 1)->getSourceRange().getEnd();
            rewriter_.InsertTextAfterToken(lastParamLoc, ssParams.str());
        }
    } else if(!f->isImplicit()) { // Not a CUDA kernel
        SourceManager& sm = rewriter_.getSourceMgr();
        if(sm.isInMainFile(sm.getExpansionLoc(f->getBeginLoc()))) {
            rewriter_.RemoveText(SourceRange(f->getBeginLoc(), f->getEndLoc()));
        }
    }

    return true;

}

bool KernelDAGTransform::VisitStmt(Stmt *s) {

    if(CUDAKernelCallExpr* kernelCall = dyn_cast<CUDAKernelCallExpr>(s)) {

        if(analyzer_.isCallFromKernelCandidate(kernelCall)) {

            /*
             * Transform the call to the original kernel:
             *
             *     kernel_name <<< gDim, bDim >>> (p1, p2);
             *
             * into an aggregated call to the transformed kernel:
             *
             *      T1* p1_array = __memPool.grid_allocate<T1>(blockDim.x*gridDim.x);
             *      T2* p2_array = __memPool.grid_allocate<T2>(blockDim.x*gridDim.x);
             *      unsigned int* __gDim_array = __memPool.grid_allocate<unsigned int>(blockDim.x*gridDim.x);
             *      unsigned int* __bDim_array = __memPool.grid_allocate<unsigned int>(blockDim.x*gridDim.x);
             *      unsigned int* __smem_array = __memPool.grid_allocate<unsigned int>(blockDim.x*gridDim.x);
             *
             *      unsigned int _i_ = blockDim.x*blockIdx.x + threadIdx.x;
             *      p1_array[_i_] = p1;
             *      p2_array[_i_] = p2;
             *      __gDim_array[_i_] = gDim;
             *      __bDim_array[_i_] = bDim;
             *      __smem_array[_i_] = smem;
             *
             */

            // For blockDim.x and gridDim.x, always use ‘.’, never ‘_’

            // Declare and initialize commonly used variables
            std::stringstream ss;
            FunctionDecl* kernel = kernelCall->getDirectCallee();
            CallExpr* config = kernelCall->getConfig();
            std::string gridDimConfig = toString(config->getArg(0));
            std::string blockDimConfig = toString(config->getArg(1));
            bool isScalarBlockDim = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isBlockDimInvariant(kernel);
            std::string smemConfig = toString(config->getArg(2));
            bool isSmemConfigExplicit = !dyn_cast<CXXDefaultArgExpr>(config->getArg(2));
            bool isScalarSmemConfig = CompilerOptions::scalarizeInvariantConfigurations() && analyzer_.isInvariant(config->getArg(2));
            std::vector<bool> isScalarParam(kernel->getNumParams());
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                isScalarParam[p] = CompilerOptions::scalarizeInvariantParameters() && analyzer_.isParamInvariant(kernel, p);
            }

            // Create a new scope to avoid name collisions
            ss << "{\n";

            std::stringstream granularitySizeSS;
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                ParmVarDecl* param = kernel->getParamDecl(p);
                std::string paramType = param->getType().getAsString();
                std::string paramName = param->getNameAsString();

                if(isScalarParam[p]) {
                    continue;
                }

                granularitySizeSS << "sizeof(" << paramType << ") + ";
            }

            ss << "unsigned int __granularity = __GRANULARITY > gridDim.x ? gridDim.x : __GRANULARITY;\n";

            ss << "int __bucketIdx = blockIdx.x / __granularity;\n";
            ss << "unsigned int __granularitySize = blockDim.x * __granularity * (" << granularitySizeSS.str() << " sizeof(unsigned int));\n";
            ss << "unsigned int __offset = __bucketIdx * __granularitySize;\n"
                  "unsigned int __totalNumOfBuckets = (gridDim.x + __granularity - 1) / __granularity;";

            ss << "union scan_counter *_sc_ = __memPool.grid_allocate<union scan_counter>(__totalNumOfBuckets);\n";
            ss << "unsigned int *__aggregationGranularityCounterBuckets = __memPool.grid_allocate<unsigned int>(__totalNumOfBuckets);\n";
            // Allocate memory in memory pool
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                ParmVarDecl* param = kernel->getParamDecl(p);
                std::string paramType = param->getType().getAsString();
                std::string paramName = param->getNameAsString();

                if(isScalarParam[p]) {
                    continue;
                }
                ss << paramType << "* " << paramName << "_array = __memPool.grid_allocate<" << paramType << ">(";
                ss << "blockDim.x * __granularity";
                ss << ") + __offset;\n";
            }
            ss << "unsigned int* __gDim_array = __memPool.grid_allocate<unsigned int>(blockDim.x * __granularity) + __offset;\n";
            if(isSmemConfigExplicit) {
                ss << "unsigned int* __smem_array = __memPool.grid_allocate<unsigned int>(";
                if(isScalarSmemConfig) {
                    ss << "2"; // FIXME: Only 1 needed; using 2 to avoid alignment issues; find a cleaner solution
                } else {
                    ss << "blockDim.x * __granularity";
                }
                ss << ");\n";
            }

            // Collect parameters and configurations
            if(CompilerOptions::useAtomicsBasedScan()) {

                // Identify index to store parameters
                CallExpr* config = kernelCall->getConfig();
                std::string gridDimConfig = toString(config->getArg(0));
                ss << "unsigned int __gDim = " << gridDimConfig << ";\n";
                ss << "if(__gDim > 0) {\n"; // Only parent threads with non-zero child blocks participate
                ss << "union scan_counter sc_local;\n";
                ss << "sc_local.idx = 1;\n";
                ss << "sc_local.nb = __gDim;\n";
                ss << "sc_local.fused = atomicAdd(&(_sc_[__bucketIdx].fused), sc_local.fused);\n";
                ss << "unsigned int _i_ = sc_local.idx;\n";

                // Store non-invariant kernel parameters in arrays
                unsigned int numArgs = kernelCall->getNumArgs();
                for(unsigned int a = 0; a < numArgs; ++a) {
                    if(!isScalarParam[a]) {
                        std::string arg = toString(kernelCall->getArg(a));
                        std::string paramName = kernel->getParamDecl(a)->getNameAsString();
                        ss << paramName << "_array[_i_] = " << arg << ";\n";
                    }
                }

                // Store # blocks configurations
                ss << "__gDim_array[_i_] = sc_local.nb + __gDim;\n";

                // Store # threads configurations if non-invariant

                // Store smem configuration if non-invariant
                if(isSmemConfigExplicit) {
                    if(!isScalarSmemConfig) {
                        ss << "__smem_array[_i_] = " << smemConfig << ";\n";
                    }
                }

                // All threads are active again
                ss << "} // __gDim > 0\n";
            }
            ss << "__threadfence();\n"
                  "__syncthreads();\n";

            ss << "if (threadIdx.x == 0) {";
            ss << "int __totalInCurrentBucket = atomicAdd(&__aggregationGranularityCounterBuckets[__bucketIdx], 1) + 1;";
            ss << "if (__totalInCurrentBucket == __granularity || (__bucketIdx == (__totalNumOfBuckets - 1) && __totalInCurrentBucket == (gridDim.x % __granularity))) {";


            ss << kernel->getNameAsString() << "_kernel_k_h(_sc_[__bucketIdx].nb, " << blockDimConfig << ", ";
            for(unsigned int p = 0; p < kernel->getNumParams(); ++p) {
                std::string paramName = kernel->getParamDecl(p)->getNameAsString();
                ss << paramName;
                if (!isScalarParam[p]) {
                    ss << "_array";
                }
                ss << ", ";
            }
            ss << "__gDim_array, ";
            if(CompilerOptions::useAtomicsBasedScan()) {
                ss << "_sc_[__bucketIdx].idx";
            }
            ss << ", __memPool);\n"; // FIXME: Hold mem_pool for each child to pass the right mem_pool

            ss << "}\n}";
            // Close scope
            ss << "}\n";

            // Replace original call
            rewriter_.ReplaceText(SourceRange(kernelCall->getBeginLoc(), kernelCall->getEndLoc()), ss.str());

        }

    } else if(CallExpr* CE = dyn_cast<CallExpr>(s)) {

        if(analyzer_.isAPIFromKernelCandidate(CE)) {
            if(FunctionDecl* f = CE->getDirectCallee()) {
                std::string funcName = f->getDeclName().getAsString();
                std::stringstream ss;
                ss << "__" << funcName << "_kernel(";
                for(unsigned int a = 0; a < CE->getNumArgs(); ++a) {
                    std::string arg = toString(CE->getArg(a));
                    if(a > 0) ss << ", ";
                    ss << arg;
                }
                ss << ")";
                rewriter_.ReplaceText(SourceRange(CE->getBeginLoc(), CE->getEndLoc()), ss.str());
            }
        }

    } else if(MemberExpr* memberExpr = dyn_cast<MemberExpr>(s)) {
        if(isCalledFromKernel_) {

            // Replace uses of gridDim.c, blockDim.x, and blockIdx.x with __gridDim_x, __blockDim_x, and __blockIdx_x
            SourceLocation beginExpr = memberExpr->getBeginLoc();
            SourceLocation endExpr = memberExpr->getEndLoc();
            if(toString(memberExpr) == "gridDim.__fetch_builtin_x") { // gridDim.x
                rewriter_.ReplaceText(SourceRange(beginExpr, endExpr), "__gridDim_x");
            } else if(toString(memberExpr) == "blockDim.__fetch_builtin_x") { // blockDim.x
                rewriter_.ReplaceText(SourceRange(beginExpr, endExpr), "__blockDim_x");
            } else if(toString(memberExpr) == "blockIdx.__fetch_builtin_x") { // blockIdx.x
                rewriter_.ReplaceText(SourceRange(beginExpr, endExpr), "__blockIdx_x");
            }
        }
    }

    return true;
}
