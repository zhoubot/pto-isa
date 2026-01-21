Skip to content
Chat history




You said:
//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/PTO/IR/PTO.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace pto;
using namespace mlir::bufferization;

namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult bufferizeDestinationStyleOpInterface(
    RewriterBase &rewriter, DestinationStyleOpInterface op,
    const BufferizationOptions &options,
    bool supportMixedTensorBufferMode = true) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics()) {
    return success();
  }

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics() && !supportMixedTensorBufferMode) {
    return op->emitError() << "op does not have tensor semantics";
  }

  // New operands for the cloned op.
  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (!isa<TensorType>(opOperand.get().getType())) {
      newOperands.push_back(opOperand.get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand.get(), options);
    if (failed(buffer)) {
      return failure();
    }
    newOperands.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer)) {
      return failure();
    }
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands.
  clone(rewriter, op, /*newResultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

// struct MmadL1OpInterface
//     : public DstBufferizableOpInterfaceExternalModel<MmadL1OpInterface,
//                                                      pto::MmadL1Op> {
//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options);
//   }
// };

// struct FixpipeOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<FixpipeOpInterface,
//                                                      pto::FixpipeOp> {
//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     if (dpsOp.hasPureBufferSemantics()) {
//       return success();
//     }
//     if (dpsOp.hasPureTensorSemantics()) {
//       return bufferizeDestinationStyleOpInterface(rewriter, dpsOp, options);
//     }
//     // We only handle the case where fixpipe op's input is a tensor from
//     // mmad and fixpipe op's output is a memref type.
//     auto srcOp = dpsOp.getDpsInputOperand(0);
//     auto dstOp = dpsOp.getDpsInitOperand(0);
//     if (!isa<TensorType>(srcOp->get().getType()) ||
//         !isa<MemRefType>(dstOp->get().getType())) {
//       return op->emitError() << "src and dst op should have tensor and memref "
//                                 "type, respectively";
//     }
//     // Take a guard before anything else.
//     OpBuilder::InsertionGuard g(rewriter);
//     rewriter.setInsertionPoint(op);

//     FailureOr<Value> buffer = getBuffer(rewriter, srcOp->get(), options);
//     if (failed(buffer)) {
//       return failure();
//     }
//     // Set insertion point now that potential alloc/dealloc are introduced.
//     rewriter.setInsertionPoint(op);
//     // Clone the op, but use the new operands.
//     auto newOp = cast<DestinationStyleOpInterface>(clone(
//         rewriter, op, /*newResultTypes=*/TypeRange{}, {*buffer, dstOp->get()}));
//     // We need to manually replace the old op because it has memory effects
//     // and won't be deleted automatically.
//     rewriter.replaceOp(op, newOp);
//     return success();
//   }

//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInput(&opOperand);
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInit(&opOperand);
//   }
// };

// template <typename OpType>
// struct NDNZConversionOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<
//           NDNZConversionOpInterface<OpType>, OpType> {
//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options);
//   }
// };

// struct PTOCopyOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<PTOCopyOpInterface,
//                                                      pto::CopyOp> {
//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options);
//   }
// };

struct PTOLoadOpInterface
    : public DstBufferizableOpInterfaceExternalModel<PTOLoadOpInterface,
                                                     pto::LoadDpsOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct PTOStoreOpInterface
    : public DstBufferizableOpInterfaceExternalModel<PTOStoreOpInterface,
                                                     pto::StoreDpsOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    if (dpsOp.hasPureBufferSemantics()) {
      return success();
    }
    if (dpsOp.hasPureTensorSemantics()) {
      return bufferizeDestinationStyleOpInterface(rewriter, dpsOp, options);
    }
    // We only handle the case where fixpipe op's input is a tensor from
    // mmad and fixpipe op's output is a memref type.
    auto srcOp = dpsOp.getDpsInputOperand(0);
    auto dstOp = dpsOp.getDpsInitOperand(0);
    if (!isa<TensorType>(srcOp->get().getType()) ||
        !isa<MemRefType>(dstOp->get().getType())) {
      return op->emitError() << "src and dst op should have tensor and memref "
                                "type, respectively";
    }
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    FailureOr<Value> buffer = getBuffer(rewriter, srcOp->get(), options);
    if (failed(buffer)) {
      return failure();
    }
    // Set insertion point now that potential alloc/dealloc are introduced.
    rewriter.setInsertionPoint(op);
    // Clone the op, but use the new operands.
    auto newOp = cast<DestinationStyleOpInterface>(clone(
        rewriter, op, /*newResultTypes=*/TypeRange{}, {*buffer, dstOp->get()}));
    // We need to manually replace the old op because it has memory effects
    // and won't be deleted automatically.
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct PTOAddFOpInterface
    : public DstBufferizableOpInterfaceExternalModel<PTOAddFOpInterface,
                                                     pto::AddFDpsOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it is not an input/init.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    // Src0 and dst of elemwiseOp are not conflicting if the op bufferizes
    // to element-wise access.
    // auto ptoOp = dyn_cast<PTOStructuredOp>(op);
    // return ptoOp && ptoOp.isElemwiseNaryOp();
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct PTOMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<PTOMatmulOpInterface,
                                                     pto::MatmulDpsOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

// struct PTOMixMatmulOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<PTOMixMatmulOpInterface,
//                                                      pto::MixMatmulOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInput(&opOperand);
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInit(&opOperand);
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     // The tilingParams operand might be already bufferized.
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options,
//         /*supportMixedTensorBufferMode=*/true);
//   }
// };

// struct PTOMixGroupMatmulOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<
//           PTOMixGroupMatmulOpInterface, pto::MixGroupMatmulOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInput(&opOperand);
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInit(&opOperand);
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     // The tilingParams operand might be already bufferized.
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options,
//         /*supportMixedTensorBufferMode=*/true);
//   }
// };

// template <typename OpTy>
// struct VectorOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<VectorOpInterface<OpTy>,
//                                                      OpTy> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     // Operand is read if it is used in the computation.
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInput(&opOperand);
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     // Operand is written to if it is not an input/init.
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInit(&opOperand);
//   }

//   bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
//                                      ArrayRef<OpOperand *> opOperands) const {
//     // Src0 and dst of elemwiseOp are not conflicting if the op bufferizes
//     // to element-wise access.
//     auto ptoOp = dyn_cast<PTOStructuredOp>(op);
//     return ptoOp && ptoOp.isElemwiseNaryOp();
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options);
//   }
// };

// struct PrintOpInterface
//     : public BufferizableOpInterface::ExternalModel<PrintOpInterface,
//                                                     pto::PrintOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     return true;
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     return false;
//   }

//   AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
//                                       const AnalysisState &state) const {
//     return {};
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     auto printOp = cast<pto::PrintOp>(op);

//     auto prefix = printOp.getPrefix();
//     auto hex = printOp.getHex();

//     Value newArg;
//     const auto &arg = printOp.getArg();
//     Value value = arg;
//     if (isa<TensorType>(value.getType())) {
//       FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);
//       if (failed(maybeBuffer))
//         return failure();
//       Value buffer = *maybeBuffer;
//       newArg = buffer;
//     } else {
//       newArg = value;
//     }

//     replaceOpWithNewBufferizedOp<pto::PrintOp>(
//         rewriter, op, prefix, hex, newArg,
//         pto::TCoreTypeAttr::get(op->getContext(),
//                                  pto::TCoreType::CUBE_OR_VECTOR));

//     return success();
//   }
// };

// struct VPadOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<VPadOpInterface,
//                                                      pto::VPadOp> {
//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     // TODO
//     return failure();
//   }
// };

// struct VConcatOpInterface
//     : public DstBufferizableOpInterfaceExternalModel<VConcatOpInterface,
//                                                      pto::VConcatOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInput(&opOperand);
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     auto dpsOp = cast<DestinationStyleOpInterface>(op);
//     return dpsOp.isDpsInit(&opOperand);
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     return bufferizeDestinationStyleOpInterface(
//         rewriter, cast<DestinationStyleOpInterface>(op), options,
//         /*supportMixedTensorBufferMode=*/true);
//   }
// };

/// Helper structure that iterates over all VectorOps in OpTys and registers
/// the BufferizableOpInterface with each of them.
// template <typename... Ops> struct VectorOpInterfaceHelper {
//   static void registerOpInterface(MLIRContext *ctx) {
//     (Ops::template attachInterface<VectorOpInterface<Ops>>(*ctx), ...);
//   }
// };

// struct BitcastOpInterface
//     : public BufferizableOpInterface::ExternalModel<BitcastOpInterface,
//                                                     pto::BitcastOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     return false;
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     return false;
//   }

//   AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
//                                       const AnalysisState &state) const {
//     return {{op->getResult(0), BufferRelation::Equivalent}};
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     auto bitcastOp = dyn_cast<pto::BitcastOp>(op);
//     auto resultTensorType = dyn_cast<TensorType>(bitcastOp.getType());
//     if (!resultTensorType)
//       return success();

//     FailureOr<Value> source = getBuffer(rewriter, bitcastOp.getSrc(), options);
//     if (failed(source))
//       return failure();
//     auto sourceType = cast<BaseMemRefType>(source->getType());

//     // Result type should have same layout and address space as the source type.
//     BaseMemRefType resultType;
//     if (auto rankedMemRefType = dyn_cast<MemRefType>(sourceType)) {
//       resultType = MemRefType::get(
//           rankedMemRefType.getShape(), resultTensorType.getElementType(),
//           rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());
//     } else {
//       auto unrankedMemrefType = cast<UnrankedMemRefType>(sourceType);
//       resultType = UnrankedMemRefType::get(resultTensorType.getElementType(),
//                                            unrankedMemrefType.getMemorySpace());
//     }

//     replaceOpWithNewBufferizedOp<pto::BitcastOp>(rewriter, op, resultType,
//                                                   *source);
//     return success();
//   }
// };

// struct EmbeddingGatherOpInterface
//     : public BufferizableOpInterface::ExternalModel<EmbeddingGatherOpInterface,
//                                                     pto::EmbeddingGatherOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     return opOperand.getOperandNumber() < 2;
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     return opOperand.getOperandNumber() == 2; // $dst
//   }

//   AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
//                                      const AnalysisState &state) const {
//     auto gatherOp = cast<EmbeddingGatherOp>(op);
//     AliasingValueList result;

//     if (opOperand.getOperandNumber() == 2) { // $dst
//       // dst is alias of the result
//       result.addAlias({AliasingValue(gatherOp->getResult(0),
//                                     BufferRelation::Equivalent,
//                                     /*isMustAlias=*/true)});
//     }

//     return result;
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     auto gatherOp = cast<pto::EmbeddingGatherOp>(op);

//     auto srcBuffer = gatherOp.getSrc();

//     FailureOr<Value> indexBuffer = getBuffer(rewriter, gatherOp.getIndex(), options);
//     if (failed(indexBuffer))
//       return failure();

//     FailureOr<Value> dstBuffer = getBuffer(rewriter, gatherOp.getDst(), options);
//     if (failed(dstBuffer))
//       return failure();

//     Value bound = gatherOp.getBound();
//     auto offsets = gatherOp.getOffsets();
//     auto numels = gatherOp.getNumels();

//     auto bufferizedOp = rewriter.create<EmbeddingGatherOp>(
//         gatherOp.getLoc(),
//         /*resultType*/ TypeRange{},
//         /*operands*/
//         srcBuffer, *indexBuffer, *dstBuffer, bound, offsets, numels);

//     if (gatherOp->getNumResults() > 0) {
//       replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
//     }

//     return success();
//   }
// };

// struct IndirectLoadOpInterface
//     : public BufferizableOpInterface::ExternalModel<IndirectLoadOpInterface,
//                                                     pto::IndirectLoadOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const AnalysisState &state) const {
//     return opOperand.getOperandNumber() < 2 || opOperand.getOperandNumber() ==3 || opOperand.getOperandNumber() == 4;
//   }
 
//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const AnalysisState &state) const {
//     return opOperand.getOperandNumber() == 2; // $dst
//   }
 
//   AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
//                                      const AnalysisState &state) const {
//     auto indirectLoadOp = cast<IndirectLoadOp>(op);
//     AliasingValueList result;
 
//     if (opOperand.getOperandNumber() == 2) { // $dst
//       result.addAlias({AliasingValue(indirectLoadOp->getResult(0),
//                                     BufferRelation::Equivalent,
//                                     /*isMustAlias=*/true)});
//     }
 
//     return result;
//   }
 
//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const BufferizationOptions &options) const {
//     auto indirectLoadOp = cast<pto::IndirectLoadOp>(op);
 
//     auto srcBuffer = indirectLoadOp.getSrc();
 
//     FailureOr<Value> offsetBuffer = getBuffer(rewriter, indirectLoadOp.getOffsets(), options);
//     if (failed(offsetBuffer))
//       return failure();
 
//     FailureOr<Value> dstBuffer = getBuffer(rewriter, indirectLoadOp.getDst(), options);
//     if (failed(dstBuffer))
//       return failure();
    
//     FailureOr<Value> maskBuffer = failure();
//     if (indirectLoadOp.getMask()) {
//       maskBuffer = getBuffer(rewriter, indirectLoadOp.getMask(), options);
//       if (failed(dstBuffer))
//         return failure();
//     }

//     FailureOr<Value> otherBuffer = failure();
//     if (indirectLoadOp.getOther()) {
//       otherBuffer = getBuffer(rewriter, indirectLoadOp.getOther(), options);
//       if (failed(dstBuffer))
//         return failure();
//     }

//     auto mask = indirectLoadOp.getMask();
//     auto other = indirectLoadOp.getOther();
 
//     if (indirectLoadOp.getMask()) {
//       if (indirectLoadOp.getOther()) {
//         rewriter.create<IndirectLoadOp>(
//         indirectLoadOp.getLoc(),
//         /*resultType*/ TypeRange{},
//         /*operands*/
//         srcBuffer, *offsetBuffer, *dstBuffer, *maskBuffer, *otherBuffer);
//       }else {
//         rewriter.create<IndirectLoadOp>(
//         indirectLoadOp.getLoc(),
//         /*resultType*/ TypeRange{},
//         /*operands*/
//         srcBuffer, *offsetBuffer, *dstBuffer, *maskBuffer, other);
//       }
//     } else {
//       rewriter.create<IndirectLoadOp>(
//         indirectLoadOp.getLoc(),
//         /*resultType*/ TypeRange{},
//         /*operands*/
//         srcBuffer, *offsetBuffer, *dstBuffer, mask, other);
//     }
 
//     if (indirectLoadOp->getNumResults() > 0) {
//       replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
//     }
 
//     return success();
//   }
// };

} // namespace

void mlir::pto::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, pto::PTODialect *dialect) {
    // FixpipeOp::attachInterface<FixpipeOpInterface>(*ctx);
    // MmadL1Op::attachInterface<MmadL1OpInterface>(*ctx);
    // ND2NZOp::attachInterface<NDNZConversionOpInterface<ND2NZOp>>(*ctx);
    // NZ2NDOp::attachInterface<NDNZConversionOpInterface<NZ2NDOp>>(*ctx);
    // CopyOp::attachInterface<PTOCopyOpInterface>(*ctx);
    LoadDpsOp::attachInterface<PTOLoadOpInterface>(*ctx);
    StoreDpsOp::attachInterface<PTOStoreOpInterface>(*ctx);
    AddFDpsOp::attachInterface<PTOAddFOpInterface>(*ctx);
    MatmulDpsOp::attachInterface<PTOMatmulOpInterface>(*ctx);
    // MixMatmulOp::attachInterface<PTOMixMatmulOpInterface>(*ctx);
    // MixGroupMatmulOp::attachInterface<PTOMixGroupMatmulOpInterface>(*ctx);
    // PrintOp::attachInterface<PrintOpInterface>(*ctx);
    // VConcatOp::attachInterface<VConcatOpInterface>(*ctx);
    // BitcastOp::attachInterface<BitcastOpInterface>(*ctx);
    // EmbeddingGatherOp::attachInterface<EmbeddingGatherOpInterface>(*ctx);
    // IndirectLoadOp::attachInterface<IndirectLoadOpInterface>(*ctx);
    // Register all PTO Vector Ops
//     VectorOpInterfaceHelper<
// #define GET_OP_LIST
// #include "bishengir/Dialect/PTO/IR/PTOVectorOps.cpp.inc"
//         >::registerOpInterface(ctx);
  });
}
ChatGPT said:
我先按“这份实现现在能不能稳、有哪些明显坑、怎么改更像 MLIR upstream 的写法”给你做一次 code review（重点放在 PTOStoreOpInterface / 通用 helper / memory effects / mixed tensor-buffer 这几块）。
1) bufferizeDestinationStyleOpInterface 这个通用 helper：整体 OK，但有两个隐患
隐患 A：newOutputBuffers 只按 result 找 init operand，默认 result 与 init 一一对应
你这里：
for (OpResult opResult : op->getOpResults()) {
  OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
  ...
}
这要求 每个 result 都对应一个 dps init，而且 result index == init index（DestinationStyleOpInterface 的常见约定是这样，但不是所有 DPS op 都必然满足，尤其是：
op 没有 result（纯写入）
op 有多个 result，但只部分是 dps init 的别名/等价
op 的 result 顺序与 init operand 顺序不一致（少见，但在自定义 op 中可能出现）
如果 StoreDpsOp 是 “0 results” 的纯写入，这段 loop 就没问题（空），但后面 replaceOpWithBufferizedValues 会用 newOutputBuffers 替换 results —— 也是空，OK。
建议：把这段写成更 defensive 的方式：只遍历 dpsInits()，并用 getDpsInitOperand(i)/getOpResults() 的关系来决定替换；或者加 assert(op.getNumDpsInits() == op->getNumResults())（如果你的 PTO DPS ops 就是这么定义的）。

隐患 B：clone 后没有显式处理 “result aliases init” 的语义
DPS bufferization 通常依赖 DstBufferizableOpInterfaceExternalModel 来推导别名关系，但你这里是直接：
clone（结果类型置空）
replace old results with newOutputBuffers
这对“结果就是 init 的等价别名”是对的，但如果某些 op 语义是 “结果是 init 的 subview / reinterpret_cast / copy”，那么这就不对了。
建议：如果你确认 PTO 的 DPS ops 都是 “结果等价于 init（in-place）”，就 OK；否则要给具体 op 做自定义 bufferize（比如 bitcast、subview 类）。

2) PTOStoreOpInterface：注释和逻辑不一致 + 缺了 read/write 描述
你现在的 store bufferize 逻辑是从以前 Fixpipe 的特殊 case 改出来的，但注释还在说 fixpipe/mmad：
// We only handle the case where fixpipe op's input is a tensor from
// mmad and fixpipe op's output is a memref type.
这会误导以后维护的人。
更关键的问题：你没有 override bufferizesToMemoryRead/Write
对 DstBufferizableOpInterfaceExternalModel 来说，默认实现通常能工作，但 Store 这种 op 往往需要明确：
input 是 read（从 tensor/buffer 读）
init/dst 是 write（写入 memref/buffer）
你在 PTOAddFOpInterface、PTOMatmulOpInterface 都写了，但 PTOStoreOpInterface 没写。
**建议改法：**给 Store 加上显式读写（这对冲突分析、placement、eliminate copies 都更稳）。

bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const override {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  return dpsOp.isDpsInput(&opOperand);
}

bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                             const AnalysisState &state) const override {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  return dpsOp.isDpsInit(&opOperand);
}
你的 mixed-mode 特判（tensor -> memref）是合理的，但写法可以更“MLIR 风格”
你现在是：
如果纯 tensor：走通用 helper
如果 mixed：只允许 input tensor + init memref，然后把 input tensor bufferize 成 memref，再 clone
这个思路对 store 很常见。建议把错误信息和判断写得更精确，比如：
“expected operand #0 to be tensor and init #0 to be memref in mixed mode”
或直接打印类型
3) PTOAddFOpInterface::bufferizesToElementwiseAccess 现在总是 true，要小心
你这里直接返回 true：
bool bufferizesToElementwiseAccess(...){ return true; }
这会告诉 bufferization 分析：“这些 operands 之间不会发生跨 element 的冲突”，从而放宽某些依赖/拷贝要求。
如果你的 AddFDpsOp 真的是严格 elementwise，并且 indexing maps 是 identity / broadcasting 也能证明 elementwise，那 OK。

但如果 AddF 支持：

broadcast
transpose-like indexing
或更复杂的 access pattern
那直接 true 会让分析 低估冲突风险。
建议：

要么像你注释里那样：dyn_cast<PTOStructuredOp>(op) 并用 isElemwiseNaryOp() 判断
要么保守返回 false（宁可多一些拷贝，也不要错）
4) PTOMatmulOpInterface 允许 mixed-mode：可以，但要确保 “tilingParams / attrs” 之类非 tensor operand 不会误处理
你通用 helper 里对非 tensor operand 是直接透传：
if (!isa<TensorType>(opOperand.get().getType())) {
  newOperands.push_back(opOperand.get());
  continue;
}
所以像 index, i32, attr（attrs 不在 operands）都安全。
但如果 MatmulDpsOp 有某些 operand 是 memref（比如 tiling params buffer），你设置 supportMixedTensorBufferMode=true，那这个 helper 会把 tensor 变 memref、memref 保持 memref ——

You said:
//===- AllocToPointerCast.h --Convert memref.AllocOp to pto.pointercastOp-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#ifndef LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#define LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#include "mlir/Dialect/PTO/IR/PTO.h"
#include "mlir/Dialect/PTO/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace pto {
class MemrefAllocaOpToPointerCastOpPattern
    : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  /// map from buffer to its allocated addresses
  /// note: the buffer which does multibuffer n optimization will be allocated n
  /// addresses.
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

  explicit MemrefAllocaOpToPointerCastOpPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<memref::AllocOp>(context),
        buffer2Offsets(buffer2Offsets) {}
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final;
};

// class UpdateWorkSpaceAllocaOpOffsetPattern
//     : public OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp> {
// public:
//   using OpRewritePattern<
//       bishengir::memref_ext::AllocWorkspaceOp>::OpRewritePattern;

//   DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

//   explicit UpdateWorkSpaceAllocaOpOffsetPattern(
//       MLIRContext *context,
//       DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
//       : OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp>(context),
//         buffer2Offsets(buffer2Offsets) {}
//   LogicalResult matchAndRewrite(bishengir::memref_ext::AllocWorkspaceOp op,
//                                 PatternRewriter &rewriter) const final;
// };
} // namespace pto
} // namespace mlir

#endif // LLVM_PROJECT_ALLOCTOPOINTERCAST_H
ChatGPT said:
这份头文件整体结构没问题，但有几处会在 MLIR/LLVM 风格、性能、以及正确性 上踩坑。我按“你现在这样写会出什么问题 + 建议怎么改”给你直接点出来，并给一版更合理的写法（保持你现在的意图：给每个 alloc 生成 pto.pointercast，并能支持 multibuffer 对应多个地址）。
1) 这个 Pattern 里把 DenseMap 当成员 + 构造函数按值传参：会很贵，也不太对
你现在是：
DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

explicit MemrefAllocaOpToPointerCastOpPattern(
    MLIRContext *context,
    DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
    : OpRewritePattern<memref::AllocOp>(context),
      buffer2Offsets(buffer2Offsets) {}
问题：
按值传参会把整张 map 拷贝一遍；
Pattern 通常会被复制/移动（尤其是 RewritePatternSet 内部管理），你这个大成员会导致额外拷贝；
这张 map 本质是“pass 级别的分析结果/配置”，更适合 外部持有，pattern 引用/指针访问。
✅建议：把它改成 指针或引用包装，最常见是 const DenseMap<...> *：
const DenseMap<Value, SmallVector<uint64_t>> *buffer2Offsets;
构造函数传 const DenseMap&，存指针即可（由 pass 保证生命周期覆盖 applyPatternsAndFoldGreedily）：
MemrefAllocaOpToPointerCastOpPattern(
    MLIRContext *ctx,
    const DenseMap<Value, SmallVector<uint64_t>> &map)
  : OpRewritePattern(ctx), buffer2Offsets(&map) {}
You said:
//===- AllocToPointerCast.cpp - convert memref.AllocOp to pto.pointercastOp.//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AllocToPointerCast.h"
#include "mlir/Dialect/PTO/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOPOINTERCAST
#include "mlir/Dialect/PTO/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {} // namespace

LogicalResult MemrefAllocaOpToPointerCastOpPattern::matchAndRewrite(
    memref::AllocOp op, PatternRewriter &rewriter) const {
  const auto &currentMemRefType = cast<BaseMemRefType>(op.getType());
  auto iter = buffer2Offsets.find(op.getResult());
  assert(iter != buffer2Offsets.end() && "address should be found");
  SmallVector<Value> addrs;
  for (auto &offset : iter->second) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }
  auto ptoPointerCastOp = rewriter.create<pto::PointerCastOp>(
      op.getLoc(), currentMemRefType, ValueRange(addrs),
      ValueRange(op.getDynamicSizes()));
  rewriter.replaceOp(op, ptoPointerCastOp->getResults());
  return success();
}

// LogicalResult UpdateWorkSpaceAllocaOpOffsetPattern::matchAndRewrite(
//     bishengir::memref_ext::AllocWorkspaceOp op,
//     PatternRewriter &rewriter) const {
//   if (!op.getOffset().empty()) {
//     return failure();
//   }
//   auto iter = buffer2Offsets.find(op.getResult());
//   assert(iter != buffer2Offsets.end() && "address should be found");

//   SmallVector<Value> argOffset;
//   for (auto &offset : iter->second) {
//     Value newOffset =
//         rewriter.create<arith::ConstantIndexOp>(op->getLoc(), offset)
//             .getResult();
//     argOffset.push_back(newOffset);
//   }
//   auto allocWorkspaceOp =
//       rewriter.create<bishengir::memref_ext::AllocWorkspaceOp>(
//           op.getLoc(), op->getResultTypes(), op.getWorkspaceArg(),
//           op.getDynamicSize(), argOffset);
//   rewriter.replaceOp(op, allocWorkspaceOp->getResults());
//   return success();
// }
ChatGPT said:
这段 .cpp 的核心逻辑是对的：拿到 alloc 的 memref type + dynamic sizes，再用你预先算好的 offsets 生成多个地址常量，创建 pto.pointercast，最后 replace 掉 alloc。
但这里有 几个会直接导致编译失败/IR 类型不一致/后续 pass 乱掉 的点，我按严重程度从高到低说，并给你一版“更 MLIR/更稳”的改写。

0) 这段代码当前很可能编译不过：缺少 arith 头文件
你用了：
rewriter.create<arith::ConstantIntOp>(...)
但 .cpp 里没 include arith：
需要 #include "mlir/Dialect/Arith/IR/Arith.h"
否则会报 unknown namespace arith 或类型不完整。
✅修复：加上 arith include。

1) 你的常量类型用的是 i64，但“地址/offset”到底应该是 index 还是 i64？
你创建的是 ConstantIntOp(..., 64)，也就是 i64。
可 pto::PointerCastOp 的 operands 期待什么类型取决于你的 op 定义：

如果它的 addrs operand 类型是 index（很多 MLIR 里“offset/size”用 index），你这里会类型不匹配。
如果它定义就是 i64 地址，那 OK。
建议（更通用/更 MLIR）：如果这些 offsets 在 IR 层表达“地址”，一般建议用 index 或明确用 i64 但保持一致。最
You said:
//===- InferPTOMemScope.cpp - Infer Memory Scope for pto Ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InferPTOMemScope.h"
#include "mlir/Dialect/PTO/IR/PTO.h"
#include "mlir/Dialect/PTO/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"


#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "PTO-infer-mem-scope"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INFERPTOMEMSCOPE
#include "mlir/Dialect/PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace pto;

LogicalResult
MemScopeInferAndPropagateHelper::propagateMemScopeToUsers(Value val) {
  // Get new memory scope from result.
  auto memrefScope = getPTOAddressSpaceAttr(val.getType());
  // This function propagates the type change of an SSA result to the operation
  // that uses it. The result type of the updated operation might be affected,
  // so we need to cascade the change.
  auto propagateFn = [&](OpOperand &user) -> LogicalResult {
    Operation *userDefiningOp = user.getOwner();
    return TypeSwitch<Operation *, LogicalResult>(userDefiningOp)
        .Case<scf::YieldOp>([&](scf::YieldOp op) {
          Operation *parentOp = op->getParentOp();
          auto yieldResult = op.getOperand(user.getOperandNumber());
          auto parentResult = parentOp->getResult(user.getOperandNumber());

          Type yieldType = yieldResult.getType();
          Type valType = val.getType();
          if (!isa<BaseMemRefType>(yieldType))
            return success();
          if (!isa<BaseMemRefType>(valType))
            return success();
          auto mtype = dyn_cast<BaseMemRefType>(yieldType);
          auto vtype = dyn_cast<BaseMemRefType>(valType);
          if (mtype.getElementType() != vtype.getElementType())
            return success();
          setBaseMemRefTypeScope(parentResult, memrefScope);
          if (failed(propagateMemScopeToUsers(parentResult))) {
            return failure();
          }
          return success();
        })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          auto result = op.getTiedLoopResult(&user);
          setBaseMemRefTypeScope(result, memrefScope);
          auto bbArg = op.getTiedLoopRegionIterArg(&user);
          setBaseMemRefTypeScope(bbArg, memrefScope);
          return success(propagateMemScopeToUsers(bbArg).succeeded() &&
                         propagateMemScopeToUsers(result).succeeded());
        })
        .Case<memref::SubViewOp, memref::ViewOp, memref::ReinterpretCastOp,
              memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::ReshapeOp, memref::TransposeOp,
              memref::ExtractStridedMetadataOp>([&](auto op) {
          auto result = op->getResult(0);
          setBaseMemRefTypeScope(result, memrefScope);
          return propagateMemScopeToUsers(result);
        })
        // .Case<pto::BitcastOp>([&](auto op) {
        //   auto result = op->getResult(0);
        //   setBaseMemRefTypeScope(result, memrefScope);
        //   return propagateMemScopeToUsers(result);
        // })
        .Case<func::CallOp>([&](auto op) {
          // For function calls, we cannot propagate the memory scope because
          // we don't know the relationship between the inputs and results.
          // But we don't need to report failure because we can run propagation
          // for the results.
          return success();
        })
        .Case<gpu::LaunchFuncOp>([&](auto op) {
          // Same as above
          return success();
        })
        .Default([&](Operation *op) {
          // Don't need to update Ops that don't have results.
          if (op->getNumResults() == 0) {
            return success();
          }
          // Or results that are not memrefs.
          if (llvm::none_of(op->getResults(), [&](OpResult result) {
                return isa<MemRefType>(result.getType());
              })) {
            return success();
          }
          op->emitOpError("Unsupported user for root alloc op.");
          return failure();
        });
  };
  // Iterate over the users of the val.
  for (OpOperand &user : val.getUses()) {
    // Update the type of the result that corresponds to the operand.
    if (failed(propagateFn(user))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
MemScopeInferAndPropagateHelper::Run(Value operand,
                                     const AddressSpaceAttr &targetMemScope) {
  auto memRefType = dyn_cast<BaseMemRefType>(operand.getType());
  if (!memRefType) {
    return failure();
  }

  auto memSpace = memRefType.getMemorySpace();
  if (memSpace) {
    if (memSpace != targetMemScope) {
      return failure();
    }
    return success();
  }

  // Update its scope.
  setBaseMemRefTypeScope(operand, targetMemScope);

  // Propagate the new memref type to its users.
  return propagateMemScopeToUsers(operand);
}

namespace {
struct InferPTOMemScopePass
    : public impl::InferPTOMemScopeBase<InferPTOMemScopePass> {
  void runOnOperation() override;

private:
  LogicalResult fixDeviceCallSite(func::FuncOp op);
  LogicalResult fixHostFuncSignature(func::FuncOp op);
};
} // namespace

LogicalResult pto::inferAndPropagateMemScopeForMovDps(pto::MovDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  Value mA = op.getSrc();
  Value mB = op.getDst();

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);

  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  auto memRefType = dyn_cast<BaseMemRefType>(allocB.value().getType());
  if (!memRefType) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }

  auto memSpace = memRefType.getMemorySpace();
  if (!memSpace) {
    return success();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);
  auto l1SpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::MAT);
  auto ubSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::UB);
  auto biasSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::BIAS);

  MemScopeInferAndPropagateHelper helper;

  if (memSpace == ubSpaceAttr) {
    // For MmadL1Op, operand mA should be in L1.
    if (failed(helper.Run(*allocA, ubSpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for mA");
    }
    return success();
  }

  if (memSpace == l1SpaceAttr) {
    // For MmadL1Op, operand mA should be in L1.
    if (failed(helper.Run(*allocA, l0cSpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for mA");
    }
    return success();
  }

  if (memSpace == l0aSpaceAttr ||
      memSpace == l0bSpaceAttr ||
      memSpace == biasSpaceAttr) {
    // For MmadL1Op, operand mA should be in L1.
    if (failed(helper.Run(*allocA, l1SpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for mA");
    }
    return success();
  }

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForMatmulAccDps(pto::MatmulAccDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  // 替换 getDpsInputOperand/getDpsInitOperand
  // 直接使用 ODS 生成的命名函数，更直观且安全
  // 原逻辑: Input(0)->LHS, Input(1)->RHS, Init(0)->DST
  Value mAcc = op.getAccIn();
  Value mA = op.getLhs();
  Value mB = op.getRhs();
  Value mC = op.getDst();

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocAcc = tracebackMemRefToAlloc(mAcc);
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);
  auto allocC = tracebackMemRefToAlloc(mC);
  

  if (!allocAcc.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mAcc of this op.";
    return failure();
  }
  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);

  MemScopeInferAndPropagateHelper helper;

   // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocAcc, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mAcc");
  }

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l0aSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l0bSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  return success();
}


LogicalResult pto::inferAndPropagateMemScopeForMatmulBiasDps(pto::MatmulBiasDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  // 替换 getDpsInputOperand/getDpsInitOperand
  // 直接使用 ODS 生成的命名函数，更直观且安全
  // 原逻辑: Input(0)->LHS, Input(1)->RHS, Init(0)->DST
  Value mA = op.getA();
  Value mB = op.getB();
  Value mC = op.getDst(); 
  Value mD = op.getBias(); 

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);
  auto allocC = tracebackMemRefToAlloc(mC);
  auto allocD = tracebackMemRefToAlloc(mD);

  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }
  if (!allocD.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mD of this op.";
    return failure();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);
  auto l0dSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::BIAS);

  MemScopeInferAndPropagateHelper helper;

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l0aSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l0bSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mD should be in BIAS.
  if (failed(helper.Run(*allocD, l0dSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForMatmulDps(pto::MatmulDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  // 替换 getDpsInputOperand/getDpsInitOperand
  // 直接使用 ODS 生成的命名函数，更直观且安全
  // 原逻辑: Input(0)->LHS, Input(1)->RHS, Init(0)->DST
  Value mA = op.getLhs();
  Value mB = op.getRhs();
  Value mC = op.getDst(); 

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);
  auto allocC = tracebackMemRefToAlloc(mC);

  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);

  MemScopeInferAndPropagateHelper helper;

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l0aSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l0bSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  return success();
}

LogicalResult InferPTOMemScopePass::fixDeviceCallSite(func::FuncOp op) {
  LDBG("Begin fixing call site for " << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  SymbolTable::UseRange uses = *op.getSymbolUses(getOperation());
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp call = cast<func::CallOp>(use.getUser());
    // propagate call operand's memory scope
    for (auto [idx, callOperand] : llvm::enumerate(call.getArgOperands())) {
      if (!isa<BaseMemRefType>(callOperand.getType()))
        continue;

      auto funcOperandType = op.getFunctionType().getInput(idx);
      if (!isa<BaseMemRefType>(funcOperandType))
        continue;

      LDBG("call operand: " << callOperand);
      if (failed(helper.Run(tracebackMemRef(callOperand),
                            getPTOAddressSpaceAttr(funcOperandType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for operand "
               << callOperand;
      }
      LDBG("call operand after: " << callOperand);
    }
    // propagate call return value memory scope
    for (auto [idx, returnValue] : llvm::enumerate(call->getResults())) {
      if (!isa<BaseMemRefType>(returnValue.getType()))
        continue;

      auto funcReturnType = op.getFunctionType().getResult(idx);
      if (!isa<BaseMemRefType>(funcReturnType))
        continue;

      if (failed(helper.Run(returnValue,
                            getPTOAddressSpaceAttr(funcReturnType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for result " << returnValue;
      }
    }
  }
  return success();
}

/// Update the function type for the host function.
///
/// Because we propagate information from the call site to the caller, we only
/// updated the memref type of the BlockArgument of or the return operation
/// within the function (if they are updated at all). So we need to use those
/// information to update the function's type.
LogicalResult InferPTOMemScopePass::fixHostFuncSignature(func::FuncOp op) {
  // Skip external host functions because we know nothing about it.
  if (op.isExternal())
    return success();

  func::ReturnOp returnOp = getAssumedUniqueReturnOp(op);
  if (!returnOp)
    return failure();

  SmallVector<Type> newArgsType(llvm::map_to_vector(
      op.getArguments(), [](const BlockArgument &ba) { return ba.getType(); }));
  SmallVector<Type> newReturnType(llvm::map_to_vector(
      returnOp.getOperandTypes(), [](const Type &type) { return type; }));
  auto newFt = op.getFunctionType().clone(newArgsType, newReturnType);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult inferAndPropagateMemScopeForExternFunc(func::FuncOp op) {
  if (!op.isExternal())
    return failure();

  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);
  LDBG("Begin infer and propagate memory scope for extern func"
       << op.getSymName());
  auto newArgTypes = SmallVector<Type>(op.getArgumentTypes());
  for (auto &argType : newArgTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(argType)) {
      if (memrefType.getMemorySpace())
        continue;
      argType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  // For extern functions that have results, we assume that the memory scope
  // is Global Memory.
  auto newReturnTypes = SmallVector<Type>(op.getResultTypes());
  for (auto &resultType : newReturnTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(resultType)) {
      if (memrefType.getMemorySpace())
        continue;
      resultType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  auto newFt = op.getFunctionType().clone(newArgTypes, newReturnTypes);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForFunc(func::FuncOp op) {
  if (op.isExternal())
    return inferAndPropagateMemScopeForExternFunc(op);

  LDBG("Begin infer and propagate memory scope for func" << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::UB);
  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    if (op->hasAttr(pto::VectorFunctionAttr::name)) {
      if (failed(helper.Run(arg, ubSpaceAttr)))
        return op->emitOpError()
               << "Failed to propagate UB memory scope for argument # in VF"
               << arg.getArgNumber();
    } else if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }
  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }
  if (op->getNumResults() > 0)
    op.emitWarning()
        << "non-externl function has return value after bufferization!";

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op) {
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);

  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    // TODO: handle case when ub arguments are passed in the GPUFuncOp
    if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }

  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }

  return success();
}

LogicalResult pto::inferAndPropagateUbufMemScope(memref::AllocOp op) {
  LDBG("Begin infer and propagate memory scope for: " << *op);
  auto memorySpace = op.getType().getMemorySpace();
  if (memorySpace)
    return success();

  MemScopeInferAndPropagateHelper helper;
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::UB);
  if (failed(helper.Run(op, ubSpaceAttr))) {
    return op->emitOpError("Failed to propagate memory scope ub for allocOp");
  }
  return success();
}

void InferPTOMemScopePass::runOnOperation() {
  llvm::errs() << "Hello PTO Infer Mem Scope!\n";
  auto op = getOperation();
  op->dump();

  SmallVector<func::FuncOp> deviceFuncList;
  SetVector<StringRef> deviceFuncNames;
  SmallVector<func::FuncOp> hostFuncList;
  getOperation()->walk([&](func::FuncOp func) {
    // if (!hacc::utils::isHost(func)) {
    //   deviceFuncList.push_back(func);
    //   deviceFuncNames.insert(func.getSymName());
    //   return;
    // }
    //hostFuncList.push_back(func);
    deviceFuncList.push_back(func);
    deviceFuncNames.insert(func.getSymName());
    return;
  });

  SmallVector<gpu::GPUFuncOp> gpuFuncList;
  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    gpuModule->walk([&](gpu::GPUFuncOp gpuFunc) -> void {
      gpuFuncList.push_back(gpuFunc);
    });
  });

  for (auto func : gpuFuncList) {
    if (failed(inferAndPropagateMemScopeForGpuFunc(func)))
      signalPassFailure();
  }

  // Infer and propagate memory scope for device functions.
  for (auto func : deviceFuncList) {
    // Set the memory scope of values related to pto::MmadL1Op to L1 or L0C.
    func->walk([&](mlir::pto::MatmulDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulDps(op)))
        signalPassFailure();
    });

    func->walk([&](mlir::pto::MatmulAccDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulAccDps(op)))
        signalPassFailure();
    });

    func->walk([&](mlir::pto::MatmulBiasDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulBiasDps(op)))
        signalPassFailure();
    });

    func->walk([&](mlir::pto::MovDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMovDps(op)))
        signalPassFailure();
    });

    // Set device function arguments' memory scope to GM.
    if (failed(pto::inferAndPropagateMemScopeForFunc(func)))
      signalPassFailure();

    // Finally, set the remaining memory scope in the device kernel to UB.
    func->walk([&](memref::AllocOp op) {
      if (failed(pto::inferAndPropagateUbufMemScope(op)))
        signalPassFailure();
    });
  }

  for (auto func : deviceFuncList) {
    if (failed(fixDeviceCallSite(func)))
      signalPassFailure();
  }

  for (auto func : hostFuncList) {
    if (failed(fixHostFuncSignature(func)))
      signalPassFailure();
  }

  llvm::errs() << "end PTO Infer Mem Scope!\n";
  op = getOperation();
  op->dump();
}

std::unique_ptr<Pass> mlir::pto::createInferPTOMemScopePass() {
  return std::make_unique<InferPTOMemScopePass>();
}
ChatGPT said:
我按“会崩/会错”优先，把这份 InferPTOMemScope.cpp 里最关键的坑、以及怎么改成更稳的 MLIR 写法给你梳一遍。你这份代码的总体思路（从特定 PTO op 推导 address space，然后把 scope 沿 SSA 传播到 users，再修 func/call signature）是对的，但目前传播逻辑和 pass 的遍历顺序会导致 漏传播、错误失败、甚至 invalid IR。
1) 最大的逻辑 bug：propagateMemScopeToUsers 的 memScope 取错了来源
你在 propagateMemScopeToUsers(Value val) 里：
auto memrefScope = getPTOAddressSpaceAttr(val.getType());
但 val.getType() 可能仍然是 “没设置 scope 的 memref”，或者 scope 还没成功写回（尤其是你在 Run 里先 set，然后立刻递归传播，递归里又重新 getPTOAddressSpaceAttr(val.getType())）。
更关键：propagateMemScopeToUsers 的职责应该是“把 已知的 targetMemScope 传播出去”，而不是每次都从 val 的 type 里再反推一次（这一步会把未知 scope 传播成未知，或传播成错误默认）。

✅建议：把 propagateMemScopeToUsers 改为显式传 scope：

LogicalResult propagateMemScopeToUsers(Value val, AddressSpaceAttr scope);
然后 Run 里：
setBaseMemRefTypeScope(operand, targetMemScope);
return propagateMemScopeToUsers(operand, targetMemScope);
这样不会“读回错误”。
2) scf::YieldOp case 写法不对：yieldResult/parentResult 的对应关系被你假设错了
你现在：
auto yieldResult = op.getOperand(user.getOperandNumber());
auto parentResult = parentOp->getResult(user.getOperandNumber());
但这里的 user 是 “val 的 use”，也就是 yield 的某个 operand。user.getOperandNumber() 是 yield operand index，这个 index 在 scf.yield 内是对的；但 parentOp->getResult(i) 是否对应同一个 i 要看 parent 是哪种 op：
scf.if：yield operand i 对应 if 的 result i ✅
scf.execute_region：同理 ✅
scf.for/scf.while 不用 yield 的 operand index 直接对应 results（尤其 for 的 iter_args/result 有 “tied” 关系）
你虽然对 scf::ForOp 做了单独 case，但 scf::YieldOp 的 parent 不止 if/execute_region，可能也在 while 的 region。
✅建议：

在 scf::YieldOp case 里用 parent op 的类型再细分（if/execute_region/while），或者干脆只处理你确认会出现的 parent（比如 if/execute_region），其他返回 success（别 failure）。
或者把 scf::YieldOp case 删掉，改为对 parent（if/for/while）做传播（更推荐）。
3) scf::ForOp 的传播：你同时递归 bbArg 和 result，但 success(expr) 这写法很危险
你写：
return success(propagateMemScopeToUsers(bbArg).succeeded() &&
               propagateMemScopeToUsers(result).succeeded());
问题：
你这里把 LogicalResult 的失败信息吞了（只保留 bool）。
如果其中一个失败，你返回 failure()，但 错误位置/错误 op 的诊断丢了，调试很痛苦。
✅建议：按 MLIR 习惯写成：
if (failed(propagateMemScopeToUsers(bbArg, scope))) return failure();
if (failed(propagateMemScopeToUsers(result, scope))) return failure();
return success();
4) 传播到 “普通 op” 的 Default 分支太激进，会让 pass 非常脆
你 Default 分支是：
if (op->getNumResults() == 0) return success();
if (none_of(results are MemRefType)) return success();
op->emitOpError("Unsupported user for root alloc op.");
return failure();
这意味着：只要有任何一个 user 是产出 memref result 的 op（比如 linalg、arith.select?、bufferization.to_memref?、affine.apply? 等）你没列入 case，就直接 failure。
这在真实 pipeline 里几乎必炸。

✅建议两种路线（二选一）：

路线 A（保守但稳）：只传播你能证明 “结果与 operand 同 alias/同 underlying buffer”的 op
例如：各种 view/cast/subview/reshape/transpose/expand/collapse，这些你已经列了一堆。对其它 op 直接 return success()（不报错）。
优点：不炸。
缺点：可能漏传播，但后面你还有 “最后把剩余 alloc 设 UB” 兜底。
路线 B（更完整）：用 Bufferization 的 alias analysis / MemoryEffect 来推导
这比较复杂，但更正确。你现在已经 include bufferization，可以进一步：
用 `
You said:
//===---------------- InsertLoadStoreForMixCV.cpp -------------------------===//
//
// cube_matmul_vadd_2d rewrite:
//
// - add args: %arg_ws: memref<?xi8,  #pto.address_space<gm>>
//           : %arg_ffts: memref<?xi64, #pto.address_space<gm>>
// - insert: pto.set_ffts %arg_ffts
// - force tile_idx_x/y -> 0 and erase them
// - replace tail:
//     (CC -> CBUF) + addf(CBUF) + store(GM)
//   with:
//     CC -> workspace(GM view from %arg_ws) -> UB
//     C(GM) -> UB
//     addf(UB)
//     store(GM)
//
// Notes:
// - Do NOT use AsmParser / parseAttribute.
// - Avoid memref_ext.alloc_workspace (unregistered dialect crash) by using memref.view.
// - PTO generated builders: prefer the shortest builder to avoid optional-arg mismatch.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PTO/IR/PTO.h"
#include "mlir/Dialect/PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"   // IRRewriter
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
namespace pto {

namespace {

// ------------------------- helpers -------------------------

static bool memSpaceContains(Type t, StringRef needle) {
  auto mr = dyn_cast<MemRefType>(t);
  if (!mr)
    return false;
  Attribute ms = mr.getMemorySpace();
  if (!ms)
    return false;
  std::string s;
  llvm::raw_string_ostream os(s);
  ms.print(os);
  os.flush();
  return StringRef(s).contains(needle);
}

static arith::ConstantIndexOp getOrCreateC0(IRRewriter &rewriter,
                                            func::FuncOp f) {
  Block &entry = f.front();
  for (Operation &op : entry.getOperations()) {
    if (auto cst = dyn_cast<arith::ConstantIndexOp>(op)) {
      if (cst.value() == 0)
        return cst;
    }
  }
  rewriter.setInsertionPointToStart(&entry);
  return rewriter.create<arith::ConstantIndexOp>(f.getLoc(), 0);
}

static memref::AllocOp createAllocWithAlign(IRRewriter &rewriter, Location loc,
                                            MemRefType ty, int64_t align = 64) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, ty);
  alloc->setAttr("alignment", rewriter.getI64IntegerAttr(align));
  return alloc;
}

// ------------------------- matching -------------------------
//
// Match old tail pattern:
//
//   %mm_cc = pto.matmul_dps ... outs(%mm_cc : ...cc...)
//   %mm_cbuf = (pto.load_dps/pto.copy/pto.mov) ins(%mm_cc) ... -> ...cbuf...
//   %c_cbuf = pto.load_dps ins(%c_gm) outs(%c_cbuf)
//   %out_cbuf = pto.addf_dps ins(%c_cbuf, %mm_cbuf) outs(%out_cbuf)
//   pto.store_dps ins(%out_cbuf) outs(%o_gm)
//
// and capture %c_gm and %o_gm.
//

struct MatchInfo {
  Operation *matmulDps = nullptr;
  Value mmCCBuf;                 // ...cc...

  Operation *ccToVec = nullptr;  // CC->CBUF
  Value mmVecBuf;                // ...cbuf...

  Operation *cGmToCbuf = nullptr;
  Value cGMSubview;              // ...gm...
  Value cCbufBuf;                // ...cbuf...

  Operation *addfCbuf = nullptr;
  Value outCbufBuf;              // ...cbuf...

  Operation *storeToOut = nullptr;
  Value outGMSubview;            // ...gm...
};

static bool match(func::FuncOp func, MatchInfo &mi) {
  if (func.getNumArguments() != 11)
    return false;

  // 1) find matmul_dps and dst(CC)
  for (Operation &op : func.front().getOperations()) {
    if (op.getName().getStringRef() != "pto.matmul_dps")
      continue;
    if (op.getNumOperands() < 1)
      continue;
    Value dst = op.getOperand(op.getNumOperands() - 1);
    if (!memSpaceContains(dst.getType(), "cc"))
      continue;
    mi.matmulDps = &op;
    mi.mmCCBuf = dst;
    break;
  }
  if (!mi.matmulDps || !mi.mmCCBuf)
    return false;

  // 2) find final store_dps (CBUF -> GM)
  for (Operation &op : func.front().getOperations()) {
    if (op.getName().getStringRef() != "pto.store_dps")
      continue;
    if (op.getNumOperands() < 2)
      continue;
    Value src = op.getOperand(0);
    Value dst = op.getOperand(1);
    if (memSpaceContains(src.getType(), "cbuf") && memSpaceContains(dst.getType(), "gm")) {
      mi.storeToOut = &op;
      mi.outGMSubview = dst;
      break;
    }
  }
  if (!mi.storeToOut)
    return false;

  // 3) addf_dps should define store src
  Value storeSrc = mi.storeToOut->getOperand(0);
  Operation *addfDef = storeSrc.getDefiningOp();
  if (!addfDef || addfDef->getName().getStringRef() != "pto.addf_dps")
    return false;
  mi.addfCbuf = addfDef;

  if (mi.addfCbuf->getNumOperands() < 3)
    return false;
  mi.outCbufBuf = mi.addfCbuf->getOperand(mi.addfCbuf->getNumOperands() - 1);

  Value addIn0 = mi.addfCbuf->getOperand(0);
  Value addIn1 = mi.addfCbuf->getOperand(1);
  if (!(memSpaceContains(addIn0.getType(), "cbuf") &&
        memSpaceContains(addIn1.getType(), "cbuf")))
    return false;

  // 4) locate CC->CBUF op producing one add input
  auto isCCtoCBUF = [&](Operation *op, Value &dstOut) -> bool {
    StringRef n = op->getName().getStringRef();
    if (n != "pto.load_dps" && n != "pto.copy" && n != "pto.mov")
      return false;
    if (op->getNumOperands() < 2)
      return false;

    Value src = op->getOperand(0);
    Value dst = op->getOperand(op->getNumOperands() - 1);

    if (!memSpaceContains(src.getType(), "cc"))
      return false;
    if (!memSpaceContains(dst.getType(), "cbuf"))
      return false;

    dstOut = dst;
    return true;
  };

  Value mmVec;
  Operation *cc2vec = nullptr;
  for (Operation &op : func.front().getOperations()) {
    Value dst;
    if (!isCCtoCBUF(&op, dst))
      continue;
    if (op.getOperand(0) != mi.mmCCBuf)
      continue;
    if (dst == addIn0 || dst == addIn1) {
      cc2vec = &op;
      mmVec = dst;
      break;
    }
  }
  if (!cc2vec || !mmVec)
    return false;

  mi.ccToVec = cc2vec;
  mi.mmVecBuf = mmVec;

  // the other add input is C(cbuf)
  mi.cCbufBuf = (addIn0 == mmVec) ? addIn1 : addIn0;

  // 5) find GM->CBUF load for C
  for (Operation &op : func.front().getOperations()) {
    if (op.getName().getStringRef() != "pto.load_dps")
      continue;
    if (op.getNumOperands() < 2)
      continue;
    Value src = op.getOperand(0);
    Value dst = op.getOperand(1);
    if (dst != mi.cCbufBuf)
      continue;
    if (!memSpaceContains(src.getType(), "gm"))
      continue;
    if (!memSpaceContains(dst.getType(), "cbuf"))
      continue;
    mi.cGmToCbuf = &op;
    mi.cGMSubview = src;
    break;
  }
  if (!mi.cGmToCbuf || !mi.cGMSubview)
    return false;

  return true;
}

// ------------------------- pass -------------------------

struct InsertLoadStoreForMixCVPass
    : public PassWrapper<InsertLoadStoreForMixCVPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertLoadStoreForMixCVPass)

  StringRef getArgument() const override {
    return "pto-insert-load-store-for-mix-cv";
  }
  StringRef getDescription() const override {
    return "Insert CC->workspace(GM)->UB bridge + set_ffts + signature tweak (guarded)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.getName() != "cube_matmul_vadd_2d")
      return;

    MatchInfo mi;
    if (!match(func, mi))
      return;

    IRRewriter rewriter(func.getContext());
    Location loc = func.getLoc();
    Block &entry = func.front();

    llvm::errs() << "\n[InsertLoadStoreForMixCV] MATCH func: " << func.getName()
                 << "\n---- BEFORE ----\n";
    func.print(llvm::errs());
    llvm::errs() << "\n--------------\n";

    // (A) force tile_idx_x/y -> 0 and erase them (original args[9],[10])
    Value c0 = getOrCreateC0(rewriter, func).getResult();
    {
      Value tileX = entry.getArgument(9);
      Value tileY = entry.getArgument(10);
      tileX.replaceAllUsesWith(c0);
      tileY.replaceAllUsesWith(c0);
      func.eraseArgument(10);
      func.eraseArgument(9);
    }

    // (B) inherit gmSpace from existing GM subview type
    auto gmSubviewTy = dyn_cast<MemRefType>(mi.cGMSubview.getType());
    if (!gmSubviewTy || !gmSubviewTy.getMemorySpace()) {
      llvm::errs() << "[InsertLoadStoreForMixCV] ERROR: cannot get gm memorySpace\n";
      signalPassFailure();
      return;
    }
    Attribute gmSpace = gmSubviewTy.getMemorySpace();

    // (C) create ubSpace using enum API
    Attribute ubSpace = pto::AddressSpaceAttr::get(func.getContext(), pto::AddressSpace::UB);
    if (!ubSpace) {
      llvm::errs() << "[InsertLoadStoreForMixCV] ERROR: cannot create ub address_space attr\n";
      signalPassFailure();
      return;
    }

    // (D) insert new args (ws, ffts) at front
    MemRefType wsArgTy =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{}, gmSpace);
    MemRefType fftsArgTy =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type(),
                        MemRefLayoutAttrInterface{}, gmSpace);

    func.insertArgument(0, wsArgTy, DictionaryAttr{}, loc);
    func.insertArgument(1, fftsArgTy, DictionaryAttr{}, loc);

    Value argWS = entry.getArgument(0);
    Value argFFTs = entry.getArgument(1);

    // (E) insert pto.set_ffts near top (after constants)
    {
      Operation *insertPt = &entry.front();
      for (Operation &op : entry.getOperations()) {
        if (isa<arith::ConstantOp>(op) || isa<arith::ConstantIndexOp>(op)) {
          insertPt = op.getNextNode() ? op.getNextNode() : &op;
          continue;
        }
        break;
      }
      rewriter.setInsertionPoint(insertPt);
      rewriter.create<pto::SetFFTsOp>(loc, argFFTs);
    }

    // (F) types
    MemRefType wsTileTy =
        MemRefType::get({32, 32}, rewriter.getF32Type(),
                        MemRefLayoutAttrInterface{}, gmSpace);
    MemRefType ubTileTy =
        MemRefType::get({32, 32}, rewriter.getF32Type(),
                        MemRefLayoutAttrInterface{}, ubSpace);

    // (G) insert bridge after matmul
    rewriter.setInsertionPointAfter(mi.matmulDps);

    // workspace view: memref.view %arg_ws -> memref<32x32xf32, gm>
    // byte_shift = 0; sizes empty (no dynamic dims).
    Value wsTile = rewriter.create<memref::ViewOp>(
        loc, wsTileTy, argWS, c0, ValueRange{}).getResult();

    // CC -> workspace(GM)
    rewriter.create<pto::StoreDpsOp>(loc, TypeRange{}, mi.mmCCBuf, wsTile);

    // workspace(GM) -> UB
    auto tmatUb = createAllocWithAlign(rewriter, loc, ubTileTy, 64);
    rewriter.create<pto::LoadDpsOp>(loc, TypeRange{}, wsTile, tmatUb.getResult());

    // C(GM) -> UB
    auto cUb = createAllocWithAlign(rewriter, loc, ubTileTy, 64);
    rewriter.create<pto::LoadDpsOp>(loc, TypeRange{}, mi.cGMSubview, cUb.getResult());

    // addf on UB
    auto outUb = createAllocWithAlign(rewriter, loc, ubTileTy, 64);
    rewriter.create<pto::AddFDpsOp>(loc, TypeRange{}, cUb.getResult(),
                                    tmatUb.getResult(), outUb.getResult());

    // UB -> OUT(GM)
    rewriter.create<pto::StoreDpsOp>(loc, TypeRange{}, outUb.getResult(), mi.outGMSubview);

    // (H) erase old tail ops
    rewriter.eraseOp(mi.storeToOut);
    rewriter.eraseOp(mi.addfCbuf);
    rewriter.eraseOp(mi.ccToVec);
    rewriter.eraseOp(mi.cGmToCbuf);

    llvm::errs() << "---- AFTER ----\n";
    func.print(llvm::errs());
    llvm::errs() << "\n==============\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createPTOInsertLoadStoreForMixCVPass() {
  return std::make_unique<InsertLoadStoreForMixCVPass>();
}

} // namespace pto
} // namespace mlir
ChatGPT said:
这份 InsertLoadStoreForMixCV.cpp 写得挺“能跑起来”的，但我能一眼看到几个 会在 MLIR pipeline 里直接炸 或 会悄悄生成非法 IR / 语义不对 的点。下面按优先级从高到低给你做一次“带修法”的 code review（不啰嗦，直接落到具体行/模式）。
1) 绝对会炸的：eraseArgument 顺序 + BlockArgument 缓存失效
你这里：
Value tileX = entry.getArgument(9);
Value tileY = entry.getArgument(10);
tileX.replaceAllUsesWith(c0);
tileY.replaceAllUsesWith(c0);
func.eraseArgument(10);
func.eraseArgument(9);
这段本身顺序是对的（先删 10 再删 9），但 前面你后来又 insertArgument(0/1)，会导致后续你“认为的 arg index”变化。如果你把插参放在删参之前，9/10 就错了；你现在删参在插参之前 ✅，但是仍有一个坑：
Value tileX/tileY 是 BlockArgument，删 argument 后这些 handle 不能再用（你没再用，OK）。
但你后面又用 entry.getArgument(0/1)，这是基于“插参以后 entry args 重排”的正确用法 ✅。
建议：为了不让未来维护的人踩坑，把 (A) 的注释写成“必须在 insertArgument 之前做”，并把 9/10 用常量命名（更稳）。
2) memref.view 的 byte_shift 你传了 index 0，但这个 op 要求 i64
你写：
Value wsTile = rewriter.create<memref::ViewOp>(
    loc, wsTileTy, argWS, c0, ValueRange{}).getResult();
memref.view 的 ODS 一般是：
source: memref<?xi8>
byte_shift: index 或 i64？ —— 这里不同 MLIR 版本确实有过变动，但在很多版本里 byte_shift 是 index，在另一些版本里是 i64。
你上面 arg_ws 是 memref<?xi8, gm>，你又要 view 成 memref<32x32xf32, gm>，这会涉及 element type reinterpret，view 允许，但 byte_shift 的类型必须严格匹配当前 MLIR 的定义。
✅稳妥写法（兼容

You said:
#include "mlir/Dialect/PTO/Transforms/MemoryDependentAnalyzer.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
// [Debug] 打印 Value 详细信息
static void printValueDebug(const char* tag, Value v) {
  llvm::errs() << tag << ": ";
  if (!v) {
    llvm::errs() << "NULL\n";
    return;
  }
  
  if (auto *op = v.getDefiningOp()) {
    llvm::errs() << "OpResult defined by [" << op->getName() << "]";
  } else {
    llvm::errs() << "BlockArgument";
  }
  llvm::errs() << " | Type: " << v.getType() << "\n";
}
 
// [Fix & Debug] 增强版 GetRealRoot
static Value GetRealRoot(Value v) {
  llvm::errs() << "  [Trace] GetRealRoot Start:\n";
  printValueDebug("    Current", v);
  
  int depth = 0;
  const int maxDepth = 20;
 
  while (v && depth++ < maxDepth) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
        llvm::errs() << "    -> Reached BlockArgument. Stop.\n";
        break; 
    }
 
    if (auto op = dyn_cast<memref::CollapseShapeOp>(defOp)) {
        llvm::errs() << "    -> Hit CollapseShapeOp. Peel off.\n";
        v = op.getSrc();
        continue;
    }
    if (auto op = dyn_cast<memref::ExpandShapeOp>(defOp)) {
        llvm::errs() << "    -> Hit ExpandShapeOp. Peel off.\n";
        v = op.getSrc();
        continue;
    }
    if (auto op = dyn_cast<memref::ViewOp>(defOp)) {
        llvm::errs() << "    -> Hit ViewOp. Peel off.\n";
        v = op.getSource();
        continue;
    }
    if (auto view = dyn_cast<ViewLikeOpInterface>(defOp)) {
        llvm::errs() << "    -> Hit ViewLikeInterface. Peel off.\n";
        v = view.getViewSource();
        continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
        v = cast.getSource();
        continue;
    }
    if (auto reCast = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
        v = reCast.getSource();
        continue;
    }
 
    llvm::errs() << "    -> Hit Alloc/Other [" << defOp->getName() << "]. Stop.\n";
    break;
  }
  return v;
}
 
bool MemoryDependentAnalyzer::DepBetween(
    const SmallVector<const BaseMemInfo *> &a,
    const SmallVector<const BaseMemInfo *> &b,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  
  // [Debug Log] 关键入口信息
  llvm::errs() << "\n[DepBetween] Checking dependency...\n";
  llvm::errs() << "  Vec A Size: " << a.size() << "\n";
  llvm::errs() << "  Vec B Size: " << b.size() << "\n";
 
  bool hasAlias = false;
  for (auto &i : a) {
    for (auto &j : b) {
      if (MemAlias(i, j)) {
        depBaseMemInfosVec.push_back(std::make_pair(i, j));
        hasAlias = true;
      }
    }
  }
  return hasAlias;
}
 
bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
                                       const BaseMemInfo *b) {
  pto::AddressSpace as = a->scope;
  pto::AddressSpace bs = b->scope;
 
  // [Debug Log] 打印比较对象
  llvm::errs() << "  [MemAlias Check]\n";
  printValueDebug("    Root A", a->rootBuffer);
  printValueDebug("    Root B", b->rootBuffer);
  llvm::errs() << "    Scope A: " << (int)as << ", Scope B: " << (int)bs << "\n";
 
  if (as != bs) {
    llvm::errs() << "    -> Scope Mismatch. False.\n";
    return false;
  }
 
  // 1. GM 内存
  if (as == pto::AddressSpace::GM) {
    return isGMBufferOverlap(a, b);
  }
 
  // 2. Local Memory (UB/L1)
  
  if (a->rootBuffer == b->rootBuffer) {
    if (a->baseAddresses.empty() || b->baseAddresses.empty()) return true;
    return isBufferAddressRangeOverlap(a, b);
  }
 
  // 2.2 深层比较：穿透 View
  Value realRootA = GetRealRoot(a->rootBuffer);
  Value realRootB = GetRealRoot(b->rootBuffer);
 
  llvm::errs() << "    [Deep Check] Surface Roots differ. Digging deeper...\n";
  printValueDebug("      Real Root A", realRootA);
  printValueDebug("      Real Root B", realRootB);
 
  if (realRootA == realRootB && realRootA != nullptr) {
      llvm::errs() << "      -> MATCH! Real roots are the same.\n";
      return true;
  } else {
      llvm::errs() << "      -> Mismatch. Real roots differ.\n";
  }
 
  return false;
}
 
bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                const BaseMemInfo *b) {
  if (a->rootBuffer != b->rootBuffer) {
    Value realRootA = GetRealRoot(a->rootBuffer);
    Value realRootB = GetRealRoot(b->rootBuffer);
    
    if (realRootA != realRootB) {
        return false;
    }
    return true; 
  }
 
  if (a->baseAddresses.empty() || b->baseAddresses.empty()) return true; 
  if (a->allocateSize == 0 || b->allocateSize == 0) return true;
 
  return isBufferAddressRangeOverlap(a, b);
}
 
bool MemoryDependentAnalyzer::isBufferAddressRangeOverlap(
    const BaseMemInfo *a, const BaseMemInfo *b) {
  int aBaseAddressesSize = static_cast<int>(a->baseAddresses.size());
  int bBaseAddressesSize = static_cast<int>(b->baseAddresses.size());
  
  for (int i = 0; i < aBaseAddressesSize; i++) {
    for (int j = 0; j < bBaseAddressesSize; j++) {
      if (isBufferOverlap(a, b, i, j)) {
        return true;
      }
    }
  }
  return false;
}
 
bool MemoryDependentAnalyzer::isBufferOverlap(const BaseMemInfo *a,
                                              const BaseMemInfo *b, int aIndex,
                                              int bIndex) {
  uint64_t aStart = a->baseAddresses[aIndex];
  uint64_t bStart = b->baseAddresses[bIndex];
  uint64_t aEnd = aStart + a->allocateSize;
  uint64_t bEnd = bStart + b->allocateSize;
 
  uint64_t maxStart = std::max(aStart, bStart);
  uint64_t minEnd = std::min(aEnd, bEnd);
 
  return maxStart < minEnd;
}
ChatGPT said:
你这版 MemoryDependentAnalyzer 的 debug 很有帮助，但现在的 alias 逻辑会出现 “误判 alias” 和 “漏判 alias” 两类问题，而且有一个明显的 C++/MLIR 小 bug 会让判断永远走错分支。下面我按优先级给你指出来，并给出一版“更稳、更 MLIR 风格”的修法（保持你现在的设计：GM 用 address range overlap，Local 用 root + optional range）。
0) 直接的 bug：realRootA != nullptr 这一句是错的
Value 不是指针，不能跟 nullptr 比。你这里：
if (realRootA == realRootB && realRootA != nullptr) {
应该写：
if (realRootA == realRootB && realRootA) { ... }
否则这句要么编译不过，要么被隐式转换搞出诡异行为。
1) 最大的语义问题：Local Memory 下 “real root 相同就 alias=true” 太粗
你现在 Local（UB/L1 等）逻辑：
如果 rootBuffer 相同：
没有 baseAddresses -> true
否则按 range overlap
如果 rootBuffer 不同：
穿透 view/cast 得到 realRoot
realRoot 相同就直接 true
这会造成 大量误判：
memref.view/subview 从同一个大 buffer 切出两个不重叠区域
你现在会因为 realRoot 相同直接 alias=true，但实际上应该是 可能不 alias（需要结合 offset/size）。
memref.reinterpret_cast、expand/collapse_shape 也可能表达不同的 slice。
✅建议：
“realRoot 相同”只能说明 来自同一块 backing buffer，不能直接说明 overlap。应该尽可能拿到 byte offset + size 再判断；拿不到再保守返回 true。
你现在已经有 baseAddresses + allocateSize，这其实可以扩展为：

把 GetRealRoot 改成返回 (root, byteOffset) 或者 (root, maybeConstantOffset)
对 memref.view/subview/reinterpret_cast 能推常量 offset 的推出来
推不出来就保守 alias=true
2) GetRealRoot：你 peel off 的 op 列表不完整，而且顺序有风险
你已经处理了：
collapse/expand
view
ViewLikeOpInterface（很好）
cast / reinterpret_cast
但 漏了最常见的：
memref::SubViewOp（非常关键）
memref::TransposeOp
memref::ReshapeOp
memref::ExtractStridedMetadataOp（有时会出现）
bufferization.to_memref / to_tensor（如果 pipeline 里还会出现）
你用 ViewLikeOpInterface 可以覆盖一部分（subview/view），但不是所有 memref op 都实现这个 interface（尤其是某些版本/自定义 op）。
✅建议：显式加上 memref::SubViewOp 优先处理；其次再用 ViewLikeInterface 兜底。

3) GM 判定：你在 “rootBuffer 不同但 realRoot 相同” 直接返回 true，可能也误判
GM 分支：
if (a->rootBuffer != b->rootBuffer) {
  Value realRootA = GetRealRoot(a->rootBuffer);
  Value realRootB = GetRealRoot(b->rootBuffer);
  if (realRootA != realRootB) return false;
  return true;
}
这同样会误判：两个 subview/view 来自同一 GM 大 buffer，但 slice 不重叠。
如果 GM 的 baseAddresses 代表“最终物理地址”（比如你 AllocToPointerCast 已经算好每个 view 的 base address），那 应该用 baseAddresses 去比，而不是只看 root。

✅建议：

如果 baseAddresses 非空，就优先用 range overlap（无论 rootBuffer 是否相同）
只有 baseAddresses 为空才 fallback 到 realRoot
4) 现在的行为对 “未知地址/未知大小” 太随意：会导致过多依赖边
你很多地方遇到不完整信息就直接 return true：
if (a->baseAddresses.empty() || b->baseAddresses.empty()) return true;
if (a->allocateSize == 0 || b->allocateSize == 0) return true;
这会让依赖分析变得非常保守（几乎都冲
You said:
#include "mlir/Dialect/PTO/Transforms/MoveSyncState.h"
#include "llvm/ADT/STLExtras.h" // For llvm::reverse
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
void MoveSyncState::Run() {
  MoveOutBranchSync();
  MoveForSync();
}
 
// ============================================================================
// Branch (If/Else) Logic
// ============================================================================
 
void MoveSyncState::MoveOutBranchSync() {
  for (auto &e : syncIR_) {
    if (auto *branchElement = dyn_cast<BranchInstanceElement>(e.get())) {
      // 只处理 IF_BEGIN，它是整个 Block 的入口
      if (branchElement->getBranchKind() == KindOfBranch::IF_BEGIN) {
        std::pair<unsigned, unsigned> bound = {branchElement->beginId,
                                               branchElement->endId};
        
        // 1. 遍历 THEN 分支内的指令
        for (unsigned i = branchElement->beginId + 1;
             i < branchElement->branchId; i++) {
          PlanMoveOutBranchSync(
              syncIR_[i].get(),
              {branchElement->beginId, branchElement->branchId}, bound);
        }
 
        // 如果没有 ELSE 分支，跳过
        if (branchElement->endId == branchElement->branchId) {
          continue;
        }
 
        // 2. 遍历 ELSE 分支内的指令
        for (unsigned i = branchElement->branchId + 1; i < branchElement->endId;
             i++) {
          PlanMoveOutBranchSync(syncIR_[i].get(),
                                {branchElement->branchId, branchElement->endId},
                                bound);
        }
      }
    }
  }
}
 
void MoveSyncState::PlanMoveOutBranchSync(
    InstanceElement *e, std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  
  // 处理 PipeBefore (Wait/Barrier) - 保持优化 (Hoist Wait)
  SyncOps newPipeBefore;
  for (auto &s : e->pipeBefore) {
    PlanMoveOutIfWaitSync(newPipeBefore, s, pair, bound);
  }
  e->pipeBefore = newPipeBefore;
 
  // [Fix] 禁用 If/Else 分支的 Set Sink 优化
  // 原因：如果在 If/Else 中通过补偿机制生成了 Set，移动它们会导致双重执行和逻辑错误。
  // NPU 场景下，条件 Set 必须保留在控制流内部。
  /* SyncOps newPipeAfter;
  for (auto &s : llvm::reverse(e->pipeAfter)) {
    PlanMoveOutIfSetSync(newPipeAfter, s, pair, bound);
  }
  e->pipeAfter = newPipeAfter;
  */
  // 保持原样，不做任何移动
}
 
void MoveSyncState::PlanMoveOutIfWaitSync(
    SyncOps &newPipeBefore, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  
  // 只处理 WaitEvent
  if (s->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    newPipeBefore.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(!syncPair.empty(), "expected syncPair not to be empty");
  
  // 找到配对的 Set 操作
  auto *setSync = syncPair[0].get();
 
  // 如果 Set 操作在 If 块的外部 (index < pair.first 或 index > pair.second)
  // 那么这个 Wait 可以被提至 If 之前 (bound.first)
  if ((setSync->GetSyncIRIndex() >= pair.second) ||
      (setSync->GetSyncIRIndex() <= pair.first)) {
    
    // [Optimization]: Hoist Wait out of If
    checkSyncIRIndex(syncIR_, bound.first);
    syncIR_[bound.first]->pipeBefore.push_back(s); // 移到 IfBegin 之前
    s->SetSyncIRIndex(bound.first); // 更新索引
  } else {
    // 无法移动，保留在原地
    newPipeBefore.push_back(s);
  }
}
 
void MoveSyncState::PlanMoveOutIfSetSync(
    SyncOps &newPipeAfter, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  
  if (s->GetType() != SyncOperation::TYPE::SET_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) {
    newPipeAfter.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(syncPair.size() > 1, "expected syncPair size > 1");
  
  // 找到配对的 Wait 操作
  auto *waitSync = syncPair[1].get();
 
  // 如果 Wait 操作在 If 块的外部
  // 那么这个 Set 可以沉降到 If 之后 (bound.second)
  if ((waitSync->GetSyncIRIndex() >= pair.second) ||
      (waitSync->GetSyncIRIndex() <= pair.first)) {
    
    // [Optimization]: Sink Set out of If
    checkSyncIRIndex(syncIR_, bound.second);
    syncIR_[bound.second]->pipeAfter.push_front(s); // 移到 IfEnd 之后
    s->SetSyncIRIndex(bound.second);
  } else {
    newPipeAfter.push_back(s);
  }
}
 
// ============================================================================
// Loop Optimization Logic
// ============================================================================
 
void MoveSyncState::MoveForSync() {
  for (auto &e : syncIR_) {
    if (auto *forCompound = dyn_cast<LoopInstanceElement>(e.get())) {
      // 找到 Loop End 节点（代表循环体的结束）
      if (forCompound->getLoopKind() == KindOfLoop::LOOP_END) {
        if (forCompound->ignore_block_sync_move_out) {
          continue;
        }
        // 遍历循环体内的所有指令
        for (unsigned i = forCompound->beginId + 1; i < forCompound->endId; i++)
          MoveOutSync(syncIR_[i].get(),
                      {forCompound->beginId, forCompound->endId});
      }
    }
  }
}
 
void MoveSyncState::MoveOutSync(InstanceElement *e,
                                std::pair<unsigned int, unsigned int> pair) {
  checkCondition(pair.first < e->GetIndex() && e->GetIndex() < pair.second,
                 "MoveOutSync expected element to be within pair bounds");
  
  // 处理 PipeBefore (Wait/Barrier)
  SyncOps newPipeBefore;
  for (auto &s : e->pipeBefore) {
    PlanMoveOutWaitSync(newPipeBefore, s, pair);
  }
  e->pipeBefore = newPipeBefore;
 
  // 处理 PipeAfter (Set)
  SyncOps newPipeAfter;
  for (auto &s : llvm::reverse(e->pipeAfter)) {
    PlanMoveOutSetSync(newPipeAfter, s, pair);
  }
  e->pipeAfter = newPipeAfter;
}
 
void MoveSyncState::PlanMoveOutWaitSync(
    SyncOps &newPipeBefore, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair) {
  
  if (s->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    newPipeBefore.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(!syncPair.empty(), "expected syncPair not to be empty");
  auto *setSync = syncPair[0].get();
 
  // 如果 Set 操作在 Loop 外部 (index > loop_end 或 index < loop_begin)
  // 说明依赖不来自循环内部（非 Loop-Carried Dependency）
  // 可以将 Wait 提至 Loop Begin 之前
  if ((setSync->GetSyncIRIndex() > pair.second) ||
      (setSync->GetSyncIRIndex() < pair.first)) {
    
    // [Optimization]: Hoist Wait out of Loop
    checkSyncIRIndex(syncIR_, pair.first);
    // pair.first 是 LoopBegin 节点
    syncIR_[pair.first]->pipeBefore.push_back(s); 
    s->SetSyncIRIndex(pair.first);
    return;
  }
  
  // 否则依赖来自循环内部，必须在循环内等待
  newPipeBefore.push_back(s);
}
 
void MoveSyncState::PlanMoveOutSetSync(
    SyncOps &newPipeAfter, SyncOperation *s,
    const std::pair<unsigned int, unsigned int> pair) {
  
  if (s->GetType() != SyncOperation::TYPE::SET_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) {
    newPipeAfter.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(syncPair.size() > 1, "expected syncPair size > 1");
  auto *waitSync = syncPair[1].get();
 
  // 如果 Wait 操作在 Loop 外部
  // 说明循环内产生的信号，只在循环外被消费
  // 可以将 Set 沉降到 Loop End 之后
  if ((waitSync->GetSyncIRIndex() > pair.second) ||
      (waitSync->GetSyncIRIndex() < pair.first)) {
    
    // [Optimization]: Sink Set out of Loop
    checkSyncIRIndex(syncIR_, pair.second);
    // pair.second 是 LoopEnd 节点
    syncIR_[pair.second]->pipeAfter.push_front(s); 
    s->SetSyncIRIndex(pair.second);
    return;
  }
  
  newPipeAfter.push_back(s);
}
