// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
//
// Strategy (tiled TG scatter, register-resident C):
//   Scatter/load 8 rows at a time to minimize TG memory usage.
//   TG buffer = 8 * max(K, N) floats -- phases alias the same memory.
//
//   1. For each 8-row strip tm: scatter A[8xK] -> TG, barrier, load A[tm][*], barrier
//   2. For each 8-row strip tm: scatter C[8xN] -> TG, barrier, load C[tm][*], barrier
//   3. For each 8-row strip tk: scatter B[8xN] -> TG, barrier, load B[tk][*], barrier
//      then MMA: C[tm][tn] += A[tm][tk] * B[tk][tn] for all tm,tn
//   4. For each 8-row strip tm: store C[tm][*] -> TG, barrier, gather, barrier
//
// For 64x64x64: TG = 8*64 = 512 floats = 2 KB (vs 16 KB untiled).
// Fits within Apple's 32 KB TG limit even with large tiles.
//
// Supports arbitrary M*K x K*N where M,N,K are multiples of 8.
// Handles any blocked encoding (reads sizePerThread/threadsPerWarp/warpsPerCTA).
// Supports batched (3D+) dot: processes each batch slice independently.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
using namespace mlir::triton::applegpu;

namespace {

static Type getSimdgroupMatrixType(MLIRContext *ctx) {
    return LLVM::getVectorType(Float32Type::get(ctx), 64);
}

static Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val) {
    auto ty  = LLVM::getVectorType(IntegerType::get(b.getContext(), 64), 2);
    Value vec = UndefOp::create(b, loc, ty);
    Value va  = arith::ConstantIntOp::create(b, loc, a,     64);
    Value vb  = arith::ConstantIntOp::create(b, loc, b_val, 64);
    Value i0  = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value i1  = arith::ConstantIntOp::create(b, loc, 1, 32);
    vec = InsertElementOp::create(b, loc, ty, vec, va, i0);
    vec = InsertElementOp::create(b, loc, ty, vec, vb, i1);
    return vec;
}

static LLVMFuncOp getOrInsertIntrinsic(ConversionPatternRewriter &rewriter,
                                        ModuleOp mod,
                                        StringRef name, LLVMFunctionType fnTy) {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                               Linkage::External);
}

static LLVM::GlobalOp getOrCreateTGGlobal(ConversionPatternRewriter &rewriter,
                                            ModuleOp mod,
                                            StringRef name, int64_t size) {
    if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
        return g;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto f32Ty = Float32Type::get(mod.getContext());
    auto arrTy = LLVMArrayType::get(f32Ty, size);
    return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                   /*isConstant=*/false,
                                   LLVM::Linkage::Internal,
                                   name,
                                   /*value=*/Attribute(),
                                   /*alignment=*/4,
                                   /*addrspace=*/3u);
}

// Convert a value to f32. Handles both float and integer element types.
static Value toF32(OpBuilder &rewriter, Location loc, Value val, Type f32Ty) {
    auto valTy = val.getType();
    if (valTy == f32Ty)
        return val;
    if (isa<FloatType>(valTy))
        return arith::ExtFOp::create(rewriter, loc, f32Ty, val);
    // Integer type (e.g. i8, i16, i32) -- use signed conversion
    return arith::SIToFPOp::create(rewriter, loc, f32Ty, val);
}

// Convert f32 to the target element type. Handles both float and integer types.
static Value fromF32(OpBuilder &rewriter, Location loc, Value val, Type targetTy) {
    if (val.getType() == targetTy)
        return val;
    if (isa<FloatType>(targetTy))
        return arith::TruncFOp::create(rewriter, loc, targetTy, val);
    // Integer type -- use signed conversion
    return arith::FPToSIOp::create(rewriter, loc, targetTy, val);
}

struct DotOpAppleMmaConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    static unsigned &getCounter(MLIRContext *ctx) {
        static llvm::DenseMap<MLIRContext *, unsigned> counters;
        return counters[ctx];
    }

    LogicalResult matchAndRewrite(
        tt::DotOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        auto cType = cast<RankedTensorType>(op.getC().getType());
        auto cEnc = dyn_cast<ttg::BlockedEncodingAttr>(cType.getEncoding());
        if (!cEnc && !isa<AppleMmaEncodingAttr>(cType.getEncoding()))
            return failure();
        if (!cEnc)
            return failure();

        auto aType = cast<RankedTensorType>(op.getA().getType());
        auto bType = cast<RankedTensorType>(op.getB().getType());

        // ── Extract M, N, K from last two dimensions ────────────────────
        unsigned rank = cType.getRank();
        unsigned rowDim = rank - 2;
        unsigned colDim = rank - 1;

        int64_t M = cType.getShape()[rowDim];
        int64_t N = cType.getShape()[colDim];
        int64_t K = aType.getShape()[colDim]; // A is [..., M, K]

        // Compute batch size (product of all dims except last 2)
        int64_t batchSize = 1;
        for (unsigned d = 0; d < rowDim; ++d)
            batchSize *= cType.getShape()[d];


        auto f32Ty     = Float32Type::get(ctx);
        auto tgPtrTy   = LLVMPointerType::get(ctx, 3);
        auto matTy     = getSimdgroupMatrixType(ctx);
        auto i32Ty     = IntegerType::get(ctx, 32);
        auto i64Ty     = IntegerType::get(ctx, 64);

        // ── Declare air intrinsics ────────────────────────────────────────

        auto laneIdFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_index_in_simdgroup",
            LLVMFunctionType::get(i32Ty, {}, false));

        auto voidTy = LLVMVoidType::get(ctx);
        auto barrTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
        auto tgBarrFn = getOrInsertIntrinsic(rewriter, mod,
            "air.threadgroup.barrier", barrTy);
        (void)getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup.barrier", barrTy);

        auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
        auto loadFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
            LLVMFunctionType::get(matTy, {tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));
        auto mmaFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
            LLVMFunctionType::get(matTy, {matTy, matTy, matTy}, false));
        auto storeFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
            LLVMFunctionType::get(voidTy, {matTy, tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));

        // ── Constants ────────────────────────────────────────────────────

        Value fenceTG  = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value execMod  = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

        // ── Thread identification ─────────────────────────────────────────

        Value laneId = LLVM::CallOp::create(rewriter, loc, laneIdFn,
                                             ValueRange{}).getResult();

        auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto tidFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_position_in_threadgroup",
            LLVMFunctionType::get(arrI32x3Ty, {}, false));
        Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                ValueRange{}).getResult();
        Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                          tidStruct, ArrayRef<int64_t>{0});
        Value c32    = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
        Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

        // ── Get blocked encoding params for A, B, C ───────────────────────

        // Unpack struct elements
        auto unpack = [&](Value v) -> SmallVector<Value> {
            SmallVector<Value> elems;
            if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
                for (unsigned i = 0; i < sTy.getBody().size(); ++i)
                    elems.push_back(ExtractValueOp::create(rewriter, loc,
                        sTy.getBody()[i], v, ArrayRef<int64_t>{(int64_t)i}));
            } else {
                elems = {v};
            }
            return elems;
        };

        auto resolveOperand = [&](Value tritonVal, Value adaptorVal,
                                  RankedTensorType opTy)
            -> std::tuple<SmallVector<Value>,
                          SmallVector<SmallVector<unsigned>>,
                          ttg::BlockedEncodingAttr> {
            // Path 1: convert_layout -- look through to source blocked values
            if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
                Value mapped = rewriter.getRemappedValue(cvt.getSrc());
                if (mapped) {
                    auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
                    auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
                    if (srcEnc) {
                        auto offsets = emitOffsetForLayout(srcEnc, srcTy);
                        return {unpack(mapped), offsets, srcEnc};
                    }
                }
            }
            // Path 2: DotOperandEncoding (e.g. local_load after optimize_dot_operands)
            auto enc = opTy.getEncoding();
            if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(enc)) {
                auto parentEnc = dyn_cast<ttg::BlockedEncodingAttr>(dotEnc.getParent());
                if (parentEnc) {
                    auto offsets = emitOffsetForLayout(enc, opTy);
                    return {unpack(adaptorVal), offsets, parentEnc};
                }
            }
            // Path 3: direct blocked encoding
            if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(enc)) {
                auto offsets = emitOffsetForLayout(blk, opTy);
                return {unpack(adaptorVal), offsets, blk};
            }
            return {{}, {}, nullptr};
        };

        auto [elemsA, aOffsets, aSrcEnc] = resolveOperand(op.getA(), adaptor.getA(), aType);
        auto [elemsB, bOffsets, bSrcEnc] = resolveOperand(op.getB(), adaptor.getB(), bType);
        auto elemsC = unpack(adaptor.getC());
        auto cOffsets = emitOffsetForLayout(cEnc, cType);

        if (!aSrcEnc || !bSrcEnc)
            return failure();

        // Verify element counts match
        if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
            (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
            (int64_t)elemsC.size() != (int64_t)cOffsets.size())
            return failure();

        // ── Compute runtime thread base position ──────────────────────────
        // For 3D+ tensors, use only the last two dims of the encoding for
        // the MMA row/col base. The batch dims are handled via compile-time
        // offset matching.
        auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            auto order = enc.getOrder();

            unsigned encRank = spt.size();
            unsigned encRowDim = encRank - 2;
            unsigned encColDim = encRank - 1;

            int64_t sM = spt[encRowDim], sN = spt[encColDim];
            int64_t tM = tpw[encRowDim], tN = tpw[encColDim];
            int64_t wM = wpc[encRowDim], wN = wpc[encColDim];
            int64_t tileM = wM * tM * sM;
            int64_t tileN = wN * tN * sN;

            // For 3D+ encodings, strip batch warp component from warpId.
            // Batch dims use warps for batch distribution; the 2D row/col
            // decomposition should only use the row/col warps.
            int64_t batchWarps = 1;
            for (unsigned d = 0; d < encRowDim; ++d)
                batchWarps *= wpc[d];
            Value matWarpId = warpId;
            if (batchWarps > 1) {
                Value bw = arith::ConstantIntOp::create(rewriter, loc, batchWarps, 32);
                matWarpId = arith::RemUIOp::create(rewriter, loc, warpId, bw);
                // Actually we need warpId within the 2D tile, not within batch.
                // The warp decomposition maps warpId → (batch_warp, mat_warp).
                // mat_warp = warpId % (wM * wN), batch_warp = warpId / (wM * wN)
                int64_t matWarps = wM * wN;
                Value mw = arith::ConstantIntOp::create(rewriter, loc, matWarps, 32);
                matWarpId = arith::RemUIOp::create(rewriter, loc, warpId, mw);
            }

            // Similarly strip batch lanes from laneId
            int64_t batchLanes = 1;
            for (unsigned d = 0; d < encRowDim; ++d)
                batchLanes *= tpw[d];
            Value matLaneId = laneId;
            if (batchLanes > 1) {
                int64_t matLanes = tM * tN;
                Value ml = arith::ConstantIntOp::create(rewriter, loc, matLanes, 32);
                matLaneId = arith::RemUIOp::create(rewriter, loc, laneId, ml);
            }

            // For the last two dims, check if col is the fastest-varying dim.
            // order[0] is the fastest dim index. For 2D: order[0]==1 means col-fast.
            // For 3D with order=[2,1,0]: order[0]==2 means colDim is fastest.
            bool colFastest = (order[0] == (unsigned)encColDim);

            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value tN_val  = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
            Value tMsM    = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
            Value sM_val  = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
            Value tNsN    = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
            Value sN_val  = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

            // Warp decomposition: faster dim uses mod, slower uses div
            Value wR, wC;
            if (colFastest) {
                wR = arith::DivUIOp::create(rewriter, loc, matWarpId, wN_val);
                wC = arith::RemUIOp::create(rewriter, loc, matWarpId, wN_val);
            } else {
                Value wM_val = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
                wR = arith::RemUIOp::create(rewriter, loc, matWarpId, wM_val);
                wC = arith::DivUIOp::create(rewriter, loc, matWarpId, wM_val);
            }
            // Lane decomposition: faster dim uses mod, slower uses div
            Value lR, lC;
            if (colFastest) {
                lR = arith::DivUIOp::create(rewriter, loc, matLaneId, tN_val);
                lC = arith::RemUIOp::create(rewriter, loc, matLaneId, tN_val);
            } else {
                Value tM_val = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
                lR = arith::RemUIOp::create(rewriter, loc, matLaneId, tM_val);
                lC = arith::DivUIOp::create(rewriter, loc, matLaneId, tM_val);
            }

            Value baseRow = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wR, tMsM),
                arith::MulIOp::create(rewriter, loc, lR, sM_val));
            Value baseCol = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wC, tNsN),
                arith::MulIOp::create(rewriter, loc, lC, sN_val));

            // Wrap to handle redundant threads (tileM > rows)
            if (tileM > rows) {
                Value modRow = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
                baseRow = arith::RemUIOp::create(rewriter, loc, baseRow, modRow);
            }
            if (tileN > cols) {
                Value modCol = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
                baseCol = arith::RemUIOp::create(rewriter, loc, baseCol, modCol);
            }

            return {baseRow, baseCol};
        };

        auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
        auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
        auto [cBaseRow, cBaseCol] = makeBase(cEnc, M, N);

        // ── Create threadgroup global ─────────────────────────────────────
        unsigned id = getCounter(ctx)++;
        int64_t tgStripSize = 8 * std::max(K, N);
        int64_t tgSize = tgStripSize + 1;  // +1 garbage slot
        auto tgBuf = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);

        Value ptrTG = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());

        // ── GEP helpers ───────────────────────────────────────────────────

        auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
        };

        // stripFlatIdx: (baseRow + rowOff - stripRowStart) * stride + (baseCol + colOff)
        auto stripFlatIdx = [&](Value baseRow, Value baseCol,
                                int64_t rowOff, int64_t colOff,
                                int64_t stride, int64_t stripRowStart) -> Value {
            Value row32 = arith::AddIOp::create(rewriter, loc, baseRow,
                arith::ConstantIntOp::create(rewriter, loc, rowOff - stripRowStart, 32));
            Value col32 = arith::AddIOp::create(rewriter, loc, baseCol,
                arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
            Value flat32 = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, row32,
                    arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
                col32);
            return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
        };

        int64_t tilesM = M / 8;
        int64_t tilesN = N / 8;
        int64_t tilesK = K / 8;

        // Garbage bin index -- last slot in TG, used for out-of-strip stores.
        Value garbageIdx = arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 64);

        // ── Collect unique batch indices from offsets ─────────────────────
        // For 2D tensors, there is one batch "slice" with empty batch offset.
        // For 3D+ tensors, group elements by their batch offset (all dims except last 2).
        using BatchKey = SmallVector<unsigned>;
        SmallVector<BatchKey> uniqueBatches;
        auto getBatchKey = [&](const SmallVector<unsigned> &offset) -> BatchKey {
            BatchKey key;
            for (unsigned d = 0; d < rowDim; ++d)
                key.push_back(offset[d]);
            return key;
        };

        // Collect unique batch keys from C offsets (C covers all output batches)
        {
            llvm::DenseSet<unsigned> seen;
            for (size_t i = 0; i < cOffsets.size(); ++i) {
                auto key = getBatchKey(cOffsets[i]);
                // Use a simple hash for the batch key
                unsigned hash = 0;
                for (unsigned d = 0; d < key.size(); ++d)
                    hash = hash * 137 + key[d];
                if (seen.insert(hash).second)
                    uniqueBatches.push_back(key);
            }
        }

        // If no batch dims (2D), add a single empty batch key
        if (uniqueBatches.empty())
            uniqueBatches.push_back({});

        auto matchesBatch = [&](const SmallVector<unsigned> &offset,
                                const BatchKey &batch) -> bool {
            for (unsigned d = 0; d < batch.size(); ++d)
                if (offset[d] != batch[d])
                    return false;
            return true;
        };

        // Helper: scatter elements matching a batch key into TG for an 8-row strip.
        auto batchStripScatter = [&](Value baseRow, Value baseCol,
                                     SmallVector<Value> &elems,
                                     SmallVector<SmallVector<unsigned>> &offsets,
                                     int64_t stride, int64_t rowStart,
                                     const BatchKey &batch) {
            for (size_t i = 0; i < elems.size(); ++i) {
                if (!matchesBatch(offsets[i], batch))
                    continue;
                int64_t rowOff = offsets[i][rowDim];
                int64_t colOff = offsets[i][colDim];
                Value actualRow = arith::AddIOp::create(rewriter, loc, baseRow,
                    arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                Value inStrip = arith::AndIOp::create(rewriter, loc,
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
                Value idx = stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
                Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, idx, garbageIdx);
                Value val = toF32(rewriter, loc, elems[i], f32Ty);
                Value gep = LLVM::GEPOp::create(rewriter, loc,
                    tgPtrTy, f32Ty, ptrTG, ArrayRef<LLVM::GEPArg>{safeIdx});
                LLVM::StoreOp::create(rewriter, loc, val, gep);
            }
        };

        // ── Initialize result to zero ────────────────────────────────────
        auto outElemTy = cType.getElementType();
        SmallVector<Value> resultElems(elemsC.size());
        for (size_t i = 0; i < elemsC.size(); ++i)
            resultElems[i] = arith::ConstantOp::create(rewriter, loc,
                rewriter.getZeroAttr(outElemTy));

        // ── Process each batch slice independently ───────────────────────
        for (const auto &batch : uniqueBatches) {

            // Phase 1: Load A tiles (8-row strips)
            SmallVector<SmallVector<Value>> matA_tiles(tilesM);
            for (int64_t tm = 0; tm < tilesM; ++tm) {
                matA_tiles[tm].resize(tilesK);
                int64_t rowStart = tm * 8;

                batchStripScatter(aBaseRow, aBaseCol, elemsA, aOffsets, K, rowStart, batch);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                for (int64_t tk = 0; tk < tilesK; ++tk) {
                    Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
                    Value aStride = makeI64Vec2(rewriter, loc, 1, K);
                    Value aShape  = makeI64Vec2(rewriter, loc, K, 8);
                    matA_tiles[tm][tk] = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrTG, aShape, aStride, aOff}).getResult();
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }

            // Phase 2: Load C tiles (8-row strips)
            SmallVector<SmallVector<Value>> matC_tiles(tilesM);
            for (int64_t tm = 0; tm < tilesM; ++tm) {
                matC_tiles[tm].resize(tilesN);
                int64_t rowStart = tm * 8;

                batchStripScatter(cBaseRow, cBaseCol, elemsC, cOffsets, N, rowStart, batch);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                Value cStride = makeI64Vec2(rewriter, loc, 1, N);
                Value cShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrTG, cShape, cStride, cOff}).getResult();
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }

            // Phase 3: B strips + MMA
            for (int64_t tk = 0; tk < tilesK; ++tk) {
                int64_t rowStart = tk * 8;

                batchStripScatter(bBaseRow, bBaseCol, elemsB, bOffsets, N, rowStart, batch);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                Value bStride = makeI64Vec2(rewriter, loc, 1, N);
                Value bShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrTG, bShape, bStride, bOff}).getResult();

                    for (int64_t tm = 0; tm < tilesM; ++tm) {
                        matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, mmaFn,
                            ValueRange{matA_tiles[tm][tk], matB, matC_tiles[tm][tn]}).getResult();
                    }
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }

            // Phase 4: Store C tiles -> TG (8-row strips), gather
            for (int64_t tm = 0; tm < tilesM; ++tm) {
                int64_t rowStart = tm * 8;

                Value cStoreStride = makeI64Vec2(rewriter, loc, 1, N);
                Value cStoreShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    LLVM::CallOp::create(rewriter, loc, storeFn,
                        ValueRange{matC_tiles[tm][tn], ptrTG, cStoreShape, cStoreStride, cOff});
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                // Gather: each thread reads its C elements that fall in this strip and batch.
                for (size_t i = 0; i < elemsC.size(); ++i) {
                    if (!matchesBatch(cOffsets[i], batch))
                        continue;
                    int64_t rowOff = cOffsets[i][rowDim];
                    int64_t colOff = cOffsets[i][colDim];
                    Value actualRow = arith::AddIOp::create(rewriter, loc, cBaseRow,
                        arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                    Value inStrip = arith::AndIOp::create(rewriter, loc,
                        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                            actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                            actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
                    Value idx = stripFlatIdx(cBaseRow, cBaseCol, rowOff, colOff, N, rowStart);
                    Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, idx, garbageIdx);
                    Value val = gather1(ptrTG, safeIdx);
                    if (val.getType() != outElemTy)
                        val = fromF32(rewriter, loc, val, outElemTy);
                    resultElems[i] = arith::SelectOp::create(rewriter, loc, inStrip,
                        val, resultElems[i]);
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }
        } // end batch loop

        // ── Pack result ───────────────────────────────────────────────────
        auto outLLVMTy = getTypeConverter()->convertType(cType);
        if (!outLLVMTy) return failure();

        if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
            Value result = UndefOp::create(rewriter, loc, outStructTy);
            for (size_t i = 0; i < resultElems.size(); ++i)
                result = InsertValueOp::create(rewriter, loc, outStructTy,
                             result, resultElems[i],
                             ArrayRef<int64_t>{(int64_t)i});
            rewriter.replaceOp(op, result);
        } else {
            rewriter.replaceOp(op, resultElems[0]);
        }
        return success();
    }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit) {
    patterns.add<DotOpAppleMmaConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
