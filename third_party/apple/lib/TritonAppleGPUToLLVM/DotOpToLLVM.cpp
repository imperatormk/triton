// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
//
// Strategy (tiled TG scatter, register-resident C):
//   Scatter/load 8 rows at a time to minimize TG memory usage.
//   TG buffer = 8 * max(K, N) floats — phases alias the same memory.
//
//   1. For each 8-row strip tm: scatter A[8×K] → TG, barrier, load A[tm][*], barrier
//   2. For each 8-row strip tm: scatter C[8×N] → TG, barrier, load C[tm][*], barrier
//   3. For each 8-row strip tk: scatter B[8×N] → TG, barrier, load B[tk][*], barrier
//      then MMA: C[tm][tn] += A[tm][tk] * B[tk][tn] for all tm,tn
//   4. For each 8-row strip tm: store C[tm][*] → TG, barrier, gather, barrier
//
// For 64×64×64: TG = 8*64 = 512 floats = 2 KB (vs 16 KB untiled).
// Fits within Apple's 32 KB TG limit even with large tiles.
//
// Supports arbitrary M×K × K×N where M,N,K are multiples of 8.
// Handles any blocked encoding (reads sizePerThread/threadsPerWarp/warpsPerCTA).

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
        auto cMmaEnc = dyn_cast<AppleMmaEncodingAttr>(cType.getEncoding());
        if (!cEnc && !cMmaEnc)
            return failure();

        auto aType = cast<RankedTensorType>(op.getA().getType());
        auto bType = cast<RankedTensorType>(op.getB().getType());

        int64_t M = cType.getShape()[0];
        int64_t N = cType.getShape()[1];
        int64_t K = aType.getShape()[1];

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

        Value shape88  = makeI64Vec2(rewriter, loc, 8, 8);
        Value stride18 = makeI64Vec2(rewriter, loc, 1, 8);
        Value zeroOff  = makeI64Vec2(rewriter, loc, 0, 0);
        Value fenceTG  = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        // fenceSG not used in register-resident approach
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

        // Resolve operand: get LLVM values, per-element offsets, and the
        // blocked encoding needed for makeBase.
        //
        // Path 1 (convert_layout → DotOperandEncoding, identity pass-through):
        //   Elements are in source blocked order. Look through the cvt to
        //   get source LLVM values, use source blocked encoding for offsets.
        //
        // Path 2 (local_load → DotOperandEncoding, from optimize_dot_operands):
        //   Elements are in DotOperandEncoding order (LinearLayout-based).
        //   Use DotOperandEncoding for offsets, parent blocked for makeBase.
        //
        // Path 3 (direct blocked encoding):
        //   Elements are in blocked order. Use blocked encoding for offsets.
        auto resolveOperand = [&](Value tritonVal, Value adaptorVal,
                                  RankedTensorType opTy)
            -> std::tuple<SmallVector<Value>,
                          SmallVector<SmallVector<unsigned>>,
                          ttg::BlockedEncodingAttr> {
            // Path 1: convert_layout — look through to source blocked values
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
        auto cOffsets = emitOffsetForLayout(cType.getEncoding(), cType);

        if (!aSrcEnc || !bSrcEnc)
            return failure();

        // Verify element counts match
        if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
            (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
            (int64_t)elemsC.size() != (int64_t)cOffsets.size())
            return failure();

        // ── Compute runtime thread base position ──────────────────────────
        // Compute base (row, col) for this thread within a tensor of shape [rows, cols].
        // Respects the encoding's order field for warp/lane decomposition.
        // Wraps by tileM/tileN to handle redundant threads.
        auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            auto order = enc.getOrder();

            int64_t sM = spt[0], sN = spt[1];
            int64_t tM = tpw[0], tN = tpw[1];
            int64_t wM = wpc[0], wN = wpc[1];
            int64_t tileM = wM * tM * sM;
            int64_t tileN = wN * tN * sN;

            bool colFastest = (order[0] == 1);

            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value tN_val  = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
            Value tMsM    = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
            Value sM_val  = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
            Value tNsN    = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
            Value sN_val  = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

            // Warp decomposition: faster dim uses mod, slower uses div
            Value wR, wC;
            if (colFastest) {
                wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_val);
                wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_val);
            } else {
                Value wM_val = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
                wR = arith::RemUIOp::create(rewriter, loc, warpId, wM_val);
                wC = arith::DivUIOp::create(rewriter, loc, warpId, wM_val);
            }
            // Lane decomposition: faster dim uses mod, slower uses div
            Value lR, lC;
            if (colFastest) {
                lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_val);
                lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_val);
            } else {
                Value tM_val = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
                lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_val);
                lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_val);
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

        // MMA base: lane→(row,col) within 8x8 tile, warp→tile position
        auto makeBaseMma = [&](AppleMmaEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto wpc = enc.getWarpsPerCTA();
            unsigned wN = wpc[1];

            // Lane decomposition: col = laneId & 7, row_group = laneId >> 3
            Value c7    = arith::ConstantIntOp::create(rewriter, loc, 7, 32);
            Value c3    = arith::ConstantIntOp::create(rewriter, loc, 3, 32);
            Value laneCol = arith::AndIOp::create(rewriter, loc, laneId, c7);
            Value laneRow = arith::ShRUIOp::create(rewriter, loc, laneId, c3);

            // Warp decomposition: warpRow = warpId / warpsN, warpCol = warpId % warpsN
            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value c8      = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
            Value warpRow = arith::DivUIOp::create(rewriter, loc, warpId, wN_val);
            Value warpCol = arith::RemUIOp::create(rewriter, loc, warpId, wN_val);

            // baseRow = warpRow * 8 + laneRow, baseCol = warpCol * 8 + laneCol
            Value baseRow = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, warpRow, c8), laneRow);
            Value baseCol = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, warpCol, c8), laneCol);

            return {baseRow, baseCol};
        };

        auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
        auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);

        Value cBaseRow, cBaseCol;
        if (cEnc) {
            std::tie(cBaseRow, cBaseCol) = makeBase(cEnc, M, N);
        } else {
            std::tie(cBaseRow, cBaseCol) = makeBaseMma(cMmaEnc, M, N);
        }

        // ── Create threadgroup global ─────────────────────────────────────
        // Single TG buffer reused across phases (A, C, B scatter + result gather).
        // Size = 8 * max(K, N) + 1 (garbage bin slot).

        unsigned id = getCounter(ctx)++;
        int64_t tgStripSize = 8 * std::max(K, N);
        int64_t tgSize = tgStripSize + 1;
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

        // ── Static strip filtering ────────────────────────────────────────
        // For each operand, compute the baseRow range from encoding params,
        // then for each strip, precompute which element indices are possibly
        // in range, avoiding runtime comparisons for elements that can never
        // match.

        // Compute max possible baseRow for a blocked encoding.
        auto maxBaseRow = [](ttg::BlockedEncodingAttr enc) -> int64_t {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            return wpc[0] * tpw[0] * spt[0] - 1;
        };

        int64_t aMaxBase = maxBaseRow(aSrcEnc);
        int64_t cMaxBase = maxBaseRow(cEnc ? cEnc : aSrcEnc);
        int64_t bMaxBase = maxBaseRow(bSrcEnc);

        // Pre-bucket: for each strip, which element indices could be in it?
        // Element i is in strip [s, s+8) when baseRow + rowOff[i] ∈ [s, s+8),
        // i.e. baseRow ∈ [s - rowOff[i], s - rowOff[i] + 8).
        // Intersect with [0, maxBase] to check feasibility.
        auto bucketElements = [](SmallVector<SmallVector<unsigned>> &offsets,
                                 int64_t maxBase, int64_t numStrips)
            -> SmallVector<SmallVector<size_t>> {
            SmallVector<SmallVector<size_t>> buckets(numStrips);
            for (size_t i = 0; i < offsets.size(); ++i) {
                int64_t rowOff = offsets[i][0];
                for (int64_t s = 0; s < numStrips; ++s) {
                    int64_t stripStart = s * 8;
                    int64_t lo = stripStart - rowOff;
                    int64_t hi = lo + 8;
                    if (lo <= maxBase && hi > 0)
                        buckets[s].push_back(i);
                }
            }
            return buckets;
        };

        auto aBuckets = bucketElements(aOffsets, aMaxBase, tilesM);
        auto cBuckets = bucketElements(cOffsets, cMaxBase, tilesM);
        auto bBuckets = bucketElements(bOffsets, bMaxBase, tilesK);

        // Garbage bin index (last slot in buffer)
        Value garbageIdx = arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 64);

        // Filtered scatter: only processes elements in the bucket for this strip.
        // Still uses garbage bin for runtime edge cases (multiple rowOffs mapping
        // to same strip depending on baseRow).
        auto filteredScatter = [&](Value ptr, Value garbIdx,
                                   Value baseRow, Value baseCol,
                                   SmallVector<Value> &elems,
                                   SmallVector<SmallVector<unsigned>> &offsets,
                                   SmallVector<size_t> &bucket,
                                   int64_t stride, int64_t rowStart) {
            for (size_t idx : bucket) {
                int64_t rowOff = offsets[idx][0];
                int64_t colOff = offsets[idx][1];
                Value actualRow = arith::AddIOp::create(rewriter, loc, baseRow,
                    arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                Value inStrip = arith::AndIOp::create(rewriter, loc,
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
                Value sIdx = stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
                Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, sIdx, garbIdx);
                Value val = elems[idx];
                if (val.getType() != f32Ty)
                    val = arith::ExtFOp::create(rewriter, loc, f32Ty, val);
                Value gep = LLVM::GEPOp::create(rewriter, loc,
                    tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{safeIdx});
                LLVM::StoreOp::create(rewriter, loc, val, gep);
            }
        };

        // ── Phase 1: Load C tiles (filtered strip scatter) ────────────────
        SmallVector<SmallVector<Value>> matC_tiles(tilesM);
        for (int64_t tm = 0; tm < tilesM; ++tm) {
            matC_tiles[tm].resize(tilesN);
            int64_t rowStart = tm * 8;

            filteredScatter(ptrTG, garbageIdx, cBaseRow, cBaseCol,
                            elemsC, cOffsets, cBuckets[tm], N, rowStart);
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

        // ── Phase 2: A/B strips + MMA (A loaded per-strip, not pre-loaded) ──
        // Live matrices: tilesM × tilesN (C) + tilesM (A per strip) + 1 (B)
        // Much less register pressure than pre-loading all A tiles.
        for (int64_t tk = 0; tk < tilesK; ++tk) {
            // Load A tiles for this K-strip
            SmallVector<Value> matA_strip(tilesM);
            {
                int64_t rowStartK = tk * 8;  // K-dim strip for B
                // A strip: scatter A rows that map to this K-column strip.
                // A[M, K]: for the MMA A[tm][tk], we need rows tm*8..(tm+1)*8,
                // cols tk*8..(tk+1)*8. But A is scattered by M-strips.
                // We need to scatter each M-strip of A separately.
                for (int64_t tm = 0; tm < tilesM; ++tm) {
                    int64_t rowStart = tm * 8;
                    filteredScatter(ptrTG, garbageIdx, aBaseRow, aBaseCol,
                                    elemsA, aOffsets, aBuckets[tm], K, rowStart);
                    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                    Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
                    Value aStride = makeI64Vec2(rewriter, loc, 1, K);
                    Value aShape  = makeI64Vec2(rewriter, loc, K, 8);
                    matA_strip[tm] = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrTG, aShape, aStride, aOff}).getResult();
                    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
                }
            }

            // Load B tiles for this K-strip and MMA
            {
                int64_t rowStart = tk * 8;
                filteredScatter(ptrTG, garbageIdx, bBaseRow, bBaseCol,
                                elemsB, bOffsets, bBuckets[tk], N, rowStart);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                Value bStride = makeI64Vec2(rewriter, loc, 1, N);
                Value bShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrTG, bShape, bStride, bOff}).getResult();

                    for (int64_t tm = 0; tm < tilesM; ++tm) {
                        matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, mmaFn,
                            ValueRange{matA_strip[tm], matB, matC_tiles[tm][tn]}).getResult();
                    }
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }
        }

        // ── Phase 4: Store C tiles → TG, gather ─────────────────────────
        auto outElemTy = cType.getElementType();
        SmallVector<Value> resultElems(elemsC.size());
        for (size_t i = 0; i < elemsC.size(); ++i)
            resultElems[i] = arith::ConstantOp::create(rewriter, loc,
                rewriter.getZeroAttr(outElemTy));

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

            for (size_t idx : cBuckets[tm]) {
                int64_t rowOff = cOffsets[idx][0];
                int64_t colOff = cOffsets[idx][1];
                Value actualRow = arith::AddIOp::create(rewriter, loc, cBaseRow,
                    arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                Value inStrip = arith::AndIOp::create(rewriter, loc,
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
                Value sIdx = stripFlatIdx(cBaseRow, cBaseCol, rowOff, colOff, N, rowStart);
                Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, sIdx, garbageIdx);
                Value val = gather1(ptrTG, safeIdx);
                if (val.getType() != outElemTy)
                    val = arith::TruncFOp::create(rewriter, loc, outElemTy, val);
                resultElems[idx] = arith::SelectOp::create(rewriter, loc, inStrip,
                    val, resultElems[idx]);
            }
            LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
        }

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
