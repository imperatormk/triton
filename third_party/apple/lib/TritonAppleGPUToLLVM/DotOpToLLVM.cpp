// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
//
// Strategy (register-resident C):
//   1. All threads scatter A into TG, barrier, pre-load A tiles to simdgroup regs.
//   2. All threads scatter B into TG (reusing A's buffer), barrier.
//   3. Each warp accumulates ONLY its owned C tiles in registers (no TG for C).
//      C starts as zeroinitializer <64 x float>, MMA accumulates in regs.
//   4. Each warp stores its C tiles to TG (no race — tiles are partitioned).
//   5. Barrier, then each thread gathers its C elements back.
//
// TG memory = max(M*K, K*N, M*N) — phases alias the same buffer.
// This fits 64×64 in 16 KB (vs 32+ KB with separate C buffer).
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
        if (!cEnc && !isa<AppleMmaEncodingAttr>(cType.getEncoding()))
            return failure();
        if (!cEnc)
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

        // A and B may come through convert_layout (blocked→dot_op) or directly.
        auto getBlockedVal = [&](Value tritonVal, Value adaptorVal) -> Value {
            if (auto cvtOp = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
                Value mapped = rewriter.getRemappedValue(cvtOp.getSrc());
                if (mapped) return mapped;
            }
            return adaptorVal;
        };

        Value llvmA = getBlockedVal(op.getA(), adaptor.getA());
        Value llvmB = getBlockedVal(op.getB(), adaptor.getB());
        Value llvmC = adaptor.getC();
        if (!llvmA || !llvmB || !llvmC)
            return failure();

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

        auto elemsA = unpack(llvmA);
        auto elemsB = unpack(llvmB);
        auto elemsC = unpack(llvmC);

        // Get the blocked encoding for A and B.
        // If convert_layout exists, use the source encoding; otherwise use directly.
        auto getBlockedEnc = [](Value v) -> ttg::BlockedEncodingAttr {
            if (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>()) {
                auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
                if (srcTy)
                    return dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
            }
            auto ty = dyn_cast<RankedTensorType>(v.getType());
            if (ty) return dyn_cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
            return nullptr;
        };

        auto aSrcEnc = getBlockedEnc(op.getA());
        auto bSrcEnc = getBlockedEnc(op.getB());
        if (!aSrcEnc || !bSrcEnc)
            return failure();

        // ── Compute per-thread element coordinates ────────────────────────
        // Use emitOffsetForLayout (LinearLayout-based) to match the canonical
        // element ordering used by ConvertLayoutOp.  getBlockedElemCoords
        // ignored the encoding's order field, causing mismatches when a
        // convert_layout followed the dot (e.g. trans epilogue).

        // emitOffsetForLayout needs a RankedTensorType with the blocked encoding.
        // aType/bType have DotOperandEncoding, so construct types with the source encoding.
        auto aBlockedTy = RankedTensorType::get(aType.getShape(), aType.getElementType(), aSrcEnc);
        auto bBlockedTy = RankedTensorType::get(bType.getShape(), bType.getElementType(), bSrcEnc);
        auto aOffsets = emitOffsetForLayout(aSrcEnc, aBlockedTy);
        auto bOffsets = emitOffsetForLayout(bSrcEnc, bBlockedTy);
        auto cOffsets = emitOffsetForLayout(cEnc, cType);

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

        auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
        auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
        auto [cBaseRow, cBaseCol] = makeBase(cEnc, M, N);

        // ── Create threadgroup global ─────────────────────────────────────
        // Single TG buffer shared across phases (A/B scatter, then C store).
        // TG = max(M*K, K*N, M*N) — phases alias the same memory.

        unsigned id = getCounter(ctx)++;
        int64_t tgSize = std::max({M * K, K * N, M * N});
        auto tgBuf = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);

        Value ptrTG = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());

        // ── GEP helpers ───────────────────────────────────────────────────

        auto scatter1 = [&](Value ptr, Value val, Value flatIdx64) {
            // MMA operates in f32 — convert non-f32 elements before storing
            if (val.getType() != f32Ty)
                val = arith::ExtFOp::create(rewriter, loc, f32Ty, val);
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            LLVM::StoreOp::create(rewriter, loc, val, gep);
        };

        auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
        };

        // Helper: compute flat index = (baseRow + staticRowOff) * stride + (baseCol + staticColOff)
        auto flatIdx = [&](Value baseRow, Value baseCol,
                           int64_t rowOff, int64_t colOff, int64_t stride) -> Value {
            Value row32 = arith::AddIOp::create(rewriter, loc, baseRow,
                arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
            Value col32 = arith::AddIOp::create(rewriter, loc, baseCol,
                arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
            Value flat32 = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, row32,
                    arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
                col32);
            return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
        };

        // ── Phase 1: Scatter A into TG_AB, barrier, pre-load A tiles ─────
        for (size_t i = 0; i < elemsA.size(); ++i) {
            scatter1(ptrTG, elemsA[i], flatIdx(aBaseRow, aBaseCol, aOffsets[i][0], aOffsets[i][1], K));
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

        // Pre-load all A 8×8 tiles into simdgroup registers before B overwrites.
        int64_t tilesM = M / 8;
        int64_t tilesN = N / 8;
        int64_t tilesK = K / 8;
        (void)(cEnc.getWarpsPerCTA()); // numWarps not needed — all warps compute all tiles

        // matA_tiles[tm][tk] = simdgroup_matrix loaded from TG_AB
        SmallVector<SmallVector<Value>> matA_tiles(tilesM);
        for (int64_t tm = 0; tm < tilesM; ++tm) {
            matA_tiles[tm].resize(tilesK);
            for (int64_t tk = 0; tk < tilesK; ++tk) {
                Value aOff = makeI64Vec2(rewriter, loc, tk * 8, tm * 8);
                Value aStride = makeI64Vec2(rewriter, loc, 1, K);
                Value aShape  = makeI64Vec2(rewriter, loc, K, 8);
                matA_tiles[tm][tk] = LLVM::CallOp::create(rewriter, loc, loadFn,
                    ValueRange{ptrTG, aShape, aStride, aOff}).getResult();
            }
        }

        // ── Phase 2: Scatter C into TG (reuses A's buffer), barrier ──────
        // C input must be in TG so all warps can load their initial accumulators.
        // This overwrites A in TG — that's fine, A is already in simdgroup regs.
        // Barrier ensures all warps finished loading A before any warp writes C.
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
        for (size_t i = 0; i < elemsC.size(); ++i) {
            scatter1(ptrTG, elemsC[i], flatIdx(cBaseRow, cBaseCol, cOffsets[i][0], cOffsets[i][1], N));
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

        // ── Phase 3: Load C tiles to regs, then scatter B into TG ────────
        // Load all C 8×8 tiles into simdgroup registers (register-resident C).
        SmallVector<SmallVector<Value>> matC_tiles(tilesM);
        Value cStride = makeI64Vec2(rewriter, loc, 1, N);
        Value cShape  = makeI64Vec2(rewriter, loc, N, 8);
        for (int64_t tm = 0; tm < tilesM; ++tm) {
            matC_tiles[tm].resize(tilesN);
            for (int64_t tn = 0; tn < tilesN; ++tn) {
                Value cOff = makeI64Vec2(rewriter, loc, tn * 8, tm * 8);
                matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, loadFn,
                    ValueRange{ptrTG, cShape, cStride, cOff}).getResult();
            }
        }

        // Scatter B into TG (overwrites C — fine, C is in regs now).
        // Need TG barrier first so all warps finish loading C before any writes B.
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
        for (size_t i = 0; i < elemsB.size(); ++i) {
            scatter1(ptrTG, elemsB[i], flatIdx(bBaseRow, bBaseCol, bOffsets[i][0], bOffsets[i][1], N));
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

        // ── Phase 4: MMA in registers ────────────────────────────────────
        // All warps compute ALL tiles identically (same A regs, same B in TG,
        // same C regs). MMA is deterministic so all warps get same results.
        // This means all warps can store without races (identical values).
        // Redundant compute but correct, branchless, and no extra TG needed.
        for (int64_t tm = 0; tm < tilesM; ++tm) {
            for (int64_t tn = 0; tn < tilesN; ++tn) {
                for (int64_t tk = 0; tk < tilesK; ++tk) {
                    Value matA = matA_tiles[tm][tk];

                    Value bOff = makeI64Vec2(rewriter, loc, tn * 8, tk * 8);
                    Value bStride = makeI64Vec2(rewriter, loc, 1, N);
                    Value bShape  = makeI64Vec2(rewriter, loc, N, 8);
                    Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrTG, bShape, bStride, bOff}).getResult();

                    matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, mmaFn,
                        ValueRange{matA, matB, matC_tiles[tm][tn]}).getResult();
                }
            }
        }

        // ── Phase 5: Store C from regs → TG, barrier, gather ────────────
        // All warps store all tiles (identical values, no race).
        // B is no longer needed — C overwrites TG.
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
        for (int64_t tm = 0; tm < tilesM; ++tm) {
            for (int64_t tn = 0; tn < tilesN; ++tn) {
                Value cOff = makeI64Vec2(rewriter, loc, tn * 8, tm * 8);
                LLVM::CallOp::create(rewriter, loc, storeFn,
                    ValueRange{matC_tiles[tm][tn], ptrTG, cShape, cStride, cOff});
            }
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

        // ── Gather C elements back ────────────────────────────────────────
        auto outElemTy = cType.getElementType();
        SmallVector<Value> resultElems(elemsC.size());
        for (size_t i = 0; i < elemsC.size(); ++i) {
            Value val = gather1(ptrTG, flatIdx(cBaseRow, cBaseCol, cOffsets[i][0], cOffsets[i][1], N));
            // MMA operates in f32 — truncate back if output type is narrower
            if (val.getType() != outElemTy)
                val = arith::TruncFOp::create(rewriter, loc, outElemTy, val);
            resultElems[i] = val;
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
