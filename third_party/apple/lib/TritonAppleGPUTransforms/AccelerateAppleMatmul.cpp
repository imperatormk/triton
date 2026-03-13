// AccelerateAppleMatmul: rewrite tt.dot ops to use AppleMmaEncodingAttr
//
// This is the Triton GPU lowering pass that replaces the generic
// BlockedEncoding on dot ops with AppleMmaEncoding, enabling
// simdgroup_multiply_accumulate code generation.
//
// Mirrors AccelerateAMDMatmul.cpp (BlockedToMFMA) for Apple.
//
// Pipeline position:
//   make_ttgir: TritonGPU IR → Apple MMA tiled TritonGPU IR
//
// What it does:
//   1. Find all tt.dot ops
//   2. Check that element types are supported (f16, bf16, f32)
//   3. Replace output encoding: BlockedEncoding → AppleMmaEncoding
//   4. Insert ConvertLayoutOp on operands to match expected input layout
//   5. Insert ConvertLayoutOp on output to convert back to user's expected layout

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_ACCELERATEAPPLEMATMUL
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace {

// Determine warpsPerCTA for a given dot op shape and total warp count.
// Apple simdgroup tile = 8x8, so:
//   warpsM = ceil(M / 8), warpsN = ceil(N / 8), capped by numWarps.
SmallVector<unsigned> warpsPerTileApple(int64_t M, int64_t N, int numWarps) {
    unsigned warpsM = std::max(1LL, M / 8);
    unsigned warpsN = std::max(1LL, N / 8);

    // Clamp to numWarps budget (prefer square allocation)
    while (warpsM * warpsN > (unsigned)numWarps) {
        if (warpsM > warpsN) warpsM /= 2;
        else warpsN /= 2;
    }
    return {warpsM, warpsN};
}

// Check if element type is supported by simdgroup_multiply_accumulate
bool isSupportedDotType(mlir::Type elemTy) {
    return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32();
}

// Pattern: BlockedEncoding tt.dot → AppleMmaEncoding tt.dot
struct BlockedToAppleMma : public OpRewritePattern<tt::DotOp> {
    int numWarps;

    BlockedToAppleMma(MLIRContext *ctx, int numWarps, PatternBenefit benefit = 1)
        : OpRewritePattern(ctx, benefit), numWarps(numWarps) {}

    LogicalResult matchAndRewrite(tt::DotOp dot,
                                  PatternRewriter &rewriter) const override {
        auto ctx = dot.getContext();
        auto cType = cast<RankedTensorType>(dot.getC().getType());
        auto aType = cast<RankedTensorType>(dot.getA().getType());

        // Already converted — skip
        if (isa<AppleMmaEncodingAttr>(cType.getEncoding()))
            return failure();

        // Check supported element types
        if (!isSupportedDotType(aType.getElementType()))
            return failure();

        auto shape = cType.getShape();
        if (shape.size() != 2) return failure();

        int64_t M = shape[0], N = shape[1];

        // Skip if tile doesn't divide evenly into 8×8 simdgroup tiles,
        // or if too small for meaningful warp tiling.
        if (M % 8 != 0 || N % 8 != 0) return failure();
        if (M < 16 || N < 16) return failure();

        // Create AppleMmaEncoding for the result only.
        // A and B keep their blocked encoding — DotOpToLLVM handles the
        // mismatch by scattering blocked inputs through TG while producing
        // MMA-encoded output. Only one ConvertLayoutOp (result → blocked)
        // is needed downstream, vs 4 if we converted all operands.
        auto wpc = warpsPerTileApple(M, N, numWarps);

        int64_t K = aType.getShape()[1];

        // Guard: core Triton's ConvertLayoutOp(mma→blocked) via shared memory
        // produces wrong results for AppleMma when:
        // - wpc is non-square (e.g. [2,1] from num_warps=2), or
        // - K > 32 (large K dimension causes data corruption)
        // Root cause is in transferSwizzlingLocalMemImpl's handling of our
        // LinearLayout. Works correctly for square wpc with K ≤ 32.
        // TODO: root-cause the core conversion issue for these cases.
        if (wpc[0] != wpc[1]) return failure();
        if (K > 32) return failure();
        auto mmaEnc = AppleMmaEncodingAttr::get(ctx, wpc);

        auto newCType = RankedTensorType::get(shape, cType.getElementType(), mmaEnc);

        auto loc = dot.getLoc();

        // If A/B have DotOperandEncoding, strip it back to plain blocked.
        // DotOperandEncoding's parent must match the result encoding, but
        // we're changing the result to AppleMma while keeping A/B blocked.
        auto stripDotOpEnc = [&](Value operand) -> Value {
            auto ty = cast<RankedTensorType>(operand.getType());
            if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(ty.getEncoding())) {
                auto parentTy = RankedTensorType::get(ty.getShape(),
                    ty.getElementType(), dotEnc.getParent());
                return ttg::ConvertLayoutOp::create(rewriter, loc, parentTy, operand);
            }
            return operand;
        };
        Value newA = stripDotOpEnc(dot.getA());
        Value newB = stripDotOpEnc(dot.getB());

        // Convert C to MMA encoding (it's the accumulator)
        Value newC = ttg::ConvertLayoutOp::create(rewriter, loc, newCType, dot.getC());

        // Create new dot: blocked A, blocked B, AppleMma C → AppleMma result
        auto newDot = tt::DotOp::create(rewriter,
            loc, newCType, newA, newB, newC,
            dot.getInputPrecisionAttr(), dot.getMaxNumImpreciseAccAttr());

        // Convert result back to original blocked encoding for stores
        auto result = ttg::ConvertLayoutOp::create(rewriter,
            loc, cType, newDot.getResult());

        rewriter.replaceOp(dot, result.getResult());
        return success();
    }
};

// The pass
struct AccelerateAppleMatmul
    : public ::impl::AccelerateAppleMatmulBase<AccelerateAppleMatmul> {

    void runOnOperation() override {
        auto mod = getOperation();

        // Get numWarps from module attribute
        int numWarps = ttg::lookupNumWarps(mod);

        RewritePatternSet patterns(&getContext());
        patterns.add<BlockedToAppleMma>(&getContext(), numWarps);

        if (failed(applyPatternsGreedily(mod, std::move(patterns))))
            signalPassFailure();
    }
};

} // anonymous namespace

namespace mlir::triton::applegpu {
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass() {
    return ::createAccelerateAppleMatmul();
}
} // namespace mlir::triton::applegpu
