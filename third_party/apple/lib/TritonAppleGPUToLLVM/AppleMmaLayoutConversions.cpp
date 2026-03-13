// AppleMmaEncodingAttr::toLinearLayout
//
// Converts Apple simdgroup MMA encoding → LinearLayout used throughout
// the Triton compiler for layout propagation, conversion insertion,
// and shared memory access analysis.
//
// Verified layout (verify_simdgroup.metal, M1 hardware):
//   lane T, reg R → row = (T >> 3) + R*4,  col = T & 7

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#define S(v) StringAttr::get(ctx, (v))

namespace mlir::triton::applegpu {

LinearLayout AppleMmaEncodingAttr::toLinearLayout(
    llvm::ArrayRef<int64_t> shape) const {

    MLIRContext *ctx = getContext();
    int rank = shape.size();
    assert(rank == 2 && "AppleMmaEncoding only supports 2D tensors for now");

    auto dimNames = standardOutDimNames(ctx, rank);
    auto dimRow = dimNames[0];  // "dim0"
    auto dimCol = dimNames[1];  // "dim1"

    // ── Single 8×8 simdgroup tile ─────────────────────────────────────────
    //
    // Hardware mapping: lane[0:2]→col, lane[3:4]→row, reg[0]→row+4
    //
    // Convention: "register" must be the first in-dim (asserted by
    // ensureLayoutNotSmallerThan). We construct via explicit bases:
    //
    //   register bit 0 → row bit 2  (rows 0-3 vs 4-7)
    //   lane bit 0     → col bit 0
    //   lane bit 1     → col bit 1
    //   lane bit 2     → col bit 2
    //   lane bit 3     → row bit 0
    //   lane bit 4     → row bit 1
    std::vector<std::vector<int32_t>> registerBases = {
        {4, 0}  // reg bit 0 → row=4, col=0
    };
    std::vector<std::vector<int32_t>> laneBases = {
        {0, 1},  // lane bit 0 → row=0, col=1
        {0, 2},  // lane bit 1 → row=0, col=2
        {0, 4},  // lane bit 2 → row=0, col=4
        {1, 0},  // lane bit 3 → row=1, col=0
        {2, 0},  // lane bit 4 → row=2, col=0
    };
    LinearLayout ctaLayout(
        SmallVector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>{
            {S("register"), registerBases}, {S("lane"), laneBases}},
        SmallVector<StringAttr>{dimRow, dimCol});

    // ── Tile simdgroups across M and N ────────────────────────────────────
    auto wpc = getWarpsPerCTA();
    assert(wpc.size() == 2);

    // Column-major warp tiling: warpId % wN → col, warpId / wN → row
    SmallVector<unsigned> warpOrder{1, 0};
    ctaLayout *=
        identityStandardND(S("warp"), wpc, warpOrder)
            .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

    // ── Broadcast to full tensor shape ────────────────────────────────────
    // Handles shapes larger than one CTA tile by repeating the pattern.
    // Apple has no CGA — use trivial 1-CTA layout.
    return combineCtaCgaWithShape(ctaLayout, getCGALayout(), shape);
}

} // namespace mlir::triton::applegpu
