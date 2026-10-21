/**
 * @file tempo_ops.td
 * @brief Provides the tempo dialect, which states a kernel as machine operations over virtual registers and VMEM
 * buffers, leaving registers, rows, loops and timing to the passes that lower it into tempoisa.
 *
 * @details Values replace register numbers: a !tempo.vreg is a vector register's contents and an i32 is a scalar
 * register's, so a register is live exactly as long as its value is used. A !tempo.vmem is a VMEM region whose base
 * row the allocator chooses. Loops are scf.for, and scalar arithmetic is arith, so pointer advances and loop-carried
 * accumulators are ordinary SSA. A tempo.reg attribute on the operation defining a value names its physical
 * register, written by the register allocator or by hand to pin it, and a base attribute pins a buffer's row.
 */

#ifndef TEMPO_TEMPO_OPS_TD
#define TEMPO_TEMPO_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tempo/tempoisa/isa.td"

/// @brief Provides the dialect a kernel is stated in before its registers, rows and timing are chosen.
def Tempo_Dialect : Dialect
{
    let name = "tempo";
    let cppNamespace = "::mlir::tempo";
    let summary = "MiniTPU machine operations over virtual registers and VMEM buffers";
    let useDefaultTypePrinterParser = 1;
    let hasOperationAttrVerify = 1;
    let extraClassDeclaration = [{
        /// Returns the name of the attribute naming the physical register a value lives in.
        static ::llvm::StringRef get_register_attribute_name()
        {
            return "tempo.reg";
        }
    }];
}

/// @brief Provides the base class of every tempo type.
class Tempo_Type<string name, string type_mnemonic> : TypeDef<Tempo_Dialect, name>
{
    let mnemonic = type_mnemonic;
}

/// @brief Provides a vector register's contents: 64 BF16 elements, four sublanes of sixteen lanes.
def Tempo_VRegType : Tempo_Type<"VReg", "vreg">
{
    let summary = "A vector register's contents";
}

/// @brief Provides a VMEM region of a fixed row count, whose base row the allocator chooses.
def Tempo_VmemType : Tempo_Type<"Vmem", "vmem">
{
    let summary = "A VMEM region of a fixed row count";
    let parameters = (ins "uint32_t":$rows);
    let assemblyFormat = "`<` $rows `>`";
}

/// Constrains an attribute to the values a field of the given range holds.
class Tempo_Field<int low, int high> : ConfinedAttr<I32Attr, [IntMinValue<low>, IntMaxValue<high>]>;

/**
 * @brief Stores the position of a mnemonic within an operation group, or -1 when the group does not name it.
 *
 * @param group Names the group to search.
 * @param mnemonic Names the mnemonic to find.
 */
class Tempo_IndexIn<OpGroup group, string mnemonic>
{
    /// Stores the mnemonic's position.
    int value = !foldl(-1, !range(group.mnemonics), found, index,
                       !if(!eq(group.mnemonics[index], mnemonic), index, found));
}

/// @brief Provides the base class of every tempo operation.
class Tempo_Op<string mnemonic, list<Trait> traits = []> : Op<Tempo_Dialect, mnemonic, traits>;

/**
 * @brief Provides an operation the machine runs as the instruction of the same mnemonic, which tempoisa/isa.td
 * must name in `group`.
 *
 * @param group Names the operation group the instruction belongs to.
 * @param mnemonic Names the instruction.
 */
class Tempo_InstructionOp<OpGroup group, string mnemonic, list<Trait> traits = []> : Tempo_Op<mnemonic, traits>
{
    defvar index = Tempo_IndexIn<group, mnemonic>.value;
    assert !ge(index, 0), "tempoisa/isa.td's " # group.slot # " group names no '" # mnemonic # "'.";
    let summary = "`" # mnemonic # "`";
    let description = group.issue_requirement;
}

/// @brief Provides an elementwise operation over two vector registers.
class Tempo_BinaryOp<OpGroup group, string mnemonic> : Tempo_InstructionOp<group, mnemonic, [Pure]>
{
    let arguments = (ins Tempo_VRegType:$lhs, Tempo_VRegType:$rhs);
    let results = (outs Tempo_VRegType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

/// @brief Provides an operation over one vector register.
class Tempo_UnaryOp<OpGroup group, string mnemonic> : Tempo_InstructionOp<group, mnemonic, [Pure]>
{
    let arguments = (ins Tempo_VRegType:$operand);
    let results = (outs Tempo_VRegType:$result);
    let assemblyFormat = "$operand attr-dict";
}

foreach index = !range(VectorAlu.mnemonics) in
    def Tempo_ # !toupper(!substr(VectorAlu.mnemonics[index], 0, 1)) # !substr(VectorAlu.mnemonics[index], 1) # Op
        : Tempo_BinaryOp<VectorAlu, VectorAlu.mnemonics[index]>;

foreach group = [VectorMove, VectorSfu, VectorReduce] in
    foreach index = !range(group.mnemonics) in
        def Tempo_ # !toupper(!substr(group.mnemonics[index], 0, 1)) # !substr(group.mnemonics[index], 1) # Op
            : Tempo_UnaryOp<group, group.mnemonics[index]>;

/// @brief Provides vtxin, which stages a vector register into a quarter of the transpose tile.
def Tempo_VtxinOp : Tempo_InstructionOp<TransposeIn, "vtxin">
{
    let arguments = (ins Tempo_VRegType:$operand, Tempo_Field<0, 3>:$quarter);
    let assemblyFormat = "$operand `,` `quarter` $quarter attr-dict";
}

/// @brief Provides vtxout, which drains a quarter of the transpose tile into a vector register.
def Tempo_VtxoutOp : Tempo_InstructionOp<TransposeOut, "vtxout">
{
    let arguments = (ins Tempo_Field<0, 3>:$quarter);
    let results = (outs Tempo_VRegType:$result);
    let assemblyFormat = "`quarter` $quarter attr-dict";
}

/**
 * @brief Provides vld, which loads the four rows starting at `row` of a VMEM buffer into a vector register.
 *
 * @details The row is an index the loop lowering resolves to the instruction's static row plus an induction term: a
 * constant, or a constant plus an enclosing scf.for's induction variable shifted left by a power of two.
 */
def Tempo_VldOp : Tempo_InstructionOp<VectorLoad, "vld", [MemoryEffects<[MemRead]>]>
{
    let arguments = (ins Tempo_VmemType:$buffer, Index:$row, DefaultValuedAttr<Tempo_Field<1, 15>, "15">:$mask);
    let results = (outs Tempo_VRegType:$result);
    let assemblyFormat = "$buffer `[` $row `]` (`mask` $mask^)? attr-dict `:` qualified(type($buffer))";
}

/// @brief Provides vst, which stores a vector register into the four rows starting at `row` of a VMEM buffer.
def Tempo_VstOp : Tempo_InstructionOp<VectorStore, "vst", [MemoryEffects<[MemWrite]>]>
{
    let arguments = (ins Tempo_VRegType:$value, Tempo_VmemType:$buffer, Index:$row,
                         DefaultValuedAttr<Tempo_Field<1, 15>, "15">:$mask);
    let assemblyFormat = "$value `,` $buffer `[` $row `]` (`mask` $mask^)? attr-dict `:` qualified(type($buffer))";
}

/**
 * @brief Provides vmatload, which loads a 16x16 weight tile from four vector registers into the array.
 *
 * @details The instruction reads four consecutive registers from a base, so the allocator places the four operands
 * in consecutive registers.
 */
def Tempo_VmatloadOp : Tempo_InstructionOp<MatrixLoad, "vmatload", [MemoryEffects<[MemWrite]>]>
{
    let arguments = (ins Tempo_VRegType:$quarter0, Tempo_VRegType:$quarter1, Tempo_VRegType:$quarter2,
                         Tempo_VRegType:$quarter3);
    let assemblyFormat = "$quarter0 `,` $quarter1 `,` $quarter2 `,` $quarter3 attr-dict";
}

/// @brief Provides vmatpush, which streams one vector register of activations through the array.
def Tempo_VmatpushOp : Tempo_InstructionOp<MatrixPush, "vmatpush", [MemoryEffects<[MemWrite]>]>
{
    let arguments = (ins Tempo_VRegType:$operand);
    let assemblyFormat = "$operand attr-dict";
}

/// @brief Provides vmatpop, which drains the oldest push's results into a vector register.
def Tempo_VmatpopOp : Tempo_InstructionOp<MatrixPop, "vmatpop", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let results = (outs Tempo_VRegType:$result);
    let assemblyFormat = "attr-dict";
}

/// @brief Provides a launch argument, which the machine reads into a scalar register.
def Tempo_ArgOp : Tempo_Op<"arg", [Pure]>
{
    let summary = "A launch argument, read into a scalar register";
    let arguments = (ins Tempo_Field<0, 3>:$index);
    let results = (outs I32:$result);
    let assemblyFormat = "$index attr-dict";
}

/// @brief Provides a DMA descriptor moving rows between host memory and a VMEM buffer.
class Tempo_DmaOp<string mnemonic, list<Trait> traits> : Tempo_InstructionOp<Descriptor, mnemonic, traits>
{
    let arguments = (ins Tempo_VmemType:$buffer, Index:$row, I32:$base, I32:$stride, Tempo_Field<1, 128>:$rows,
                         Tempo_Field<0, 1>:$dma_slot);
    let assemblyFormat = [{
        $buffer `[` $row `]` `,` `base` $base `,` `stride` $stride `,` `rows` $rows `,` `slot` $dma_slot attr-dict
        `:` qualified(type($buffer))
    }];
    let hasVerifier = 1;
}

def Tempo_VmemldOp : Tempo_DmaOp<"vmemld", [MemoryEffects<[MemRead, MemWrite]>]>;
def Tempo_VmemstOp : Tempo_DmaOp<"vmemst", [MemoryEffects<[MemRead, MemWrite]>]>;

/// @brief Provides flush_slot, which blocks until every DMA slot its mask names reports done.
def Tempo_FlushSlotOp : Tempo_InstructionOp<Control, "flush_slot", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let arguments = (ins Tempo_Field<1, 3>:$mask);
    let assemblyFormat = "$mask attr-dict";
}

/// @brief Provides vmemswap, which exchanges the VMEM slots compute and DMA address.
def Tempo_VmemswapOp : Tempo_InstructionOp<Control, "vmemswap", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let assemblyFormat = "attr-dict";
}

/// @brief Provides a VMEM buffer, whose base row the allocator chooses unless `base` pins it.
def Tempo_AllocOp : Tempo_Op<"alloc", [MemoryEffects<[MemAlloc]>]>
{
    let summary = "A VMEM region";
    let arguments = (ins OptionalAttr<Tempo_Field<0, 16383>>:$base);
    let results = (outs Tempo_VmemType:$buffer);
    let assemblyFormat = "attr-dict `:` qualified(type($buffer))";
}

/**
 * @brief Provides a product of one block of tokens by a weight matrix, staged from and drained to host memory.
 *
 * @details The tile op names what is computed and the host layout it is computed over, leaving the instructions, the
 * buffers and the loops to -tempo-tile-to-vector. The activations are packed as `[depth / 16][tokens][16]`, the
 * weights as `[columns / 16][depth][16]`, and the drain buffer holds a zeroed 16x16 tile followed by the product as
 * `[columns / 16][tokens][16]`, which is the layout board_package/bf16.py's pack_activations and pack_weights write and
 * the board runtime reads back. With activate set, the activations pass through the GELU before they enter the array.
 * With a bias, the product's columns each add one value, the bias packed as `[columns / 16][4][16]` as pack_bias writes
 * it, so the bias costs no launch of its own. The depth and the columns are whole tiles, the host padding an operand
 * a kernel masks with zeros up to them. A depth whose activations one VMEM bank does not hold is walked in segments,
 * each resuming every column tile from the partial product the one before drained.
 */
def Tempo_MatmulOp : Tempo_Op<"matmul", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let summary = "A block of tokens multiplied by a weight matrix in host memory";
    let arguments = (ins I32:$activations, I32:$weights, I32:$drain, Optional<I32>:$bias,
                         Tempo_Field<1, 65535>:$tokens, Tempo_Field<1, 65535>:$depth,
                         Tempo_Field<1, 1048560>:$columns, UnitAttr:$activate);
    let assemblyFormat = [{
        $activations `,` $weights `->` $drain (`bias` $bias^)? `tokens` $tokens `depth` $depth `columns` $columns
        (`activate` $activate^)? attr-dict
    }];
    let hasVerifier = 1;
}

/**
 * @brief Provides an elementwise sum of one block of tokens and a second operand, staged from and drained to host
 * memory.
 *
 * @details The left operand and the output are packed as `[features / 16][tokens][16]`, the layout pack_activations
 * writes. The right operand shares that layout, or with broadcast set it is one feature row added to every token and
 * packed as `[features / 16][4][16]`, each tile repeated across the four sublanes as pack_bias writes it.
 */
def Tempo_AddOp : Tempo_Op<"add", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let summary = "A block of tokens added elementwise to a second operand in host memory";
    let arguments = (ins I32:$left, I32:$right, I32:$output, Tempo_Field<1, 65535>:$tokens,
                         Tempo_Field<1, 1048560>:$features, UnitAttr:$broadcast);
    let assemblyFormat = [{
        $left `,` $right `->` $output `tokens` $tokens `features` $features (`broadcast` $broadcast^)? attr-dict
    }];
    let hasVerifier = 1;
}

/**
 * @brief Provides a LayerNorm over each token of one block, staged from and drained to host memory.
 *
 * @details The rows and the output are packed as `[features / 16][tokens][16]`. The constants block holds gamma and then
 * beta, one row per 16 features, followed by the reciprocal of the width and then epsilon, each repeated across one
 * register's 64 elements, which is the block board_package/gpt2_inference.py's upload_layernorm_constants writes from
 * the image's manifest entry. The epsilon the op carries is the one the host writes there.
 */
def Tempo_LayernormOp : Tempo_Op<"layernorm", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let summary = "A block of tokens normalised over their features in host memory";
    let arguments = (ins I32:$rows, I32:$constants, I32:$output, Tempo_Field<1, 65535>:$tokens,
                         Tempo_Field<1, 1048560>:$features, F32Attr:$epsilon);
    let assemblyFormat = [{
        $rows `,` $constants `->` $output `tokens` $tokens `features` $features `epsilon` $epsilon attr-dict
    }];
    let hasVerifier = 1;
}

/**
 * @brief Provides attention over every head of a fused QKV tensor in host memory, its output written over the queries.
 *
 * @details The queries tensor is the QKV projection's drain, packed per token block as a zeroed 16x16 tile and then
 * `[3 * heads * head / 16][32][16]`: every head's queries, then its keys, then its values. A causal attention attends
 * its `keys` tokens to each other, each query to the keys up to its own. Otherwise one block of 16 queries, opening
 * the tensor's first token block, attends over `keys` cached keys, the cache packed per token block as a zeroed tile
 * and then `[2 * heads * head / 16][32][16]`, keys then values. The constants block holds a tile of the entering
 * maximum and one of zeros, then the mask tiles added to the scores, each laid out keys by queries: the one
 * diagonal tile a causal attention reads, or one per 16 cached keys. The queries arrive scaled by the reciprocal root
 * of the head width.
 */
def Tempo_AttentionOp : Tempo_Op<"attention", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let summary = "Attention over every head of a fused QKV tensor in host memory";
    let arguments = (ins I32:$constants, I32:$queries, Optional<I32>:$cache, Tempo_Field<1, 255>:$heads,
                         Tempo_Field<16, 64>:$head, Tempo_Field<32, 65535>:$keys, UnitAttr:$causal);
    let assemblyFormat = [{
        $constants `,` $queries (`cache` $cache^)? `heads` $heads `head` $head `keys` $keys (`causal` $causal^)?
        attr-dict
    }];
    let hasVerifier = 1;
}

/**
 * @brief Provides a copy of rows from one place in host memory to another, staged through VMEM.
 *
 * @details The copy moves `rows` rows of 32 bytes, consecutive rows `stride` bytes apart at both ends, which is how
 * the board runtime fills its KV cache: a token block's keys and values as one contiguous run, or one token's as a
 * row from each feature tile.
 */
def Tempo_CopyOp : Tempo_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let summary = "Rows copied from one place in host memory to another";
    let arguments = (ins I32:$source, I32:$destination, Tempo_Field<1, 65535>:$rows,
                         Tempo_Field<32, 65535>:$stride);
    let assemblyFormat = "$source `->` $destination `rows` $rows `stride` $stride attr-dict";
    let hasVerifier = 1;
}

/**
 * @brief Provides one image: its operations in program order, which the machine runs once per launch and ends with
 * halt.
 */
def Tempo_KernelOp : Tempo_Op<"kernel", [NoTerminator, SingleBlock, NoRegionArguments, Symbol,
                                            IsolatedFromAbove]>
{
    let summary = "One image";
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

#endif  // TEMPO_TEMPO_OPS_TD
