/**
 * @file minitpu_ops.td
 * @brief Provides the minitpu dialect, which states a kernel as machine operations over virtual registers and VMEM
 * buffers, leaving registers, rows, loops and timing to the passes that lower it into minitpuisa.
 *
 * @details Values replace register numbers: a !minitpu.vreg is a vector register's contents and an i32 is a scalar
 * register's, so a register is live exactly as long as its value is used. A !minitpu.vmem is a VMEM region whose base
 * row the allocator chooses. Loops are scf.for, and scalar arithmetic is arith, so pointer advances and loop-carried
 * accumulators are ordinary SSA. A minitpu.reg attribute on the operation defining a value names its physical
 * register, written by the register allocator or by hand to pin it, and a base attribute pins a buffer's row.
 */

#ifndef MINITPU_MINITPU_OPS_TD
#define MINITPU_MINITPU_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "codegen/minitpuisa.td"

/// @brief Provides the dialect a kernel is stated in before its registers, rows and timing are chosen.
def Minitpu_Dialect : Dialect
{
    let name = "minitpu";
    let cppNamespace = "::mlir::minitpu";
    let summary = "MiniTPU machine operations over virtual registers and VMEM buffers";
    let useDefaultTypePrinterParser = 1;
    let hasOperationAttrVerify = 1;
    let extraClassDeclaration = [{
        /// Returns the name of the attribute naming the physical register a value lives in.
        static ::llvm::StringRef get_register_attribute_name()
        {
            return "minitpu.reg";
        }
    }];
}

/// @brief Provides the base class of every minitpu type.
class Minitpu_Type<string name, string type_mnemonic> : TypeDef<Minitpu_Dialect, name>
{
    let mnemonic = type_mnemonic;
}

/// @brief Provides a vector register's contents: 64 BF16 elements, four sublanes of sixteen lanes.
def Minitpu_VRegType : Minitpu_Type<"VReg", "vreg">
{
    let summary = "A vector register's contents";
}

/// @brief Provides a VMEM region of a fixed row count, whose base row the allocator chooses.
def Minitpu_VmemType : Minitpu_Type<"Vmem", "vmem">
{
    let summary = "A VMEM region of a fixed row count";
    let parameters = (ins "uint32_t":$rows);
    let assemblyFormat = "`<` $rows `>`";
}

/// Constrains an attribute to the values a field of the given range holds.
class Minitpu_Field<int low, int high> : ConfinedAttr<I32Attr, [IntMinValue<low>, IntMaxValue<high>]>;

/**
 * @brief Stores the position of a mnemonic within an operation group, or -1 when the group does not name it.
 *
 * @param group Names the group to search.
 * @param mnemonic Names the mnemonic to find.
 */
class Minitpu_IndexIn<OpGroup group, string mnemonic>
{
    /// Stores the mnemonic's position.
    int value = !foldl(-1, !range(group.mnemonics), found, index,
                       !if(!eq(group.mnemonics[index], mnemonic), index, found));
}

/// @brief Provides the base class of every minitpu operation.
class Minitpu_Op<string mnemonic, list<Trait> traits = []> : Op<Minitpu_Dialect, mnemonic, traits>;

/**
 * @brief Provides an operation the machine runs as the instruction of the same mnemonic, which codegen/minitpuisa.td
 * must name in `group`.
 *
 * @param group Names the operation group the instruction belongs to.
 * @param mnemonic Names the instruction.
 */
class Minitpu_InstructionOp<OpGroup group, string mnemonic, list<Trait> traits = []> : Minitpu_Op<mnemonic, traits>
{
    defvar index = Minitpu_IndexIn<group, mnemonic>.value;
    assert !ge(index, 0), "codegen/minitpuisa.td's " # group.slot # " group names no '" # mnemonic # "'.";
    let summary = "`" # mnemonic # "`";
    let description = group.issue_requirement;
}

/// @brief Provides an elementwise operation over two vector registers.
class Minitpu_BinaryOp<OpGroup group, string mnemonic> : Minitpu_InstructionOp<group, mnemonic, [Pure]>
{
    let arguments = (ins Minitpu_VRegType:$lhs, Minitpu_VRegType:$rhs);
    let results = (outs Minitpu_VRegType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

/// @brief Provides an operation over one vector register.
class Minitpu_UnaryOp<OpGroup group, string mnemonic> : Minitpu_InstructionOp<group, mnemonic, [Pure]>
{
    let arguments = (ins Minitpu_VRegType:$operand);
    let results = (outs Minitpu_VRegType:$result);
    let assemblyFormat = "$operand attr-dict";
}

foreach index = !range(VectorAlu.mnemonics) in
    def Minitpu_ # !toupper(!substr(VectorAlu.mnemonics[index], 0, 1)) # !substr(VectorAlu.mnemonics[index], 1) # Op
        : Minitpu_BinaryOp<VectorAlu, VectorAlu.mnemonics[index]>;

foreach group = [VectorMove, VectorSfu, VectorReduce] in
    foreach index = !range(group.mnemonics) in
        def Minitpu_ # !toupper(!substr(group.mnemonics[index], 0, 1)) # !substr(group.mnemonics[index], 1) # Op
            : Minitpu_UnaryOp<group, group.mnemonics[index]>;

/// @brief Provides vtxin, which stages a vector register into a quarter of the transpose tile.
def Minitpu_VtxinOp : Minitpu_InstructionOp<TransposeIn, "vtxin">
{
    let arguments = (ins Minitpu_VRegType:$operand, Minitpu_Field<0, 3>:$quarter);
    let assemblyFormat = "$operand `,` `quarter` $quarter attr-dict";
}

/// @brief Provides vtxout, which drains a quarter of the transpose tile into a vector register.
def Minitpu_VtxoutOp : Minitpu_InstructionOp<TransposeOut, "vtxout">
{
    let arguments = (ins Minitpu_Field<0, 3>:$quarter);
    let results = (outs Minitpu_VRegType:$result);
    let assemblyFormat = "`quarter` $quarter attr-dict";
}

/**
 * @brief Provides vld, which loads the four rows starting at `row` of a VMEM buffer into a vector register.
 *
 * @details The row is an index the loop lowering resolves to the instruction's static row plus an induction term: a
 * constant, or a constant plus an enclosing scf.for's induction variable shifted left by a power of two.
 */
def Minitpu_VldOp : Minitpu_InstructionOp<VectorLoad, "vld", [MemoryEffects<[MemRead]>]>
{
    let arguments = (ins Minitpu_VmemType:$buffer, Index:$row, DefaultValuedAttr<Minitpu_Field<1, 15>, "15">:$mask);
    let results = (outs Minitpu_VRegType:$result);
    let assemblyFormat = "$buffer `[` $row `]` (`mask` $mask^)? attr-dict `:` qualified(type($buffer))";
}

/// @brief Provides vst, which stores a vector register into the four rows starting at `row` of a VMEM buffer.
def Minitpu_VstOp : Minitpu_InstructionOp<VectorStore, "vst", [MemoryEffects<[MemWrite]>]>
{
    let arguments = (ins Minitpu_VRegType:$value, Minitpu_VmemType:$buffer, Index:$row,
                         DefaultValuedAttr<Minitpu_Field<1, 15>, "15">:$mask);
    let assemblyFormat = "$value `,` $buffer `[` $row `]` (`mask` $mask^)? attr-dict `:` qualified(type($buffer))";
}

/**
 * @brief Provides vmatload, which loads a 16x16 weight tile from four vector registers into the array.
 *
 * @details The instruction reads four consecutive registers from a base, so the allocator places the four operands
 * in consecutive registers.
 */
def Minitpu_VmatloadOp : Minitpu_InstructionOp<MatrixLoad, "vmatload", [MemoryEffects<[MemWrite]>]>
{
    let arguments = (ins Minitpu_VRegType:$quarter0, Minitpu_VRegType:$quarter1, Minitpu_VRegType:$quarter2,
                         Minitpu_VRegType:$quarter3);
    let assemblyFormat = "$quarter0 `,` $quarter1 `,` $quarter2 `,` $quarter3 attr-dict";
}

/// @brief Provides vmatpush, which streams one vector register of activations through the array.
def Minitpu_VmatpushOp : Minitpu_InstructionOp<MatrixPush, "vmatpush", [MemoryEffects<[MemWrite]>]>
{
    let arguments = (ins Minitpu_VRegType:$operand);
    let assemblyFormat = "$operand attr-dict";
}

/// @brief Provides vmatpop, which drains the oldest push's results into a vector register.
def Minitpu_VmatpopOp : Minitpu_InstructionOp<MatrixPop, "vmatpop", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let results = (outs Minitpu_VRegType:$result);
    let assemblyFormat = "attr-dict";
}

/// @brief Provides a launch argument, which the machine reads into a scalar register.
def Minitpu_ArgOp : Minitpu_Op<"arg", [Pure]>
{
    let summary = "A launch argument, read into a scalar register";
    let arguments = (ins Minitpu_Field<0, 3>:$index);
    let results = (outs I32:$result);
    let assemblyFormat = "$index attr-dict";
}

/// @brief Provides a DMA descriptor moving rows between host memory and a VMEM buffer.
class Minitpu_DmaOp<string mnemonic, list<Trait> traits> : Minitpu_InstructionOp<Descriptor, mnemonic, traits>
{
    let arguments = (ins Minitpu_VmemType:$buffer, Index:$row, I32:$base, I32:$stride, Minitpu_Field<1, 128>:$rows,
                         Minitpu_Field<0, 1>:$dma_slot);
    let assemblyFormat = [{
        $buffer `[` $row `]` `,` `base` $base `,` `stride` $stride `,` `rows` $rows `,` `slot` $dma_slot attr-dict
        `:` qualified(type($buffer))
    }];
    let hasVerifier = 1;
}

def Minitpu_VmemldOp : Minitpu_DmaOp<"vmemld", [MemoryEffects<[MemRead, MemWrite]>]>;
def Minitpu_VmemstOp : Minitpu_DmaOp<"vmemst", [MemoryEffects<[MemRead, MemWrite]>]>;

/// @brief Provides flush_slot, which blocks until every DMA slot its mask names reports done.
def Minitpu_FlushSlotOp : Minitpu_InstructionOp<Control, "flush_slot", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let arguments = (ins Minitpu_Field<1, 3>:$mask);
    let assemblyFormat = "$mask attr-dict";
}

/// @brief Provides vmemswap, which exchanges the VMEM slots compute and DMA address.
def Minitpu_VmemswapOp : Minitpu_InstructionOp<Control, "vmemswap", [MemoryEffects<[MemRead, MemWrite]>]>
{
    let assemblyFormat = "attr-dict";
}

/// @brief Provides a VMEM buffer, whose base row the allocator chooses unless `base` pins it.
def Minitpu_AllocOp : Minitpu_Op<"alloc", [MemoryEffects<[MemAlloc]>]>
{
    let summary = "A VMEM region";
    let arguments = (ins OptionalAttr<Minitpu_Field<0, 16383>>:$base);
    let results = (outs Minitpu_VmemType:$buffer);
    let assemblyFormat = "attr-dict `:` qualified(type($buffer))";
}

/**
 * @brief Provides one image: its operations in program order, which the machine runs once per launch and ends with
 * halt.
 */
def Minitpu_KernelOp : Minitpu_Op<"kernel", [NoTerminator, SingleBlock, NoRegionArguments, Symbol,
                                            IsolatedFromAbove]>
{
    let summary = "One image";
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

#endif  // MINITPU_MINITPU_OPS_TD
