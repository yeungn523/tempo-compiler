/**
 * @file minitpuisa_ops.td
 * @brief Provides the minitpuisa dialect, one operation per MiniTPU machine instruction, grouped into bundles exactly
 * as an image carries them.
 *
 * @details Every slot and opcode is read from minitpuisa/isa.td, the ISA's single source, so an opcode that moves
 * there moves here, and a mnemonic dropped there fails this file's build rather than encoding a stale number. The
 * arithmetic groups' operations are generated from their mnemonic lists, so an E-slot mnemonic added there becomes an
 * operation here with no edit to this file.
 */

#ifndef MINITPU_MINITPUISA_OPS_TD
#define MINITPU_MINITPUISA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "minitpu/minitpuisa/isa.td"

/// @brief Provides the dialect every MiniTPU machine operation belongs to.
def ISA_Dialect : Dialect
{
    let name = "minitpuisa";
    let cppNamespace = "::mlir::minitpu::isa";
    let summary = "The MiniTPU's machine instructions, grouped into 128-bit bundles";
}

/**
 * @brief Provides the interface every machine instruction implements, which places it in one slot of a bundle.
 *
 * @details A slot's payload is the instruction's fields at their positions within the slot, before the slot is
 * shifted into the bundle. get_code is what the decoder matches a payload against: the opcode for the slots carrying
 * one, and the store bit for X and D. Every register operand is named as minitpuisa/isa.td's register_reads and
 * register_write name it, so the scheduler reads an instruction's registers by the roles its group publishes.
 */
def ISA_SlotOpInterface : OpInterface<"SlotOpInterface">
{
    let cppNamespace = "::mlir::minitpu::isa";
    let methods = [
        StaticInterfaceMethod<"Returns the bundle slot the instruction issues in.", "::mlir::minitpu::isa::kSlot",
                              "get_slot">,
        StaticInterfaceMethod<"Returns the code the decoder matches the slot's payload against.", "uint32_t",
                              "get_code">,
        InterfaceMethod<"Returns the slot payload the instruction encodes to.", "uint64_t", "Encode">,
        StaticInterfaceMethod<"Builds the instruction a payload carries at the builder's insertion point.",
                              "::mlir::Operation*", "Decode",
                              (ins "::mlir::OpBuilder&":$builder, "::mlir::Location":$location,
                                   "uint64_t":$payload)>,
    ];
}

/// @brief Constrains an instruction field to the values its bits hold.
class ISA_Field<int low, int high> : ConfinedAttr<I32Attr, [IntMinValue<low>, IntMaxValue<high>]>;

/// @brief Constrains an instruction field that the assembly elides at its default value.
class ISA_DefaultField<int low, int high, string default> : DefaultValuedAttr<ISA_Field<low, high>, default>;

/// Constrains a field naming one of the 32 vector registers.
def ISA_VectorRegister : ISA_Field<0, 31>;

/// Constrains a field naming one of the 4 scalar registers.
def ISA_ScalarRegister : ISA_Field<0, 3>;

/**
 * @brief Stores the position of a mnemonic within an operation group, or -1 when the group does not name it.
 *
 * @param group Names the group to search.
 * @param mnemonic Names the mnemonic to find.
 */
class ISA_IndexIn<OpGroup group, string mnemonic>
{
    /// Stores the mnemonic's position.
    int value = !foldl(-1, !range(group.mnemonics), found, index,
                       !if(!eq(group.mnemonics[index], mnemonic), index, found));
}

/**
 * @brief Stores a mnemonic as the stem of its C++ class name, vadd becoming Vadd and smov_arg becoming SmovArg.
 *
 * @param mnemonic Names the mnemonic to convert.
 */
class ISA_ClassStem<string mnemonic>
{
    /// Stores the class name stem.
    string value = !subst("_a", "A", !subst("_e", "E", !subst("_s", "S",
                   !toupper(!substr(mnemonic, 0, 1)) # !substr(mnemonic, 1))));
}

/**
 * @brief Provides one machine instruction of an operation group.
 *
 * @details Each format class below defines Encode and Decode over its own fields through formatDefinition.
 *
 * @param group Names the operation group the instruction belongs to, which supplies its slot and opcode.
 * @param mnemonic Names the instruction.
 * @param store_bit Stands in for the opcode in the X and D groups, whose records carry none because their two
 * instructions differ by the store bit alone.
 */
class ISA_Op<OpGroup group, string mnemonic, int store_bit = -1>
    : Op<ISA_Dialect, mnemonic, [ISA_SlotOpInterface]>
{
    defvar index = ISA_IndexIn<group, mnemonic>.value;
    assert !ge(index, 0), "minitpuisa/isa.td's " # group.slot # " group names no '" # mnemonic # "'.";

    /// Stores the code the instruction's payload is matched against.
    int instruction_code = !if(!ge(store_bit, 0), store_bit, group.opcodes[index]);

    /// Stores the definitions of Encode and Decode for the instruction's format.
    code formatDefinition = "";

    let summary = "`" # mnemonic # "`, " # group.slot # " slot";
    let description = group.issue_requirement;
    let extraClassDeclaration = [{
        /// Returns the bundle slot the instruction issues in.
        static ::mlir::minitpu::isa::kSlot get_slot()
        {
            return ::mlir::minitpu::isa::kSlot::k}] # group.slot # [{;
        }

        /// Returns the code the decoder matches the slot's payload against.
        static uint32_t get_code()
        {
            return }] # instruction_code # [{;
        }

        /// Returns the slot payload the instruction encodes to.
        uint64_t Encode();

        /// Builds the instruction a payload carries at the builder's insertion point.
        static ::mlir::Operation* Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload);
    }];
    let extraClassDefinition = formatDefinition;
}

/// @brief Provides an E-slot instruction reading two vector registers and writing a third.
class ISA_EBinaryOp<OpGroup group, string mnemonic> : ISA_Op<group, mnemonic>
{
    let arguments = (ins ISA_VectorRegister:$rd, ISA_VectorRegister:$ra, ISA_VectorRegister:$rb);
    let assemblyFormat = "custom<VectorRegister>($rd) `,` custom<VectorRegister>($ra) `,` "
                         "custom<VectorRegister>($rb) attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kEOpcode.Place(get_code()) | kERd.Place(getRd()) | kERa.Place(getRa()) | kERb.Place(getRb());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kERd.Extract(payload),  // rd
                kERa.Extract(payload),  // ra
                kERb.Extract(payload)   // rb
            );
        }
    }];
}

/// @brief Provides an E-slot instruction reading one vector register and writing another.
class ISA_EUnaryOp<OpGroup group, string mnemonic> : ISA_Op<group, mnemonic>
{
    let arguments = (ins ISA_VectorRegister:$rd, ISA_VectorRegister:$ra);
    let assemblyFormat = "custom<VectorRegister>($rd) `,` custom<VectorRegister>($ra) attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kEOpcode.Place(get_code()) | kERd.Place(getRd()) | kERa.Place(getRa());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kERd.Extract(payload),  // rd
                kERa.Extract(payload)   // ra
            );
        }
    }];
}

foreach index = !range(VectorAlu.mnemonics) in
    def ISA_ # ISA_ClassStem<VectorAlu.mnemonics[index]>.value # Op
        : ISA_EBinaryOp<VectorAlu, VectorAlu.mnemonics[index]>;

foreach group = [VectorMove, VectorSfu, VectorReduce] in
    foreach index = !range(group.mnemonics) in
        def ISA_ # ISA_ClassStem<group.mnemonics[index]>.value # Op : ISA_EUnaryOp<group, group.mnemonics[index]>;

/// @brief Provides vtxin, which stages one vector register into a quarter of the transpose tile, carried in rb.
def ISA_VtxinOp : ISA_Op<TransposeIn, "vtxin">
{
    let arguments = (ins ISA_VectorRegister:$ra, ISA_Field<0, 3>:$quarter);
    let assemblyFormat = "custom<VectorRegister>($ra) `,` `quarter` $quarter attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kEOpcode.Place(get_code()) | kERa.Place(getRa()) | kERb.Place(getQuarter());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kERa.Extract(payload),  // ra
                kERb.Extract(payload)   // quarter
            );
        }
    }];
}

/// @brief Provides vtxout, which drains a quarter of the transpose tile, carried in rb, into a vector register.
def ISA_VtxoutOp : ISA_Op<TransposeOut, "vtxout">
{
    let arguments = (ins ISA_VectorRegister:$rd, ISA_Field<0, 3>:$quarter);
    let assemblyFormat = "custom<VectorRegister>($rd) `,` `quarter` $quarter attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kEOpcode.Place(get_code()) | kERd.Place(getRd()) | kERb.Place(getQuarter());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kERd.Extract(payload),  // rd
                kERb.Extract(payload)   // quarter
            );
        }
    }];
}

/// @brief Provides an X-slot VMEM access, whose induction term, loop level and shift default to off and mask to every
/// sublane.
class ISA_XOp<OpGroup group, string mnemonic, int store_bit> : ISA_Op<group, mnemonic, store_bit>
{
    let arguments = (ins ISA_VectorRegister:$vreg, ISA_Field<0, 8191>:$row,
                         UnitAttr:$agu, ISA_DefaultField<0, 3, "0">:$level,
                         ISA_DefaultField<0, 7, "0">:$shift, ISA_DefaultField<1, 15, "15">:$mask);
    let assemblyFormat = "custom<VectorRegister>($vreg) `,` `row` $row (`agu` $agu^)? (`level` $level^)? "
                         "(`shift` $shift^)? (`mask` $mask^)? attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kXValid.Place(1) | kXStore.Place(get_code()) | kXVreg.Place(getVreg()) | kXRow.Place(getRow()) |
                   kXAgu.Place(getAgu() ? 1 : 0) | kXLevel.Place(getLevel()) | kXShift.Place(getShift()) |
                   kXMask.Place(getMask());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kXVreg.Extract(payload),        // vreg
                kXRow.Extract(payload),         // row
                kXAgu.Extract(payload) != 0,    // agu
                kXLevel.Extract(payload),       // level
                kXShift.Extract(payload),       // shift
                kXMask.Extract(payload)         // mask
            );
        }
    }];
}

def ISA_VldOp : ISA_XOp<VectorLoad, "vld", 0>;
def ISA_VstOp : ISA_XOp<VectorStore, "vst", 1>;

/// @brief Provides an M-slot matrix command naming one vector register.
class ISA_MOp<OpGroup group, string mnemonic> : ISA_Op<group, mnemonic>
{
    let arguments = (ins ISA_VectorRegister:$reg_idx);
    let assemblyFormat = "custom<VectorRegister>($reg_idx) attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kMOpcode.Place(get_code()) | kMRegister.Place(getRegIdx());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(builder, location, kMRegister.Extract(payload));
        }
    }];
}

def ISA_VmatloadOp : ISA_MOp<MatrixLoad, "vmatload">;
def ISA_VmatpushOp : ISA_MOp<MatrixPush, "vmatpush">;
def ISA_VmatpopOp : ISA_MOp<MatrixPop, "vmatpop">;

/// @brief Provides smovi, which writes an unsigned 16-bit immediate into a scalar register.
def ISA_SmoviOp : ISA_Op<Scalar, "smovi">
{
    let arguments = (ins ISA_ScalarRegister:$rd, ISA_Field<0, 65535>:$imm);
    let assemblyFormat = "custom<ScalarRegister>($rd) `,` $imm attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kSValid.Place(1) | kSOpcode.Place(get_code()) | kSRd.Place(getRd()) | kSImmediate.Place(getImm());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kSRd.Extract(payload),        // rd
                kSImmediate.Extract(payload)  // imm
            );
        }
    }];
}

/// @brief Provides smov_arg, which reads the launch argument its rs field names into a scalar register.
def ISA_SmovArgOp : ISA_Op<Scalar, "smov_arg">
{
    let arguments = (ins ISA_ScalarRegister:$rd, ISA_ScalarRegister:$rs);
    let assemblyFormat = "custom<ScalarRegister>($rd) `,` `arg` $rs attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kSValid.Place(1) | kSOpcode.Place(get_code()) | kSRd.Place(getRd()) | kSRs.Place(getRs());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kSRd.Extract(payload),  // rd
                kSRs.Extract(payload)   // rs
            );
        }
    }];
}

/// @brief Provides an S-slot operation reading a scalar register and a signed 16-bit immediate.
class ISA_SImmediateOp<string mnemonic> : ISA_Op<Scalar, mnemonic>
{
    let arguments = (ins ISA_ScalarRegister:$rd, ISA_ScalarRegister:$rs, ISA_Field<-32768, 32767>:$imm);
    let assemblyFormat = "custom<ScalarRegister>($rd) `,` custom<ScalarRegister>($rs) `,` $imm attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kSValid.Place(1) | kSOpcode.Place(get_code()) | kSRd.Place(getRd()) | kSRs.Place(getRs()) |
                   kSImmediate.Place(getImm());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kSRd.Extract(payload),                                // rd
                kSRs.Extract(payload),                                // rs
                static_cast<int16_t>(kSImmediate.Extract(payload))  // imm
            );
        }
    }];
}

def ISA_SaddiOp : ISA_SImmediateOp<"saddi">;
def ISA_SmacOp : ISA_SImmediateOp<"smac">;
def ISA_SshlOp : ISA_SImmediateOp<"sshl">;

/// @brief Provides lbegin, which opens a hardware loop running its induction variable from lo to hi by step.
def ISA_LbeginOp : ISA_Op<Control, "lbegin">
{
    let arguments = (ins ISA_Field<0, 65535>:$hi, ISA_DefaultField<0, 15, "0">:$lo,
                         ISA_DefaultField<0, 15, "1">:$step);
    let assemblyFormat = "$hi (`lo` $lo^)? (`step` $step^)? attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kFOpcode.Place(get_code()) | kLoopLow.Place(getLo()) | kLoopStep.Place(getStep()) |
                   kLoopHigh.Place(getHi());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kLoopHigh.Extract(payload),  // hi
                kLoopLow.Extract(payload),   // lo
                kLoopStep.Extract(payload)   // step
            );
        }
    }];
}

/// @brief Provides flush_slot, which blocks until every DMA slot its mask names reports done.
def ISA_FlushSlotOp : ISA_Op<Control, "flush_slot">
{
    let arguments = (ins ISA_Field<1, 3>:$mask);
    let assemblyFormat = "$mask attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kFOpcode.Place(get_code()) | kFlushMask.Place(getMask());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(builder, location, kFlushMask.Extract(payload));
        }
    }];
}

/// @brief Provides an F-slot control operation carrying no operand.
class ISA_FBareOp<string mnemonic> : ISA_Op<Control, mnemonic>
{
    let assemblyFormat = "attr-dict";
    let formatDefinition = [{
        // Keeps Encode an instance method, which the slot interface calls through an operation.
        uint64_t $cppClass::Encode()  // NOLINT(readability-convert-member-functions-to-static)
        {
            return kFOpcode.Place(get_code());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t)
        {
            return $cppClass::create(builder, location);
        }
    }];
}

def ISA_LoopEndOp : ISA_FBareOp<"loop_end">;
def ISA_HaltOp : ISA_FBareOp<"halt">;
def ISA_VmemswapOp : ISA_FBareOp<"vmemswap">;

/// @brief Provides a DMA descriptor, which moves rows between host memory and VMEM asynchronously.
class ISA_DOp<string mnemonic, int store_bit> : ISA_Op<Descriptor, mnemonic, store_bit>
{
    let arguments = (ins ISA_Field<0, 1>:$dma_slot, ISA_Field<0, 16383>:$vmem, ISA_Field<1, 128>:$rows,
                         ISA_ScalarRegister:$base_sreg, ISA_ScalarRegister:$stride_sreg);
    let assemblyFormat = "`slot` $dma_slot `,` `vmem` $vmem `,` `rows` $rows `,` "
                         "`base` custom<ScalarRegister>($base_sreg) `,` `stride` custom<ScalarRegister>($stride_sreg) "
                         "attr-dict";
    let formatDefinition = [{
        uint64_t $cppClass::Encode()
        {
            return kDStore.Place(get_code()) | kDDmaSlot.Place(getDmaSlot()) | kDVmem.Place(getVmem()) |
                   kDSourceMode.Place(1) | kDRowCount.Place(getRows() - 1) | kDBase.Place(getBaseSreg()) |
                   kDStride.Place(getStrideSreg());
        }

        ::mlir::Operation* $cppClass::Decode(::mlir::OpBuilder& builder, ::mlir::Location location, uint64_t payload)
        {
            return $cppClass::create(
                builder,
                location,
                kDDmaSlot.Extract(payload),       // dma_slot
                kDVmem.Extract(payload),          // vmem
                kDRowCount.Extract(payload) + 1,  // rows
                kDBase.Extract(payload),          // base_sreg
                kDStride.Extract(payload)         // stride_sreg
            );
        }
    }];
}

def ISA_VmemldOp : ISA_DOp<"vmemld", 0>;
def ISA_VmemstOp : ISA_DOp<"vmemst", 1>;

/**
 * @brief Provides one 128-bit bundle: at most one instruction per slot, and the delay before the next bundle issues.
 *
 * @details The delay holds the fetch head after the bundle issues, which is the only stall the machine has, so these
 * fields resolve every hazard a schedule leaves. A bundle carrying a DMA descriptor carries nothing else, because the
 * descriptor overlays the compute slots.
 */
def ISA_BundleOp : Op<ISA_Dialect, "bundle", [NoTerminator, SingleBlock, NoRegionArguments, HasParent<"ProgramOp">]>
{
    let summary = "One bundle and the cycles the fetch head holds after it";
    let arguments = (ins ISA_DefaultField<0, 63, "0">:$delay);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "(`delay` $delay^)? attr-dict-with-keyword $body";
    let hasVerifier = 1;
}

/// @brief Provides one image: its bundles in the order the machine fetches them.
def ISA_ProgramOp : Op<ISA_Dialect, "program", [NoTerminator, SingleBlock, NoRegionArguments, Symbol,
                                                IsolatedFromAbove]>
{
    let summary = "One image's bundles, in fetch order";
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
    let hasVerifier = 1;
}

#endif  // MINITPU_MINITPUISA_OPS_TD
