/**
 * @file
 *
 * @brief Provides the bit layout of the MiniTPU's 128-bit bundle, as one named field per instruction operand.
 *
 * The layout mirrors sequencer_pkg's compute_payload_t and d_wire_t at NUM_LANES=16 and NUM_SUBLANES=4, as
 * codegen/asm.py does. Every slot position derives from the slot widths, so a width change moves them all.
 */

#ifndef MINITPU_MINITPUISA_ENCODING_H
#define MINITPU_MINITPUISA_ENCODING_H

#include <cstdint>
#include <limits>

namespace mlir::minitpu::isa
{
    /// Stores one bundle as the machine fetches it.
    using Word = unsigned __int128;

    /// Stores the width of one bundle in bits.
    static constexpr uint32_t kBundleBits = 128;

    /// Stores the bits in one byte of an image.
    static constexpr uint32_t kBitsPerByte = 8;

    /// Stores the size of one bundle in bytes.
    static constexpr uint32_t kBundleBytes = kBundleBits / kBitsPerByte;

    /// Stores the bit that selects the DMA format over the compute format.
    static constexpr uint32_t kDmaBit = kBundleBits - 1;

    /// Defines the slots of a bundle. D is the DMA descriptor, which a DMA bundle carries in place of every other slot.
    enum class kSlot : uint8_t
    {
        kE,  ///< Vector arithmetic.
        kM,  ///< Matrix commands.
        kX,  ///< VMEM accesses.
        kS,  ///< Scalar operations.
        kF,  ///< Control operations.
        kD,  ///< DMA descriptor.
    };

    /// Stores the number of slots a bundle holds.
    static constexpr uint32_t kSlotCount = 6;

    /// Stores one field's position and width within its slot or bundle.
    struct BitField
    {
        uint32_t shift = 0;  ///< The position of the field's lowest bit.
        uint32_t width = 0;  ///< The bits the field holds.

        /// Returns the field's bits set and every other bit clear.
        [[nodiscard]]
        constexpr uint64_t Mask() const
        {
            return width >= std::numeric_limits<uint64_t>::digits ? std::numeric_limits<uint64_t>::max()
                                                                  : (uint64_t {1} << width) - 1;
        }

        /// Returns `value` at the field's position, truncated to its width.
        [[nodiscard]]
        constexpr uint64_t Place(const uint64_t value) const
        {
            return (value & Mask()) << shift;
        }

        /// Returns the field's value within a slot payload.
        [[nodiscard]]
        constexpr uint32_t Extract(const uint64_t payload) const
        {
            return static_cast<uint32_t>((payload >> shift) & Mask());
        }

        /// Returns the field's value within a whole bundle.
        [[nodiscard]]
        constexpr uint64_t ExtractFrom(const Word word) const
        {
            return static_cast<uint64_t>(word >> shift) & Mask();
        }

        /// Returns `value` at the field's position within a whole bundle.
        [[nodiscard]]
        constexpr Word PlaceIn(const uint64_t value) const
        {
            return static_cast<Word>(value & Mask()) << shift;
        }
    };

    /// Stores the E slot, which carries vector arithmetic.
    static constexpr BitField kESlot {.shift = kDmaBit - 19, .width = 19};

    /// Stores the M slot, which carries matrix commands.
    static constexpr BitField kMSlot {.shift = kESlot.shift - 7, .width = 7};

    /// Stores the X slot, which carries VMEM accesses.
    static constexpr BitField kXSlot {.shift = kMSlot.shift - 33, .width = 33};

    /// Stores the S slot, which carries scalar operations.
    static constexpr BitField kSSlot {.shift = kXSlot.shift - 27, .width = 27};

    /// Stores the F slot, which carries control operations.
    static constexpr BitField kFSlot {.shift = kSSlot.shift - 27, .width = 27};

    /// Stores the compute format's delay field, which holds the fetch head for that many cycles after the bundle
    /// issues.
    static constexpr BitField kDelay {.shift = kFSlot.shift - 6, .width = 6};

    /// Stores the reserved bits the compute format leaves below its delay field.
    static constexpr uint32_t kComputeReservedBits = 8;

    static_assert(
        kDelay.shift == kComputeReservedBits,
        "The compute payload's field positions disagree with sequencer_pkg's slot widths."
    );

    /// Stores the DMA descriptor, which a DMA bundle carries in place of every compute slot.
    static constexpr BitField kDSlot {.shift = kDmaBit - 36, .width = 36};

    /// Stores the DMA format's delay field, which sits directly below the descriptor.
    static constexpr BitField kDmaDelay {.shift = kDSlot.shift - kDelay.width, .width = kDelay.width};

    /// Stores the E slot's opcode.
    static constexpr BitField kEOpcode {.shift = 15, .width = 4};

    /// Stores the E slot's destination vector register.
    static constexpr BitField kERd {.shift = 10, .width = 5};

    /// Stores the E slot's first source vector register.
    static constexpr BitField kERa {.shift = 5, .width = 5};

    /// Stores the E slot's second source vector register, which the transposes read as the tile quarter.
    static constexpr BitField kERb {.shift = 0, .width = 5};

    /// Stores the M slot's opcode.
    static constexpr BitField kMOpcode {.shift = 5, .width = 2};

    /// Stores the vector register a matrix command reads or writes.
    static constexpr BitField kMRegister {.shift = 0, .width = 5};

    /// Stores the X slot's valid bit, which marks the slot occupied.
    static constexpr BitField kXValid {.shift = 32, .width = 1};

    /// Stores the X slot's store bit, which separates vst from vld.
    static constexpr BitField kXStore {.shift = 31, .width = 1};

    /// Stores the vector register a VMEM access writes or reads.
    static constexpr BitField kXVreg {.shift = 26, .width = 5};

    /// Stores the first VMEM row, before the loop and register terms are added.
    static constexpr BitField kXRow {.shift = 13, .width = 13};

    /// Stores the bit that adds the loop induction variable to the row.
    static constexpr BitField kXAgu {.shift = 12, .width = 1};

    /// Stores the loop whose induction variable the access reads, 0 being the outermost.
    static constexpr BitField kXLevel {.shift = 10, .width = 2};

    /// Stores the left shift applied to that induction variable.
    static constexpr BitField kXShift {.shift = 7, .width = 3};

    /// Stores the sublanes that take part, one bit each.
    static constexpr BitField kXMask {.shift = 0, .width = 4};

    /// Stores the S slot's valid bit, which marks the slot occupied.
    static constexpr BitField kSValid {.shift = 26, .width = 1};

    /// Stores the S slot's opcode.
    static constexpr BitField kSOpcode {.shift = 23, .width = 3};

    /// Stores the S slot's destination scalar register.
    static constexpr BitField kSRd {.shift = 21, .width = 2};

    /// Stores the S slot's source scalar register, which smov_arg reads as the launch argument.
    static constexpr BitField kSRs {.shift = 19, .width = 2};

    /// Stores the S slot's immediate, which smovi reads unsigned and every other scalar operation signed.
    static constexpr BitField kSImmediate {.shift = 0, .width = 16};

    /// Stores the F slot's opcode.
    static constexpr BitField kFOpcode {.shift = 24, .width = 3};

    /// Stores the value lbegin starts its induction variable from.
    static constexpr BitField kLoopLow {.shift = 20, .width = 4};

    /// Stores the amount lbegin advances its induction variable by.
    static constexpr BitField kLoopStep {.shift = 16, .width = 4};

    /// Stores the bound lbegin stops its induction variable at.
    static constexpr BitField kLoopHigh {.shift = 0, .width = 16};

    /// Stores the DMA slots flush_slot waits on, one bit each.
    static constexpr BitField kFlushMask {.shift = 0, .width = 2};

    /// Stores the descriptor's store bit, which separates vmemst from vmemld.
    static constexpr BitField kDStore {.shift = 35, .width = 1};

    /// Stores the DMA slot carrying the descriptor.
    static constexpr BitField kDDmaSlot {.shift = 34, .width = 1};

    /// Stores the VMEM address, whose top bit selects the VMEM slot.
    static constexpr BitField kDVmem {.shift = 20, .width = 14};

    /// Stores the source mode, held at D_SRC_SREG so the base and stride are read from scalar registers.
    static constexpr BitField kDSourceMode {.shift = 19, .width = 1};

    /// Stores one less than the rows the descriptor moves.
    static constexpr BitField kDRowCount {.shift = 4, .width = 7};

    /// Stores the scalar register holding the host base address.
    static constexpr BitField kDBase {.shift = 2, .width = 2};

    /// Stores the scalar register holding the host row stride.
    static constexpr BitField kDStride {.shift = 0, .width = 2};

    /// Returns the slot field that carries `slot` within a bundle.
    constexpr BitField SlotField(const kSlot slot)
    {
        switch (slot)
        {
            case kSlot::kE: return kESlot;
            case kSlot::kM: return kMSlot;
            case kSlot::kX: return kXSlot;
            case kSlot::kS: return kSSlot;
            case kSlot::kF: return kFSlot;
            case kSlot::kD: return kDSlot;
        }
        return kESlot;
    }

    /// Returns the field a slot's payload is matched on: its opcode, or for X and D its store bit.
    constexpr BitField CodeField(const kSlot slot)
    {
        switch (slot)
        {
            case kSlot::kE: return kEOpcode;
            case kSlot::kM: return kMOpcode;
            case kSlot::kX: return kXStore;
            case kSlot::kS: return kSOpcode;
            case kSlot::kF: return kFOpcode;
            case kSlot::kD: return kDStore;
        }
        return kEOpcode;
    }
}  // namespace mlir::minitpu::isa

#endif  // MINITPU_MINITPUISA_ENCODING_H
