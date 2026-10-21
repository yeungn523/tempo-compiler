/**
 * @file isa.td
 * @brief Provides the implemented ISA scheduling contract, which is the single source of truth every generated
 * artifact derives from.
 *
 * @details The minitpu and minitpuisa dialects read their opcodes and slots from these records, and minitpu-tblgen's
 * -gen-minitpu-sched writes the scheduler's latencies and capacities from them. tools/toolchain/gen_isa_doc.py reads
 * this file through a TableGen driver and writes the operation table published in docs/ISA_AND_INTERFACES.md and the
 * per-opcode facts the emulator decodes images with in tools/emulators/isa_model.py. That tool regenerates both under
 * --write and exits non-zero on a stale one under --check. The records stay readable by llvm-tblgen alone, while the operation definitions that include them need
 * mlir-tblgen for its C++ backends.
 */

/// @brief Stores the array geometry the writeback offsets below are quoted against.
class Profile
{
    /// Stores the lanes one VMEM row spans.
    int lane_count = 16;

    /// Stores the sublanes one vector register access walks.
    int sublane_count = 4;

    /// Stores the registers the vector file holds.
    int vreg_entry_count = 32;

    /// Stores the elements one vector register holds.
    int vreg_element_count = 64;

    /// Stores the convention relating a writeback offset to the cycle its consumer becomes legal.
    string note = "";
}

def MiniTpuProfile : Profile
{
    let note = "W is the sequencer's single-write-port reservation offset; the earliest dependent consumer is L = W + 1.";
}

/**
 * @brief Stores one group of opcodes sharing a physical slot, an operand shape and a writeback offset.
 *
 * @details Leaving writeback_offset uninitialized states that the group reserves no write port, which is a different
 * claim from reserving it at cycle zero. writeback_last_offset carries the closing cycle of a reservation spanning
 * several beats, and the pair then publishes as a range.
 *
 * @param slot_letter Names the physical bundle slot, being one of E, X, M, S, D and F.
 * @param mnemonic_names Names every mnemonic the group covers, positionally matching opcodes.
 */
class OpGroup<string slot_letter, list<string> mnemonic_names>
{
    /// Stores the physical bundle slot the group issues in.
    string slot = slot_letter;

    /// Stores every mnemonic the group covers.
    list<string> mnemonics = mnemonic_names;

    /// Stores the vector registers and ports one operation reads, as the published table words it.
    string vreg_reads = "";

    /// Stores the vector registers and ports one operation writes, as the published table words it.
    string vreg_writes = "";

    /// Stores the issue interval and the structural requirement a schedule must respect, as the published table words
    /// it.
    string issue_requirement = "";

    /// Stores the cycle at which the group reserves the vector write port.
    int writeback_offset;

    /// Stores the closing cycle of a reservation spanning several beats.
    int writeback_last_offset;

    /// Stores the sequencer localparam the writeback offset mirrors.
    string rtl_parameter = "";

    /// Stores the consumer convention replacing the offset plus one that every other group follows.
    string consumer_override = "";

    /// Stores the numeric opcode of each mnemonic, positionally.
    list<int> opcodes = [];

    /// Stores the operand registers one operation reads, as the assembler's hazard model names them.
    list<string> register_reads = [];

    /// Stores the operand register one operation writes, as the assembler's hazard model names it.
    string register_write;

    /// Determines whether an operation of this group claims vector register read port A.
    bit claims_read_port_a = 0;
}

def VectorAlu : OpGroup<"E", ["vadd", "vsub", "vmul", "vmax", "vmin"]>
{
    let vreg_reads = "A+B";
    let vreg_writes = "1 full VREG";
    let issue_requirement = "II=1; one E op per bundle";
    let writeback_offset = 5;
    let rtl_parameter = "WB_W_ALU";
    let opcodes = [1, 2, 3, 11, 12];
    let register_reads = ["ra", "rb"];
    let register_write = "rd";
    let claims_read_port_a = 1;
}

def VectorMove : OpGroup<"E", ["vmov"]>
{
    let vreg_reads = "A";
    let vreg_writes = "1 full VREG";
    let issue_requirement = "II=1";
    let writeback_offset = 5;
    let rtl_parameter = "WB_W_ALU";
    let opcodes = [4];
    let register_reads = ["ra"];
    let register_write = "rd";
    let claims_read_port_a = 1;
}

def VectorSfu : OpGroup<"E", ["vgelu", "vexp", "vrecip", "vrsqrt"]>
{
    let vreg_reads = "A";
    let vreg_writes = "1 full VREG";
    let issue_requirement = "II=1; SFU arithmetic includes a multiplier";
    let writeback_offset = 7;
    let rtl_parameter = "WB_W_SFU";
    let opcodes = [5, 6, 7, 8];
    let register_reads = ["ra"];
    let register_write = "rd";
    let claims_read_port_a = 1;
}

def VectorReduce : OpGroup<"E", ["vredsum", "vredmax"]>
{
    let vreg_reads = "A, all 64 elements";
    let vreg_writes = "1 broadcast VREG";
    let issue_requirement = "II=1; measured XLU source-valid latency is 13 cycles; W=15 includes the two registered writeback boundaries";
    let writeback_offset = 15;
    let rtl_parameter = "WB_W_REDUCE";
    let opcodes = [9, 10];
    let register_reads = ["ra"];
    let register_write = "rd";
    let claims_read_port_a = 1;
}

def TransposeIn : OpGroup<"E", ["vtxin"]>
{
    let vreg_reads = "A";
    let vreg_writes = "the transpose staging array";
    let issue_requirement = "II=1; stages one VREG into the transpose array and writes no register";
    let opcodes = [13];
    let register_reads = ["ra"];
    let claims_read_port_a = 1;
}

def TransposeOut : OpGroup<"E", ["vtxout"]>
{
    let vreg_reads = "the transpose staging array";
    let vreg_writes = "1 full VREG";
    let issue_requirement = "II=1; no RTL localparam names this offset";
    let writeback_offset = 4;
    let opcodes = [14];
    let register_reads = [];
    let register_write = "rd";
}

def VectorLoad : OpGroup<"X", ["vld"]>
{
    let vreg_reads = "VMEM compute port";
    let vreg_writes = "1 full/masked VREG";
    let issue_requirement = "II=1; VMEM read latency parameter is 3, followed by shared writeback staging. The NUM_SUBLANES sublanes read consecutive rows, so their banks are a permutation of the bank count and no access splits, which is why a schedule prices every access at one cycle (vpu_bank_select.sv)";
    let writeback_offset = 6;
    let rtl_parameter = "WB_W_VLD";
    let register_reads = [];
    let register_write = "vreg";
}

def VectorStore : OpGroup<"X", ["vst"]>
{
    let vreg_reads = "A";
    let vreg_writes = "VMEM compute port";
    let issue_requirement = "II=1; competes with every other A consumer";
    let register_reads = ["vreg"];
    let claims_read_port_a = 1;
}

def MatrixLoad : OpGroup<"M", ["vmatload"]>
{
    let vreg_reads = "A over four VREGs";
    let vreg_writes = "MXU weight input";
    let issue_requirement = "18-cycle command; owns VREG read port A while it streams";
    let opcodes = [1];
    let register_reads = ["reg_idx+0..3"];
    let claims_read_port_a = 1;
}

def MatrixPush : OpGroup<"M", ["vmatpush"]>
{
    let vreg_reads = "A over one VREG";
    let vreg_writes = "MXU activation input";
    let issue_requirement = "occupies the matrix controller for 5 cycles and produces its first result at +82 in the 16x4 profile; an earlier vmatpop stalls in VPOP_STREAM rather than reading a partial result";
    let opcodes = [2];
    let register_reads = ["reg_idx"];
    let claims_read_port_a = 1;
}

def MatrixPop : OpGroup<"M", ["vmatpop"]>
{
    let vreg_reads = "MXU output";
    let vreg_writes = "1 VREG, four masked beats";
    let issue_requirement = "7-cycle command; reserves the write port for every masked beat";
    let writeback_offset = 3;
    let writeback_last_offset = 6;
    let rtl_parameter = "WB_W_MPOP_FIRST..WB_W_MPOP_LAST";
    let opcodes = [3];
    let register_reads = [];
    let register_write = "reg_idx";
}

def Scalar : OpGroup<"S", ["smovi", "smov_arg", "saddi", "smac", "sshl"]>
{
    let vreg_reads = "no VREG";
    let vreg_writes = "one of four SREGs";
    let issue_requirement = "No dynamic SREG RAW scoreboard; assembly must preserve the documented boundary before D consumes a new SREG";
    let opcodes = [0, 1, 2, 3, 4];
    let register_reads = ["rs"];
    let register_write = "rd";
}

def Descriptor : OpGroup<"D", ["vmemld", "vmemst"]>
{
    let vreg_reads = "other VMEM slot";
    let vreg_writes = "no VREG";
    let issue_requirement = "Descriptor issue is asynchronous and DMA-format exclusive; `flush_slot` waits for completion";
    let consumer_override = "fence";
    let register_reads = ["base_sreg", "stride_sreg"];
}

def Control : OpGroup<"F", ["lbegin", "loop_end", "flush_slot", "halt", "vmemswap"]>
{
    let vreg_reads = "none";
    let vreg_writes = "none";
    let issue_requirement = "Halt drains all pending VREG producers";
    let opcodes = [1, 2, 3, 4, 5];
    let register_reads = [];
}

/**
 * @brief Stores the occupancy and result latency of one matrix command, which are two separate scheduling inputs.
 *
 * @details A command issued at cycle t leaves the controller busy over t plus one through t plus the occupancy, so
 * the next command becomes legal one cycle after that. The result latency counts from a push to its first result, and
 * a pop issued earlier still reads correct data because VPOP_STREAM gates every beat on mxu_output_valid_i, at the
 * cost of stalling there while it holds the controller. Both figures are measured against the RTL by
 * tb_matrix_full_vreg and the MATRIX_BUSY probe rather than derived.
 *
 * @param command_mnemonic Names the matrix command.
 * @param busy_cycles Sets the cycles the command holds the matrix controller.
 */
class MatrixCommand<string command_mnemonic, int busy_cycles>
{
    /// Stores the matrix command's mnemonic.
    string mnemonic = command_mnemonic;

    /// Stores the cycles the command holds the matrix controller.
    int occupancy_cycles = busy_cycles;

    /// Stores the cycles from the command to its first result leaving the array.
    int result_latency_cycles;

    /// Determines whether the command owns vector register read port A while it streams.
    bit holds_read_port_a = 0;
}

def MatrixLoadCommand : MatrixCommand<"vmatload", 18>
{
    let holds_read_port_a = 1;
}

def MatrixPushCommand : MatrixCommand<"vmatpush", 5>
{
    let result_latency_cycles = 82;
    let holds_read_port_a = 1;
}

def MatrixPopCommand : MatrixCommand<"vmatpop", 7>;

/**
 * @brief Stores one named value belonging to a hardware capacity.
 *
 * @param value_name Names the value, which an AsmConstant refers to when it binds an assembler constant.
 * @param capacity_value Sets the value.
 */
class ResourceValue<string value_name, int capacity_value>
{
    /// Stores the value's name.
    string name = value_name;

    /// Stores the value.
    int value = capacity_value;
}

/**
 * @brief Stores one hardware capacity a schedule can exceed, as distinct from the latencies above.
 *
 * @details These capacities were spread across RTL localparams, compiler constants and comments
 * independently. The MXU output FIFO's depth being known only to the RTL is how a kernel that overflowed it on every
 * step of every block reached silicon and returned plausible wrong numbers.
 *
 * @param group_name Names the capacity, which the published table renders with its underscores replaced by spaces.
 */
class HardwareResource<string group_name>
{
    /// Stores the capacity's name.
    string group = group_name;

    /// Stores the RTL localparam holding the capacity.
    string rtl_parameter = "";

    /// Stores every named value the capacity carries.
    list<ResourceValue> values = [];

    /// Stores the name of the value the published table and the RTL localparam state, or "" for the first value.
    string headline = "";

    /// Stores what exceeding the capacity costs, as the published table words it.
    string note = "";
}

def MatrixOutputFifo : HardwareResource<"mxu_output_fifo">
{
    let rtl_parameter = "MXU_OUTPUT_FIFO_DEPTH";
    let values = [ResourceValue<"depth", 32>,
                  ResourceValue<"entries_per_vmatpush", 4>,
                  ResourceValue<"entries_per_vmatpop", 4>];
    let headline = "depth";
    let note = "One per lane (mxu.sv gen_lane_output). A vmatpush streams NUM_SUBLANES activation rows and so produces that many entries; a vmatpop consumes the same across its masked beats. Overflow drops results silently -- vpu_fifo.sv has no replay path.";
}

def LoopBuffer : HardwareResource<"loop_buffer">
{
    let rtl_parameter = "LB_CAP";
    let values = [ResourceValue<"capacity_bundles", 24>];
    let headline = "capacity_bundles";
    let note = "A body longer than this refetches from IRAM instead of replaying, which bounds replay rather than speed.";
}

def InstructionMemory : HardwareResource<"iram">
{
    let rtl_parameter = "IRAM_DEPTH";
    let values = [ResourceValue<"depth_bundles", 4096>];
    let headline = "depth_bundles";
    let note = "Bundles the loader places, being 2**IRAM_ADDR_WIDTH. A longer image is loaded up to this and its tail never runs, so the drain it ends on leaves the buffer untouched.";
}

def DelayField : HardwareResource<"delay_field">
{
    let rtl_parameter = "DELAY_W";
    let values = [ResourceValue<"width_bits", 6>];
    let headline = "width_bits";
    let note = "One bundle spaces at most 2**width-1 cycles, while a vmatpush result is 82 away, so every push-to-pop gap needs a filler bundle.";
}

def VectorMemory : HardwareResource<"vmem">
{
    let values = [ResourceValue<"rows_per_slot", 8192>,
                  ResourceValue<"slot_count", 2>];
    let note = "Rows one lane holds in one software-owned slot, and the slots `vmemswap` alternates between. Compute and DMA address opposite slots, so a staged region reaching past rows_per_slot wraps inside its own slot rather than faulting and returns whatever the wrap lands on. These are the compute-addressed VMEM slots, which are not the dma_slots a flush mask selects between. No localparam name is quoted here because this tree carries no RTL to hold one against.";
}

def DmaSlots : HardwareResource<"dma_slots">
{
    let rtl_parameter = "DMA_SLOT_COUNT";
    let values = [ResourceValue<"count", 2>];
    let headline = "count";
    let note = "Descriptor slots a flush mask selects between. flush_slot blocks until dma_slot_done covers every bit of its mask and only a descriptor asserts that bit, so flushing a slot the kernel never issued on wedges the device until the PL is reprogrammed.";
}

/**
 * @brief Stores one issue distance the machine leaves to the schedule that no register dependence carries.
 *
 * @details The scheduler prices vector and scalar registers by their writeback offsets. These distances guard state
 * those offsets do not reach, being VMEM rows a store has written and the transpose staging array, and each was
 * verified on the board. An access the rule names issues no earlier than its distance after the access it follows.
 *
 * @param rule_name Names the distance, which a generated constant spells in camel case.
 * @param distance_cycles Sets the distance, in cycles from the first access's issue to the second's.
 */
class SpacingRule<string rule_name, int distance_cycles>
{
    /// Stores the distance's name.
    string name = rule_name;

    /// Stores the distance, in cycles between the two issues.
    int cycles = distance_cycles;
}

/// A vld reading VMEM rows a vst wrote issues this long after the vst. A descriptor draining them needs no distance,
/// since it reaches VMEM later than this after its own issue, which every drained image the board checks relies on.
def VmemTurnaround : SpacingRule<"vmem_turnaround", 9>;

/// Consecutive vtxin issue this far apart.
def TransposeInInterval : SpacingRule<"transpose_in_interval", 2>;

/// The first vtxout issues this long after the last vtxin.
def TransposeInToOut : SpacingRule<"transpose_in_to_out", 6>;

/// Consecutive vtxout issue this far apart.
def TransposeOutInterval : SpacingRule<"transpose_out_interval", 6>;

/// @brief Stores the order every generated artifact lists the records above in.
class EmissionOrder
{
    /// Stores the operation groups, in publication order.
    list<OpGroup> operations = [];

    /// Stores the matrix commands, in publication order.
    list<MatrixCommand> matrix_commands = [];

    /// Stores the hardware capacities, in publication order.
    list<HardwareResource> resources = [];

    /// Stores the issue distances the schedule keeps, in publication order.
    list<SpacingRule> spacing_rules = [];
}

def IsaOrder : EmissionOrder
{
    let operations = [VectorAlu, VectorMove, VectorSfu, VectorReduce, TransposeIn, TransposeOut,
                      VectorLoad, VectorStore,
                      MatrixLoad, MatrixPush, MatrixPop,
                      Scalar, Descriptor, Control];

    let matrix_commands = [MatrixLoadCommand, MatrixPushCommand, MatrixPopCommand];

    let resources = [MatrixOutputFifo, LoopBuffer, InstructionMemory, DelayField, DmaSlots, VectorMemory];

    let spacing_rules = [VmemTurnaround, TransposeInInterval, TransposeInToOut, TransposeOutInterval];
}

// The dialect's operation definitions belong beside the scheduling records they constrain. Adding them makes the file
// need mlir-tblgen for its C++ backends, so the records above stay readable by llvm-tblgen alone and the ISA remains
// one record set.
