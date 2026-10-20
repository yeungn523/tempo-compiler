/**
 * @file
 *
 * @brief Provides minitpu-tblgen, the TableGen driver carrying the MiniTPU's own backends over codegen/minitpuisa.td.
 *
 * -gen-minitpu-sched writes the scheduling facts every pass prices bundles against: each mnemonic's slot, opcode,
 * writeback window, register roles and read-port claim, each matrix command's occupancy and result latency, every
 * hardware capacity, and every issue distance no register dependence carries. The output is a self-contained header, so
 * a fact moves in codegen/minitpuisa.td alone.
 */

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>
#include <algorithm>
#include <string>

using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;

namespace
{
    /// Stores the value the generated tables carry for a field the records leave unset.
    constexpr int64_t kUnset = -1;

    /// Returns `name` as a constant identifier stem, mxu_output_fifo becoming MxuOutputFifo.
    std::string CamelCase(const StringRef name)
    {
        std::string result;
        bool capitalize = true;
        for (const char character : name)
        {
            if (character == '_')
            {
                capitalize = true;
                continue;
            }
            result += capitalize ? llvm::toUpper(character) : character;
            capitalize = false;
        }
        return result;
    }

    /// Returns an integer field of `record`, or kUnset when the records leave it unset.
    int64_t OptionalInt(const Record& record, const StringRef field)
    {
        return record.isValueUnset(field) ? kUnset : record.getValueAsInt(field);
    }

    /// Returns the one record deriving from EmissionOrder, which fixes the order every table lists its rows in.
    const Record& FindEmissionOrder(const RecordKeeper& records)
    {
        const llvm::ArrayRef<const Record*> orders = records.getAllDerivedDefinitions("EmissionOrder");
        if (orders.size() != 1)
        {
            llvm::PrintFatalError(
                "Unable to order the scheduling facts. The records must define exactly one EmissionOrder, but define " +
                llvm::Twine(orders.size()) + "."
            );
        }
        return *orders.front();
    }

    /// Writes the struct and table holding every mnemonic's slot, opcode, writeback window and register roles.
    void EmitOperations(const Record& order, raw_ostream& output)
    {
        const std::vector<const Record*> groups = order.getValueAsListOfDefs("operations");
        size_t widest_reads                     = 1;
        size_t mnemonic_count                   = 0;
        for (const Record* group : groups)
        {
            widest_reads = std::max(widest_reads, group->getValueAsListOfStrings("register_reads").size());
            mnemonic_count += group->getValueAsListOfStrings("mnemonics").size();
        }

        output
            << "    /// Stores the most register roles one operation reads.\n"
            << "    static constexpr size_t kMaxRegisterReads = " << widest_reads << ";\n\n"
            << "    /// Stores the scheduling facts of one mnemonic.\n"
            << "    struct OperationFacts\n    {\n"
            << "        std::string_view mnemonic;  ///< The mnemonic.\n"
            << "        kSlot slot;  ///< The bundle slot the operation issues in.\n"
            << "        int32_t opcode;  ///< The opcode, or -1 for the groups the store bit separates.\n"
            << "        int32_t writeback_first;  ///< The first cycle the write port is reserved, or -1.\n"
            << "        int32_t writeback_last;  ///< The last cycle the write port is reserved, or -1.\n"
            << "        std::array<std::string_view, kMaxRegisterReads> register_reads;  ///< The roles read.\n"
            << "        std::string_view register_write;  ///< The role written, or empty.\n"
            << "        bool claims_read_port_a;  ///< Determines whether the operation claims read port A.\n"
            << "    };\n\n"
            << "    /// Stores every mnemonic's scheduling facts, in the order codegen/minitpuisa.td publishes them.\n"
            << "    static constexpr std::array<OperationFacts, " << mnemonic_count << "> kOperations {{\n";

        for (const Record* group : groups)
        {
            const std::vector<StringRef> mnemonics = group->getValueAsListOfStrings("mnemonics");
            const std::vector<int64_t> opcodes     = group->getValueAsListOfInts("opcodes");
            const std::vector<StringRef> reads     = group->getValueAsListOfStrings("register_reads");
            const int64_t first                    = OptionalInt(*group, "writeback_offset");
            const int64_t last =
                group->isValueUnset("writeback_last_offset") ? first : OptionalInt(*group, "writeback_last_offset");
            const StringRef write =
                group->isValueUnset("register_write") ? "" : group->getValueAsString("register_write");
            for (size_t index = 0; index < mnemonics.size(); ++index)
            {
                output << "        {\"" << mnemonics[index] << "\", kSlot::k" << group->getValueAsString("slot") << ", "
                       << (index < opcodes.size() ? opcodes[index] : kUnset) << ", " << first << ", " << last << ", {";
                for (size_t role = 0; role < reads.size(); ++role)
                {
                    output << (role == 0 ? "" : ", ") << "\"" << reads[role] << "\"";
                }
                output << "}, \"" << write << "\", " << (group->getValueAsBit("claims_read_port_a") ? "true" : "false")
                       << "},\n";
            }
        }
        output << "    }};\n\n";
    }

    /// Writes the struct and table holding each matrix command's occupancy, result latency and port-A hold.
    void EmitMatrixCommands(const Record& order, raw_ostream& output)
    {
        const std::vector<const Record*> commands = order.getValueAsListOfDefs("matrix_commands");
        output << "    /// Stores the cycles one matrix command holds the controller and waits for its first result.\n"
               << "    struct MatrixCommandFacts\n    {\n"
               << "        std::string_view mnemonic;  ///< The matrix command.\n"
               << "        int32_t occupancy_cycles;  ///< The cycles the command holds the matrix controller.\n"
               << "        int32_t result_latency_cycles;  ///< The cycles to its first result, or -1.\n"
               << "        bool holds_read_port_a;  ///< Determines whether the command owns read port A while busy.\n"
               << "    };\n\n"
               << "    /// Stores every matrix command's facts.\n"
               << "    static constexpr std::array<MatrixCommandFacts, " << commands.size() << "> kMatrixCommands {{\n";
        for (const Record* command : commands)
        {
            output << "        {\"" << command->getValueAsString("mnemonic") << "\", "
                   << command->getValueAsInt("occupancy_cycles") << ", "
                   << OptionalInt(*command, "result_latency_cycles") << ", "
                   << (command->getValueAsBit("holds_read_port_a") ? "true" : "false") << "},\n";
        }
        output << "    }};\n\n";
    }

    /// Writes one constant per hardware capacity, named after its resource and value.
    void EmitResources(const Record& order, raw_ostream& output)
    {
        for (const Record* resource : order.getValueAsListOfDefs("resources"))
        {
            const std::string group = CamelCase(resource->getValueAsString("group"));
            for (const Record* value : resource->getValueAsListOfDefs("values"))
            {
                output << "    /// Stores " << resource->getValueAsString("group") << "'s "
                       << value->getValueAsString("name") << ".\n"
                       << "    static constexpr int32_t k" << group << CamelCase(value->getValueAsString("name"))
                       << " = " << value->getValueAsInt("value") << ";\n\n";
            }
        }
    }

    /// Writes one constant per issue distance the schedule keeps, named after the rule.
    void EmitSpacingRules(const Record& order, raw_ostream& output)
    {
        for (const Record* rule : order.getValueAsListOfDefs("spacing_rules"))
        {
            output << "    /// Stores " << rule->getValueAsString("name") << ", in cycles between two issues.\n"
                   << "    static constexpr int32_t k" << CamelCase(rule->getValueAsString("name"))
                   << "Cycles = " << rule->getValueAsInt("cycles") << ";\n\n";
        }
    }

    /// Writes the lookups every consumer reaches the tables through.
    void EmitLookups(raw_ostream& output)
    {
        output << "    /// Returns the scheduling facts of a mnemonic, or nullptr when the records name no such "
                  "mnemonic.\n"
               << "    constexpr const OperationFacts* FindOperation(const std::string_view mnemonic)\n    {\n"
               << "        for (const OperationFacts& facts : kOperations)\n        {\n"
               << "            if (facts.mnemonic == mnemonic) return &facts;\n        }\n"
               << "        return nullptr;\n    }\n\n"
               << "    /// Returns the facts of a matrix command, or nullptr when the records name no such command.\n"
               << "    constexpr const MatrixCommandFacts* FindMatrixCommand(const std::string_view mnemonic)\n    {\n"
               << "        for (const MatrixCommandFacts& facts : kMatrixCommands)\n        {\n"
               << "            if (facts.mnemonic == mnemonic) return &facts;\n        }\n"
               << "        return nullptr;\n    }\n";
    }

    /// Writes every scheduling fact codegen/minitpuisa.td publishes as one self-contained C++ header.
    void EmitSchedule(const RecordKeeper& records, raw_ostream& output)
    {
        llvm::emitSourceFileHeader("MiniTPU scheduling facts, from codegen/minitpuisa.td", output, records);
        const Record& order = FindEmissionOrder(records);
        output << "#include <array>\n#include <cstdint>\n#include <string_view>\n\n"
               << "#include \"minitpu/minitpuisa/encoding.h\"\n\n"
               << "namespace mlir::minitpu::isa::schedule\n{\n";
        EmitOperations(order, output);
        EmitMatrixCommands(order, output);
        EmitResources(order, output);
        EmitSpacingRules(order, output);
        EmitLookups(output);
        output << "}  // namespace mlir::minitpu::isa::schedule\n";
    }

    /// Registers -dump-json, LLVM's own record dump, which tools/toolchain/gen_isa_doc.py renders its artifacts from.
    const llvm::TableGen::Emitter::Opt
        kJsonEmitter("dump-json", llvm::EmitJSON, "Dump all records as machine-readable JSON");

    /// Registers -gen-minitpu-sched with the TableGen driver.
    const llvm::TableGen::Emitter::Opt
        kScheduleEmitter("gen-minitpu-sched", EmitSchedule, "Generate the MiniTPU's scheduling facts");
}  // namespace

/// Hands the command line to the TableGen driver, which runs the backend the flags select.
int main(int argc, char** argv)
{
    llvm::cl::ParseCommandLineOptions(argc, argv);
    return llvm::TableGenMain(argv[0], llvm::MultiFileTableGenMainFn());
}
