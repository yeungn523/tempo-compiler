"""Configures lit for the tempo pipeline's tests: one pass per file, driven through tempo-opt."""

import os

import lit.formats

config.name = "Tempo"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = ["Inputs", "CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.tempo_obj_root

# Points the image tests at the board package's committed images.
config.substitutions.append(("%images", os.path.join(os.path.dirname(config.test_source_root), "..", "board_package", "images")))

config.environment["PATH"] = os.pathsep.join(
    (config.tempo_tools_dir, config.llvm_tools_dir, config.environment.get("PATH", "")))
