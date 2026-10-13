# Core quantization API v1 conformance consumer

`consumer.cpp` is the initial source-compatibility and numerical-contract fixture for
`rabitqlib/core/api.hpp`. It checks signatures, defaults used by a real consumer,
enum values, separate factor storage, both metrics, logical code meaning and the
one-bit packed format. It is not a C ABI fixture or a release-produced index file.

Keep this consumer unchanged when extending v1. Add new consumers for additive APIs;
do not edit away failures caused by an incompatible change. Intentional breaking
changes need a new API major and migration instructions. Do not freeze workspace byte
counts or private class sizes: callers must query sizes and recompile on upgrades.

The C++ suite and external add_subdirectory/FetchContent/installed consumer tests all
compile and execute this same source, including with FastScan and indexes disabled.
