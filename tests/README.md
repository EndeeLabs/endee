# Tests

This folder contains unit tests for Endee.

## Build & Run

From the repository root:

1. Configure with tests enabled:
   - `cmake -S . -B build -DENABLE_TESTING=ON`
2. Build the test target:
   - `cmake --build build --target ndd_filter_test`
3. Run:
   - `./build/tests/ndd_filter_test`

## Notes

- Tests can also be built in a dedicated tests build directory (e.g., `tests/build/`).
- The `tests/build/` directory is ignored by git.
