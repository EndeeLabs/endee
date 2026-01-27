# Endee: High-Performance Open Source Vector Database

**Endee (nD)** is a specialized, high-performance vector database built for speed and efficiency. This guide covers supported platforms, dependency requirements, and detailed build instructions using both our automated installer and manual CMake configuration.

---

## 1. System Requirements

Before installing, ensure your system meets the following hardware and operating system requirements.

### Supported Operating Systems

* **Linux**: Ubuntu(22.04, 24.04, 25.04) Debian(12, 13), Rocky(8, 9, 10), Centos(8, 9, 10), Fedora(40, 42, 43)
* **macOS**: Apple Silicon (M Series) only.

### Required Dependencies

The following packages are required for compilation.

 `clang-19`, `cmake`, `build-essential`, `libssl-dev`, `libcurl4-openssl-dev`

> **Note:** The build system requires **Clang 19** (or a compatible recent Clang version) supporting C++20.

---

## 2. Quick Installation (Recommended)

The easiest way to build **ndd** is using the included `install.sh` script. This script handles OS detection, dependency checks, and configuration automatically.

### Usage

Run the script from the root of the repository. You **must** provide arguments for the build mode and/or CPU optimization.

```bash
./install.sh [BUILD_MODE] [CPU_OPTIMIZATION]
```

### Build Arguments

You can combine one **Build Mode** and one **CPU Optimization** flag.

#### Build Modes

| Flag | Description | CMake Equivalent |
| --- | --- | --- |
| `--release` | **Default.** Optimized release build. |  |
| `--debug_all` | Enables full debugging symbols. | `-DND_DEBUG=ON -DDEBUG=ON` |
| `--debug_nd` | Enables NDD-specific logging/timing. | `-DND_DEBUG=ON` |

#### CPU Optimization Options

Select the flag matching your hardware to enable SIMD optimizations.

| Flag | Description | Target Hardware |
| --- | --- | --- |
| `--avx2` | Enables AVX2 (FMA, F16C) | Modern x86_64 Intel/AMD |
| `--avx512` | Enables AVX512 (F, BW, VNNI, FP16) | Server-grade x86_64 (Xeon/Epyc) |
| `--neon` | Enables NEON (FP16, DotProd) | Apple Silicon / ARMv8.2+ |
| `--sve2` | Enables SVE2 (INT8/16, FP16) | ARMv9 / SVE2 compatible |

> **Note:** The `--avx512` build configuration enforces mandatory runtime checks for specific instruction sets. To successfully run this build, your CPU must support **`avx512` (Foundation), `avx512_fp16`, `avx512_vnni`, `avx512bw`, and `avx512_vpopcntdq`**; if any of these extensions are missing, the database will fail to initialize and exit immediately to avoid runtime crashes.


### Example Commands

**Build for Production (Intel/AMD with AVX2):**

```bash
./install.sh --release --avx2
```

**Example Build for Debugging (Apple Silicon):**

```bash
./install.sh --debug_all --neon
```

---

## 3. Manual Build (Advanced)

If you prefer to configure the build manually or integrate it into an existing install pipeline, you can use `cmake` directly.

### Step 1: Prepare Build Directory

```bash
mkdir build && cd build
```

### Step 2: Configure

Run `cmake` with the appropriate flags. You must manually define the compiler if it is not your system default.

**Configuration Flags:**

* **Debug Options:**
* `-DDEBUG=ON` (Enable debug symbols/O0)
* `-DND_DEBUG=ON` (Enable internal logging)


* **SIMD Selectors (Choose One):**
* `-DUSE_AVX2=ON`
* `-DUSE_AVX512=ON`
* `-DUSE_NEON=ON`
* `-DUSE_SVE2=ON`


**Example (x86_64 AVX512 Release):**

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_AVX512=ON \
      ..
```

### Step 3: Compile

```bash
make -j$(nproc)
```

---

## 4. Running ndd

After a successful build, the binary will be generated in the `build/` directory.

### Binary Naming

The output binary name depends on the SIMD flag used during compilation:

* `ndd-avx2`
* `ndd-avx512`
* `ndd-neon` (or `ndd-neon-darwin` for mac)
* `ndd-sve2`

### Runtime Environment Variables

Some enviroment variables **ndd** reads at runtime:

* `NDD_DATA_DIR`: Defines the data directory
* `NDD_AUTH_TOKEN`: Optional authentication token (see below)

### Authentication

**ndd** supports two authentication modes:

**Open Mode (No Authentication)** - Default when `NDD_AUTH_TOKEN` is not set:
```bash
# All APIs work without authentication
./build/ndd-avx2
curl http://{{BASE_URL}}/api/v1/index/list
```

**Token Mode** - When `NDD_AUTH_TOKEN` is set:
```bash
# Generate a secure token
export NDD_AUTH_TOKEN=$(openssl rand -hex 32)
./build/ndd-avx2

# All protected APIs require the token in Authorization header
curl -H "Authorization: $NDD_AUTH_TOKEN" http://{{BASE_URL}}/api/v1/index/list
```

### Execution Example

To run the database using the AVX2 binary and a local `data` folder:

```bash
# 1. Create the data directory
mkdir -p ./data

# 2. Export the environment variable and run
export NDD_DATA_DIR=$(pwd)/data
./build/ndd-avx2
```

Alternatively, as a single line:

```bash
NDD_DATA_DIR=./data ./build/ndd-avx2
```

---
## 5. Docker Deployment

We provide a Dockerfile for easy containerization. This ensures a consistent runtime environment and simplifies the deployment process across various platforms.

### Build the Image

You **must** specify the target architecture (`avx2`, `avx512`, `neon`, `sve2`) using the `BUILD_ARCH` build argument. You can optionally enable a debug build using the `DEBUG` argument.

```bash
# Production Build (AVX2) (for x86_64 systems)
docker build --ulimit nofile=100000:100000 --build-arg BUILD_ARCH=avx2 -t endee-oss:latest -f ./infra/Dockerfile .

# Debug Build (Neon) (for arm64, mac apple silicon)
docker build --ulimit nofile=100000:100000 --build-arg BUILD_ARCH=neon --build-arg DEBUG=true -t endee-oss:latest -f ./infra/Dockerfile .
```

### Run the Container

The container exposes port `8080` and stores data in `/data` inside container. You should persist this data using a docker volume.

```bash
docker run \
  -p 8080:8080 \
  -v endee-data:/data \
  --name endee-server \
  endee-oss:latest
```

### Alternatively: Docker Compose

You can also use `docker-compose` to run the service.

1. Start the container:
   ```bash
   docker-compose up
   ```

---

## 6. Running Docker container from registry

You can run Endee directly using the pre-built image from Docker Hub without building locally.

### Using Docker Compose

Create a new directory for Endee:

```bash
mkdir endee && cd endee
```

Inside this directory, create a file named `docker-compose.yml` and copy the following content into it:

```yaml
services:
  endee:
    image: endeeio/endee-server:latest
    container_name: endee-server
    ports:
      - "8080:8080"
    environment:
      NDD_NUM_THREADS: 0
      NDD_AUTH_TOKEN: ""  # Optional: set for authentication
    volumes:
      - endee-data:/data
    restart: unless-stopped

volumes:
  endee-data:
```

Then run:
```bash
docker compose up -d
```

for more details visit [docs.endee.io](https://docs.endee.io/quick-start)

---

## Contribution

We welcome contributions from the community to help make vector search faster and more accessible for everyone. To contribute:

* **Submit Pull Requests**: Have a fix or a new feature? Fork the repo, create a branch, and send a PR.
* **Report Issues**: Found a bug or a performance bottleneck? Open an issue on GitHub with steps to reproduce it.
* **Suggest Improvements**: We are always looking to optimize performance; feel free to suggest new CPU target optimizations or architectural enhancements.
* **Feature Requests**: If there is a specific functionality you need, start a discussion in the issues section.

---

## License

Endee is open source software licensed under the
**Apache License 2.0**.

You are free to use, modify, and distribute this software for
personal, commercial, and production use.

See the LICENSE file for full license terms.

---

## Trademark and Branding

“Endee” and the Endee logo are trademarks of Endee Labs.

The Apache License 2.0 does **not** grant permission to use the Endee name,
logos, or branding in a way that suggests endorsement or affiliation.

If you offer a hosted or managed service based on this software, you must:
- Use your own branding
- Avoid implying it is an official Endee service

For trademark or branding permissions, contact: enterprise@endee.io

---

## Third-Party Software

This project includes or depends on third-party software components that are
licensed under their respective open source licenses.

Use of those components is governed by the terms and conditions of their
individual licenses, not by the Apache License 2.0 for this project.
