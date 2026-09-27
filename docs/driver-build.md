# Optional source build (not needed for the tested x86_64 wheel)

Use only if the pinned wheel fails on your platform/kernel. These commands require an Ubuntu development toolchain and administrator access for OS packages; they have not been executed as a complete build on this machine.

```bash
sudo apt-get install build-essential cmake git pkg-config libusb-1.0-0-dev libudev-dev libssl-dev python3.10-dev
# Use a fresh checkout if research/librealsense does not already exist.
git clone --branch v2.53.1 --depth 1 https://github.com/realsenseai/librealsense.git research/librealsense
cmake -S research/librealsense -B research/librealsense/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_TM2=ON \
  -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build research/librealsense/build --parallel 4
```

Check CMake output: TM2 **must remain ON**. This legacy build downloads additional TM2 resources and can disable support if its network test fails. Old dependencies may need fixes on newer compilers; do not assume a successful generic SDK build contains T265 support.

Use an isolated build-specific capture environment and package the generated Python extension there; do not copy it over the locked wheel or system libraries. Verify `pyrealsense2.__file__`, `ldd` on the extension, enumeration, simultaneous D405/T265 streaming, timestamp domains and a full recording before adopting that build. Keep the same udev rules. A separate modern-D405/legacy-T265 acquisition process design is a possible future fallback, not an implemented or tested mode in this repository.
