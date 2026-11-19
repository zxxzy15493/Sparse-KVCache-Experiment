cmake_minimum_required(VERSION 3.26)
project(quest_native_batch LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
                OUTPUT_VARIABLE RUNTIME_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT DEFINED QUEST_ROOT)
  get_filename_component(QUEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(FMT_LIBRARY fmt PATHS "${RUNTIME_PREFIX}/lib" REQUIRED)
find_library(SPDLOG_LIBRARY spdlog PATHS "${RUNTIME_PREFIX}/lib" REQUIRED)
add_library(_kernels_native_batch MODULE csrc/bindings.cpp csrc/page.cu csrc/estimate.cu
            csrc/topk.cu csrc/approx_attn.cu csrc/batch_prefill.cu csrc/rms_norm.cu)
target_include_directories(_kernels_native_batch PRIVATE csrc include ${TORCH_INCLUDE_DIRS}
  "${QUEST_ROOT}/kernels/3rdparty/flashinfer/include"
  "${QUEST_ROOT}/kernels/3rdparty/raft/cpp/include" "${RUNTIME_PREFIX}/include")
target_compile_definitions(_kernels_native_batch PRIVATE SPDLOG_FMT_EXTERNAL FMT_HEADER_ONLY LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
target_compile_options(_kernels_native_batch PRIVATE -O3
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
set_property(TARGET _kernels_native_batch PROPERTY COMPILE_FLAGS "${TORCH_CXX_FLAGS}")
target_link_libraries(_kernels_native_batch PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
                      Python::Python ${FMT_LIBRARY} ${SPDLOG_LIBRARY})
set_target_properties(_kernels_native_batch PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION OFF)
