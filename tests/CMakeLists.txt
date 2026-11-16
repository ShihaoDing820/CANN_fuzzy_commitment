cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(fuzzy_commitment_standard_kernel_cpu_test LANGUAGES ASC CXX)

add_executable(fuzzy_commitment_standard_kernel_cpu_test
    standard_kernel_cpu_test.asc
)

target_compile_features(fuzzy_commitment_standard_kernel_cpu_test PRIVATE cxx_std_17)
target_compile_options(fuzzy_commitment_standard_kernel_cpu_test PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_include_directories(fuzzy_commitment_standard_kernel_cpu_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../custom_opp/op_kernel
)
target_compile_definitions(fuzzy_commitment_standard_kernel_cpu_test PRIVATE
    WIFAKEY_BG2_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../prototype/wifakey_bg2_set2.tsv"
)
