
cmake_minimum_required(VERSION 3.10)

project(cuda LANGUAGES CUDA CXX)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

add_executable("${CMAKE_PROJECT_NAME}" bitanicSort.cu)
#include_directories("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include")  # On Windows
target_include_directories("${CMAKE_PROJECT_NAME}" PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")


set_target_properties("${CMAKE_PROJECT_NAME}" PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
