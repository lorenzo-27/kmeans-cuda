cmake_minimum_required(VERSION 3.30)
project(kmeans_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

add_executable(kmeans_cuda main.cu)

set_target_properties(kmeans_cuda PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
