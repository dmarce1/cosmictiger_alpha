# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)

# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)

project(cosmictiger CXX C)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
#find_package(Silo REQUIRED)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "-ffast-math -march=native -Wno-deprecated-declarations -Wno-format-security")
set(CMAKE_CUDA_FLAGS "-std=c++14")
set(COSMICTIGER_WITH_FMM_ORDER "6" CACHE STRING "FMM Order")


execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/power.init .)
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")


set(cosmictiger_source_files
    src/cosmos.cpp
    src/cuda.cpp
    src/direct.cpp
    src/domain.cpp
    src/drift.cpp
    src/driver.cpp
    src/ewald_indices.cpp
    src/global.cpp
    src/gravity.cpp
    src/group_data.cpp
    src/groups.cpp
    src/hpx.cpp
    src/kick_return.cpp
    src/main.cpp
    src/map.cpp
    src/memory.cpp
    src/options.cpp
    src/particle.cpp
    src/particle_server.cpp
    src/power.cpp
    src/rand.cpp
    src/rockstar.cpp
    src/tensor.cpp
    src/tests.cpp
    src/tree.cpp
    src/tree_database.cpp
    src/zeldovich.cpp
	src/direct.cu
    src/ewald_indices.cu
    src/expansion.cu
    src/group_data.cu
    src/groups.cu
    src/gravity.cu
    src/initial.cu
    src/kick_return.cu
    src/map.cu
    src/math.cu
    src/particle.cu
    src/power.cu
    src/rockstar.cu
    src/tree.cu
    src/tree_database.cu
    src/zeldovich.cu
    
                  )      
                       
                        
                        

set(cosmictiger_header_files
	cosmictiger/fmm_kernels.hpp
	cosmictiger/cosmos.hpp
	cosmictiger/domain.hpp
	cosmictiger/defs.hpp
	cosmictiger/direct.hpp
	cosmictiger/drift.hpp
	cosmictiger/ewald_indices.hpp
	cosmictiger/interp.hpp
	cosmictiger/kick_return.hpp
	cosmictiger/green_direct.hpp
	cosmictiger/green_ewald.hpp
	cosmictiger/groups.hpp
	cosmictiger/gravity.hpp
	cosmictiger/map.hpp
	cosmictiger/math.hpp
	cosmictiger/multipole.hpp
	cosmictiger/options.hpp
	cosmictiger/particle.hpp
	cosmictiger/particle_server.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/stack_vector.hpp
	cosmictiger/timer.hpp
	cosmictiger/tree.hpp
	cosmictiger/cuda.hpp
	cosmictiger/fixed.hpp
	cosmictiger/fast_future.hpp
	cosmictiger/global.hpp
    cosmictiger/initial.hpp
	cosmictiger/hpx.hpp
	cosmictiger/memory.hpp
	cosmictiger/multipole.hpp
	cosmictiger/options.hpp
	cosmictiger/particle.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/timer.hpp
	cosmictiger/tests.hpp
	cosmictiger/rockstar.hpp
	cosmictiger/tree_database.hpp
	cosmictiger/tree.hpp
	cosmictiger/zeldovich.hpp
)


 add_hpx_library(
 	hpxfft
  DEPENDENCIES
  SOURCES
	hpxfft/src/fourier.cu
 	hpxfft/src/cuda.cu
	hpxfft/src/fourier.cpp
  HEADERS
	hpxfft/hpxfft/fourier.hpp
	hpxfft/hpxfft/cuda.hpp
)


set_property(TARGET hpxfft PROPERTY FOLDER "cosmictiger")

 add_custom_command(
    OUTPUT cosmictiger/fmm_kernels.hpp
    COMMAND ./fmm_kernel_codegen > ${PROJECT_SOURCE_DIR}/cosmictiger/fmm_kernels.hpp
    DEPENDS fmm_kernel_codegen
    COMMENT "Generating FFM kernels"
)

ADD_CUSTOM_TARGET( generate_fmm_kernels DEPENDS cosmictiger/fmm_kernels.hpp 
                   COMMENT "" )

add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Boost::boost silo mpi  hpxfft cufft
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
message(STATUS "CosmicTiger FMM Order: ${COSMICTIGER_WITH_FMM_ORDER}")
target_compile_definitions(cosmictiger PUBLIC COSMICTIGER_FMM_ORDER=${COSMICTIGER_WITH_FMM_ORDER})
 
 ADD_DEPENDENCIES( cosmictiger generate_fmm_kernels )
 
 add_executable(
 	hpx2mw
 	src/hpx2mw.cpp
 )
 
add_hpx_executable(
  fmm_kernel_codegen
  DEPENDENCIES
  SOURCES
    src/fmm_kernel_codegen.cpp
  HEADERS
    ${cosmictiger_header_files}
 )
target_compile_definitions(fmm_kernel_codegen PUBLIC COSMICTIGER_FMM_ORDER=${COSMICTIGER_WITH_FMM_ORDER})
 
 
target_compile_options(fmm_kernel_codegen PUBLIC "-DUSE_HPX")
target_compile_options(hpxfft PUBLIC "-DUSE_HPX")
target_include_directories(fmm_kernel_codegen PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpxfft)
target_include_directories(hpxfft PUBLIC ${PROJECT_SOURCE_DIR}/hpxfft)
set_property(TARGET hpxfft PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hpx2mw PUBLIC chealpix siloh5 boost_program_options)
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpxfft)
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount 128 ;
                        >)
target_compile_options(hpxfft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount 128 ;
                        >)
set_property(TARGET cosmictiger PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET hpxfft PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})


add_executable( cosmictiger_lite
    ${cosmictiger_source_files} ${cosmictiger_header_files}
)

 add_library(
 	hpxfft_lite
	hpxfft/src/fourier.cu
 	hpxfft/src/cuda.cu
	hpxfft/src/fourier.cpp
	hpxfft/hpxfft/fourier.hpp
	hpxfft/hpxfft/cuda.hpp
)


add_library(hpx_lite 
	hpx_lite/src/hpx.cpp
	hpx_lite/src/id_type.cpp
	hpx_lite/src/thread.cpp
	hpx_lite/src/future.cpp
	hpx_lite/src/mutex.cpp
	hpx_lite/src/serialize.cpp
)
target_compile_options( hpx_lite PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
target_compile_options( hpxfft_lite PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
target_compile_options( cosmictiger_lite PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
set_property(TARGET cosmictiger_lite PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(hpx_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite  ${PROJECT_SOURCE_DIR}/hpxfft)
target_include_directories(hpxfft_lite PUBLIC ${PROJECT_SOURCE_DIR}/hpxfft ${PROJECT_SOURCE_DIR}/hpx_lite)
target_link_libraries(cosmictiger_lite PUBLIC  cuda cudart boost_program_options hpx_lite hwloc mpi tcmalloc hpxfft_lite cufft )
target_include_directories(cosmictiger_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite)
target_compile_definitions(cosmictiger_lite PUBLIC COSMICTIGER_FMM_ORDER=${COSMICTIGER_WITH_FMM_ORDER})
target_compile_options(cosmictiger_lite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount 128 -std=c++14;
                        >)
set_property(TARGET cosmictiger_lite PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET hpxfft_lite PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
 
