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
    src/tree.cu
    src/tree_database.cu
    src/zeldovich.cu
    
                  )      
                       
                        
                        

set(cosmictiger_header_files
	cosmictiger/code_gen.hpp
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
	cosmictiger/map.hpp
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
    OUTPUT cosmictiger/code_gen.hpp
    COMMAND ./code_gen > ../cosmictiger/code_gen.hpp
    DEPENDS code_gen
    COMMENT "Generating FFM kernels"
)
ADD_CUSTOM_TARGET( run_codegen DEPENDS cosmictiger/code_gen.hpp 
                   COMMENT "" )

add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Boost::boost siloh5 mpi  hpxfft cufft
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
 ADD_DEPENDENCIES( cosmictiger run_codegen )
 
 add_executable(
 	hpx2mw
 	src/hpx2mw.cpp
 )
 
add_hpx_executable(
  code_gen
  DEPENDENCIES
  SOURCES
    src/code_gen.cpp
  HEADERS
    ${cosmictiger_header_files}
 )
 
 
target_compile_options(code_gen PUBLIC "-DUSE_HPX")
target_include_directories(code_gen PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpxfft)
target_include_directories(hpxfft PUBLIC ${PROJECT_SOURCE_DIR}/hpxfft)
set_property(TARGET hpxfft PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hpx2mw PUBLIC chealpix siloh5 boost_program_options)
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpxfft)
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount 128 --gpu-architecture=compute_70;
                        >)
target_compile_options(hpxfft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount 128 --gpu-architecture=compute_70;
                        >)
 
 

#SET_SOURCE_FILES_PROPERTIES( src/fourier.cu PROPERTIES COMPILE_FLAGS -G )
#SET_SOURCE_FILES_PROPERTIES( src/gravity.cu PROPERTIES COMPILE_FLAGS -G )
#SET_SOURCE_FILES_PROPERTIES( src/map.cu PROPERTIES COMPILE_FLAGS "--maxrregcount 40" )
#SET_SOURCE_FILES_PROPERTIES( src/tree.cu PROPERTIES COMPILE_FLAGS -G )


 
