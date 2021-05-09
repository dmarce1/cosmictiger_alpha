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
find_package(Silo REQUIRED)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "-Wno-attributes -ffast-math -march=native -Wno-deprecated-declarations")

set(cosmictiger_source_files
    src/chemistry.cpp
    src/cosmos.cpp
    src/cuda.cpp
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
    src/particle_sets.cpp
    src/rand.cpp
    src/sph.cpp
    src/hydro_particle.cpp
    src/tests.cpp
    src/tree.cpp
    src/tree_database.cpp
	src/boltzmann.cu
    src/ewald_indices.cu
    src/expansion.cu
    src/fourier.cu
    src/group_data.cu
    src/groups.cu
    src/gravity.cu
    src/initial.cu
    src/kick_return.cu
    src/map.cu
    src/math.cu
    src/particle.cu
    src/power.cu
    src/sph.cu
    src/tree.cu
    src/tree_database.cu
    src/zeldovich.cu
    src/zero_order.cu
)
                        
                       
                        
                        

set(cosmictiger_header_files
	cosmictiger/boltzmann.hpp
	cosmictiger/chemistry.hpp
	cosmictiger/cosmos.hpp
	cosmictiger/defs.hpp
	cosmictiger/drift.hpp
	cosmictiger/ewald_indices.hpp
	cosmictiger/interp.hpp
	cosmictiger/kick_return.hpp
	cosmictiger/fourier.hpp
	cosmictiger/green_direct.hpp
	cosmictiger/green_ewald.hpp
	cosmictiger/groups.hpp
	cosmictiger/gravity.hpp
	cosmictiger/math.hpp
	cosmictiger/multipole.hpp
	cosmictiger/options.hpp
	cosmictiger/particle.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/sph.hpp
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
	cosmictiger/particle_server.hpp
	cosmictiger/particle_sets.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/timer.hpp
	cosmictiger/tests.hpp
	cosmictiger/map.hpp
	cosmictiger/hydro_particle.hpp
	cosmictiger/tree_database.hpp
	cosmictiger/tree.hpp
	cosmictiger/zeldovich.hpp
)


add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    siloh5  cuda cudart tcmalloc boost_program_options
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
 add_executable(
 	hpx2mw
 	src/hpx2mw.cpp
 )
 
add_library(hpx_lite 
	hpx_lite/src/hpx.cpp
	hpx_lite/src/id_type.cpp
	hpx_lite/src/thread.cpp
	hpx_lite/src/future.cpp
	hpx_lite/src/mutex.cpp
	hpx_lite/src/serialize.cpp
)


add_executable( cosmictiger_lite
    ${cosmictiger_source_files} ${cosmictiger_header_files}
)

 
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hpx2mw PUBLIC chealpix siloh5 boost_program_options)
target_link_libraries(cosmictiger_lite PUBLIC siloh5 boost_program_options tcmalloc hpx_lite hwloc)
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(hpx_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite/)
target_include_directories(cosmictiger_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite/)
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount 128 --gpu-architecture=compute_86;
                        >)
target_compile_options(cosmictiger_lite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -dc --use_fast_math --maxrregcount 128 --gpu-architecture=compute_86;
                        >)

#SET_SOURCE_FILES_PROPERTIES( src/math.cu PROPERTIES COMPILE_FLAGS -G )
#SET_SOURCE_FILES_PROPERTIES( src/gravity.cu PROPERTIES COMPILE_FLAGS -G )
#SET_SOURCE_FILES_PROPERTIES( src/gravity.cu PROPERTIES COMPILE_FLAGS -G )
#SET_SOURCE_FILES_PROPERTIES( src/map.cu PROPERTIES COMPILE_FLAGS "--maxrregcount 40" )
#SET_SOURCE_FILES_PROPERTIES( src/tree.cu PROPERTIES COMPILE_FLAGS -G )
 
