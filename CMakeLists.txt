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

#set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -ffast-math -march=native")

set(cosmictiger_source_files
	src/sort.cu
    src/cosmos.cpp
    src/map.cpp
    src/cuda.cpp
    src/drift.cu
    src/drift.cpp
     src/driver.cpp
    src/expansion.cu
    src/ewald_indices.cu
    src/global.cpp
    src/gravity.cpp
    src/gravity.cu
    src/groups.cu
    src/group_data.cpp
    src/groups.cpp
    src/hpx.cpp
    src/main.cpp
    src/memory.cpp
    src/options.cpp
    src/particle.cpp
    src/particle.cu
    src/rand.cpp
    src/tests.cpp
    src/tree.cpp
    src/tree.cu
    src/kick_return.cu
    src/tree_database.cu
    src/kick_return.cpp
)
set(cosmictiger_header_files
	cosmictiger/cosmos.hpp
	cosmictiger/sort.hpp
	cosmictiger/drift.hpp
	cosmictiger/ewald_indices.hpp
	cosmictiger/kick_return.hpp
	cosmictiger/defs.hpp
	cosmictiger/green_direct.hpp
	cosmictiger/green_ewald.hpp
	cosmictiger/groups.hpp
	cosmictiger/gravity.hpp
	cosmictiger/math.hpp
	cosmictiger/managed_vector.hpp
	cosmictiger/multipole.hpp
	cosmictiger/options.hpp
	cosmictiger/particle.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/stack_vector.hpp
	cosmictiger/timer.hpp
	cosmictiger/tree.hpp
	cosmictiger/cuda.hpp
	cosmictiger/fixed.hpp
	cosmictiger/fast_future.hpp
	cosmictiger/global.hpp
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
)


add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Boost::boost chealpix cuda cudart tcmalloc
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
 add_executable(
 	hpx2mw
 	src/hpx2mw.cpp
 )
 
#add_library(hpx_lite 
#	hpx_lite/src/hpx.cpp
#	hpx_lite/src/id_type.cpp
#	hpx_lite/src/thread.cpp
#	hpx_lite/src/future.cpp
#	hpx_lite/src/mutex.cpp
#	hpx_lite/src/serialize.cpp
#)


#add_executable( cosmictiger_lite
#    ${cosmictiger_source_files} ${cosmictiger_header_files}
#)

 
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hpx2mw PUBLIC chealpix siloh5 boost_program_options)
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite)
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math --maxrregcount 128 --gpu-architecture=compute_86;
                        >)
                        
                        
                        
