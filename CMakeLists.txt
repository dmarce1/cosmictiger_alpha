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
find_package(Silo REQUIRED)
find_package(Boost REQUIRED)
enable_language(CUDA)

#set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -ffast-math -march=native")

set(cosmictiger_source_files
    src/cuda.cpp
    src/expansion.cu
    src/global.cpp
    src/gravity.cu
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
)
set(cosmictiger_header_files
	tiegrgrav/cosmo.hpp
	cosmictiger/defs.hpp
	cosmictiger/gravity_cuda.hpp
	cosmictiger/gravity.hpp
	cosmictiger/gravity_work.hpp
	cosmictiger/gravity.hpp
	cosmictiger/green.hpp
	cosmictiger/groups.hpp
	cosmictiger/initialize.hpp
	tigergrac/load.hpp
	cosmictiger/map.hpp
	cosmictiger/managed_vector.hpp
	cosmictiger/multipole.hpp
	cosmictiger/output.hpp
	cosmictiger/options.hpp
	cosmictiger/part_vect.hpp
	cosmictiger/particle.hpp
	cosmictiger/position.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/stack_vector.hpp
	cosmictiger/taylor.hpp
	cosmictiger/time.hpp
	cosmictiger/tree.hpp
	cosmictiger/vect.hpp
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
	cosmictiger/tree.hpp
)


add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Silo::silo Boost::boost
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
   set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
 target_link_libraries(cosmictiger PUBLIC cuda cudart tcmalloc)
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --maxrregcount 168 --use_fast_math --gpu-architecture=compute_75;
                        >)
                        
                        
                        
