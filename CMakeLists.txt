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

set(CMAKE_CXX_FLAGS "-ffast-math -march=native -Wno-deprecated-declarations -Wno-format-security")

set(cosmictiger_source_files
    src/chemistry.cpp
    src/cosmos.cpp
    src/global.cpp
    src/hpx.cpp
    src/main.cpp
    src/options.cpp
    src/rand.cpp
	src/boltzmann.cpp
    src/initial.cpp
    src/math.cpp
    src/zero_order.cpp
)
                        
                       
                        
                        

set(cosmictiger_header_files
	cosmictiger/boltzmann.hpp
	cosmictiger/chemistry.hpp
	cosmictiger/cosmos.hpp
	cosmictiger/domain.hpp
	cosmictiger/defs.hpp
	cosmictiger/direct.hpp
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
	cosmictiger/particle_server.hpp
	cosmictiger/rand.hpp
	cosmictiger/range.hpp
	cosmictiger/stack_vector.hpp
	cosmictiger/timer.hpp
	cosmictiger/tree.hpp
	cosmictiger/fixed.hpp
	cosmictiger/fast_future.hpp
	cosmictiger/global.hpp
    cosmictiger/initial.hpp
	cosmictiger/hpx.hpp
	cosmictiger/multipole.hpp
	cosmictiger/options.hpp
	cosmictiger/particle.hpp
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
    Boost::boost silo mpi tbbmalloc cufft
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
 
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
