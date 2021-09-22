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

project(cosmictiger CXX C Fortran)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
#find_package(Silo REQUIRED)

set(CMAKE_CXX_FLAGS "-ffast-math -march=native -Wno-deprecated-declarations -Wno-format-security")

set(cosmictiger_source_files
    src/chemistry.cpp
    src/global.cpp
    src/main.cpp
    src/options.cpp
    src/boltzmann.cpp
    src/initial.cpp
    src/math.cpp
    src/zero_order.cpp
)
                        
                       
                        
                        

set(cosmictiger_header_files
		cosmictiger/boltzmann.hpp  
		cosmictiger/chemistry.hpp  
		cosmictiger/constants.hpp  
		cosmictiger/defs.hpp  
		cosmictiger/global.hpp  
		cosmictiger/initial.hpp  
		cosmictiger/interp.hpp  
		cosmictiger/math.hpp  
		cosmictiger/options.hpp  
		cosmictiger/zero_order.hpp
)


add_executable(
  cosmictiger
  ${cosmictiger_source_files}
  ${cosmictiger_header_files}
 )
 
 add_executable( recfast recfast/recfast.f )
 
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(cosmictiger PUBLIC "-DUSE_HPX")
target_link_libraries(cosmictiger PUBLIC boost_program_options)