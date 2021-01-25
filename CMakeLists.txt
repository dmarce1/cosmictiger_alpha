
cmake_minimum_required(VERSION 3.12.4)
project(cosmictiger CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)
set(source_files
    src/global.cpp
    src/hpx.cpp
    src/main.cpp
    src/options.cpp
    src/particle.cpp
    src/particle_sort.cpp
    src/rand.cpp
    src/tests.cpp
    src/tree.cpp
)
set(header_files
	cosmictiger/cuda.hpp
	cosmictiger/fixed.hpp
	cosmictiger/global.hpp
	cosmictiger/hpx.hpp
	cosmictiger/memory.hpp
	cosmictiger/options.hpp
	cosmictiger/particle.hpp
	cosmictiger/particle_sort.hpp
	cosmictiger/rand.hpp
	cosmictiger/timer.hpp
	cosmictiger/tests.hpp
	cosmictiger/tree.hpp
	cosmictiger/vect3.hpp
)
list(APPEND cosmic_dependencies Silo::silo Boost::boost cuda cudart)
add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    ${cosmic_dependencies}
  SOURCES
    ${source_files}
  HEADERS
    ${header_files}
)
enable_language(CUDA)
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cosmictiger PROPERTY INTERFACE_COMPILE_OPTIONS "${MY_CUDA_FLAGS}")
set_property(TARGET cosmictiger PROPERTY FOLDER "cosmic-Tiger")
target_compile_options(cosmictiger PRIVATE
      -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
      -Wno-unused-result -ffast-math -Wno-write-strings)
target_compile_definitions(cosmictiger PUBLIC COSMICTIGER_HAVE_CUDA)
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cosmictiger PROPERTY INTERFACE_COMPILE_OPTIONS "${MY_CUDA_FLAGS}")
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})

