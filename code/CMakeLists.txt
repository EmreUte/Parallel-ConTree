cmake_minimum_required(VERSION 3.9)
project(ConTree)

set(CMAKE_CXX_STANDARD 17)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Build type not specified: Use Release by default")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Compiling in release mode")
else()
    message(STATUS "Compiling in debug mode")    
    add_compile_definitions(DEBUG=1)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    
    if (MSVC)
        message(STATUS "Using MSVC compiler")
        add_compile_options(/Ot)
    else()
        add_compile_options(-O3)
        add_compile_options(-march=native)
        add_compile_options(-funroll-loops)
    endif()

endif()
   
    

# Gather source files
file(GLOB SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# Initialize SOURCES with C++ files
set(SOURCES ${SRC_FILES})

# Check for CUDA availability
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # Add GPU source files
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/gpu_solver.cu)
    
    message(STATUS "CUDA definitions found. Compiling with GPU support.")
else()
    message(WARNING "CUDA NOT found. Compiling without GPU support.")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create the executable
add_executable(ConTree ${SOURCES})

target_include_directories(ConTree PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/include
)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(ConTree PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(ConTree PUBLIC CUDA::cudart)
    set_target_properties(ConTree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(ConTree PROPERTIES CUDA_ARCHITECTURES "89")
endif()

# Set target properties based on build type
if (CMAKE_BUILD_TYPE STREQUAL Release)        
    if( ipo_supported )
	    message(STATUS "IPO / LTO enabled")    
        set_property(TARGET ConTree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
         message(STATUS "IPO / LTO not supported: <${ipo_error}>")
    endif()
else() 
    
    # enable runtime profilling (with MSVC)
    if (MSVC)
        message(STATUS "Enabling profiler")    
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
    else ()
        message(STATUS "No profiler")
    endif()
endif()

