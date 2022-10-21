cmake_minimum_required(VERSION 3.20)
project(torchCpp)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(Torch_DIR ${LIB_TORCH_ROOT}/share/cmake/Torch)
find_package(Torch REQUIRED)

if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "include path: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "libraries path: ${TORCH_LIBRARIES}")
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(torchCpp main.cpp)
target_link_libraries(torchCpp "${TORCH_LIBRARIES}")
