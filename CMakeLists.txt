cmake_minimum_required(VERSION 3.9)
project(creativityproject LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE sources src/*.cpp src/*.cu)
file(GLOB_RECURSE headers src/*.h)

add_executable(${PROJECT_NAME} ${sources} ${headers} src/eigenArt.cu src/eigenArt.h src/cu/kmeans.cu src/cu/kmeans.h src/cu/shared_pointer.cu src/cu/shared_pointer.h)
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
if(EXISTS "/usr/local/include/clion_cuda_syntax.h") # adding syntax highligts to CLion IDE, not required for compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include /usr/local/include/clion_cuda_syntax.h")
endif()

find_package(Boost COMPONENTS filesystem system timer REQUIRED)
find_package(OpenCV COMPONENTS core imgproc imgcodecs highgui REQUIRED )
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-fopenmp")
else()
    message(WARNING "OpenMP not found")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

set(DEPENDENCY_INCLUDES
        ${OpenCV_INCLUDE_PATH}
        ${Boost_INCLUDE_PATH}
)
set(DEPENDENCY_LIBRARIES
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

include_directories(${PROJECT_NAME} ${DEPENDENCY_INCLUDES})
target_link_libraries(${PROJECT_NAME} ${DEPENDENCY_LIBRARIES})