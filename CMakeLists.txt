set(target topologicalInterpolation)
cmake_minimum_required(VERSION 3.0.0)
option(WITH_TBB "Enable Threading Building Blocks (TBB)" OFF) # ON or OFF parallelization by TBB

if(WITH_TBB)
  find_package(TBB REQUIRED)
  add_definitions(-DUSE_TBB)
endif()

find_package(OpenCV REQUIRED)

add_executable(${target} main.cpp)
target_link_libraries(${target} ${OpenCV_LIBS})
if(WITH_TBB)
  target_link_libraries(${target} TBB::tbb)
endif()

target_include_directories(${target} PRIVATE incl)
target_sources(${target} 
    PRIVATE 
        src/computeGrid.cpp 
        src/triangularReconstruction.cpp 
        src/topologicalReconstruction.cpp
        src/interpolationMethods.cpp
)