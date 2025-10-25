cmake_minimum_required(VERSION 3.17)

project(vega)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Boost REQUIRED COMPONENTS iostreams)
find_package(Eigen3 REQUIRED)

set(XTENSOR_INCLUDE_DIR "/opt/homebrew/opt/xtensor/include")

# Build vegaDataframe as a library
add_library(vegaDataframe
        vegaDataFrame/vegaDataframe.cpp
)

target_include_directories(vegaDataframe PUBLIC
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${XTENSOR_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/vegaDataFrame
)

target_link_libraries(vegaDataframe PUBLIC
        ${Boost_LIBRARIES}
)


# Optional: build standalone vega executable
add_executable(vega
        main.cpp
)

target_link_libraries(vega PRIVATE vegaDataframe)
