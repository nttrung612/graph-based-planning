cmake_minimum_required(VERSION 3.0)
project(c__)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE "Release" CACHE STRING
#       "Choose the type of build, options are: Debug Release
# RelWithDebInfo MinSizeRel."
#       FORCE)
find_package(MPI REQUIRED)

set(COMMON_ENV_SOURCES
        src/discrete_environment.cpp
        src/random_utils.cpp
        src/envs/frozen_lake.cpp
        src/envs/factored_river_swim.cpp
        src/envs/four_rooms.cpp
        src/envs/passenger_grid.cpp
        src/envs/sysadmin_ring.cpp)

add_executable(dummy
        main.cpp
        ${COMMON_ENV_SOURCES})
target_include_directories(dummy PUBLIC include)
target_include_directories(dummy PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(dummy ${MPI_LIBRARIES})

add_executable(testmcgraph
        testmcgraph.cpp
        ${COMMON_ENV_SOURCES})
target_include_directories(testmcgraph PUBLIC include)
target_include_directories(testmcgraph PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(testmcgraph ${MPI_LIBRARIES})
