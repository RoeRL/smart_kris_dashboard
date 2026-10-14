cmake_minimum_required(VERSION 3.16)
project(voice_stt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# miniaudio is a single-header audio library. FetchContent downloads it at configure time.
# TODO: pin GIT_TAG to a specific release once you pick one.
include(FetchContent)
FetchContent_Declare(
  miniaudio
  GIT_REPOSITORY https://github.com/mackron/miniaudio.git
  GIT_TAG        master
)
FetchContent_GetProperties(miniaudio)
if(NOT miniaudio_POPULATED)
  FetchContent_Populate(miniaudio)
endif()

find_package(Threads REQUIRED)

add_executable(voice_calibration mic_calibration.cpp)
target_include_directories(voic_calibration PRIVATE ${miniaudio_SOURCE_DIR})
target_link_libraries(voice_calibration PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)
