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
find_package(whisper REQUIRED)  # the whisper.cpp you installed in /usr/local

# The reusable module: mic -> noise reduction -> Whisper -> std::string.
# Other code only needs: #include "speech_listener.h" and to link this library.
add_library(speech_listener STATIC src/speech_listener.cpp)
target_include_directories(speech_listener
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src   # lets other targets #include "speech_listener.h"
  PRIVATE ${miniaudio_SOURCE_DIR})          # miniaudio stays hidden inside the module
target_link_libraries(speech_listener PRIVATE whisper Threads::Threads ${CMAKE_DL_LIBS} m)

# The demo program (src/main.cpp)
add_executable(voice_stt src/main.cpp)
target_link_libraries(voice_stt PRIVATE speech_listener)

# Later, your decision tree program can do the same:
#   add_executable(my_assistant src/assistant.cpp src/decision_tree.cpp)
#   target_link_libraries(my_assistant PRIVATE speech_listener)
