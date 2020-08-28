cmake_minimum_required(VERSION 3.15)
project(net VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_CXX_STANDARD LESS 20)
	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_library(net INTERFACE)

target_include_directories(net INTERFACE 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
$<INSTALL_INTERFACE:.>)

add_library(net::net ALIAS net)

enable_testing()
add_subdirectory(test)

# vim: set ts=4 sw=4 :
