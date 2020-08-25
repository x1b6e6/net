cmake_minimum_required(VERSION 3.15)
project(net VERSION 0.1.0 LANGUAGES CXX)

add_library(net INTERFACE)

find_package(OpenMP)
if(OpenMP_FOUND)
	target_link_libraries(net INTERFACE OpenMP::OpenMP_CXX)
endif()

target_include_directories(net INTERFACE 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
$<INSTALL_INTERFACE:.>)

add_library(net::net ALIAS net)

# vim: set ts=4 sw=4 :
