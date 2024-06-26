cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME test_apps)

project(${TARGET_NAME} VERSION "0.0.0" LANGUAGES CXX)

find_package(traact_base)
list(APPEND CMAKE_MODULE_PATH ${traact_base_INCLUDE_DIR})

include(traact_default_library_setup)


find_package(traact_spatial CONFIG REQUIRED)
find_package(traact_vision CONFIG REQUIRED)
find_package(traact_pointcloud CONFIG REQUIRED)
find_package(traact_component_cereal CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)



add_executable(ir_tracking src/main_ir_tracking.cpp )
target_link_libraries(ir_tracking PRIVATE traact_core::traact_core traact_spatial::traact_spatial traact_vision::traact_vision yaml-cpp traact_pointcloud::traact_pointcloud traact_component_cereal::traact_component_cereal)

add_executable(bundle_adjustment src/main_bundle_adjustment.cpp src/BAProblemLoader.cpp src/BAProblemLoader.h src/util/YAMLUtil.h src/util/YAMLUtil.cpp)
target_link_libraries(bundle_adjustment PRIVATE traact_core::traact_core traact_spatial::traact_spatial traact_vision::traact_vision yaml-cpp traact_pointcloud::traact_pointcloud traact_component_cereal::traact_component_cereal)

add_executable(outside_in_generator src/main_outside_in_generator.cpp src/util/YAMLUtil.cpp)
target_link_libraries(outside_in_generator PRIVATE traact_core::traact_core traact_spatial::traact_spatial traact_vision::traact_vision yaml-cpp traact_pointcloud::traact_pointcloud traact_component_cereal::traact_component_cereal)

add_executable(open3d_generator src/main_open3d.cpp src/util/YAMLUtil.cpp)
target_link_libraries(open3d_generator PRIVATE traact_core::traact_core traact_spatial::traact_spatial traact_vision::traact_vision yaml-cpp traact_pointcloud::traact_pointcloud traact_component_cereal::traact_component_cereal)

#include(traact_default_library_final)
