cmake_minimum_required(VERSION 3.5)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.

project(luaort C)			# Название проекта

find_package(Lua REQUIRED)
find_package(PNG REQUIRED)

add_library(luaort SHARED ort.c)
target_include_directories(luaort PRIVATE ${ORT_INCLUDE_DIR})
target_link_directories(luaort PRIVATE ${ORT_DIR})
target_link_libraries(luaort PRIVATE lua onnxruntime)

add_library(pngutils SHARED pngutils.c)
target_link_libraries(pngutils PRIVATE lua PNG::PNG)

add_test(NAME ORTInference COMMAND lua test.lua)