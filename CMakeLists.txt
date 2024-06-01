cmake_minimum_required(VERSION 3.5)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.

project(luaort C)			# Название проекта

add_library(luaort SHARED src/ort.c)

find_package(Lua REQUIRED)

target_link_directories(luaort PRIVATE ${ORT_DIR})
target_include_directories(luaort PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(luaort PRIVATE lua onnxruntime)

install(TARGETS luaort
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})