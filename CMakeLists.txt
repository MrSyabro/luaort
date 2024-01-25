cmake_minimum_required(VERSION 3.5)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.

project(luaort C)			# Название проекта

find_package(lua CONFIG REQUIRED)
find_package(libpng CONFIG REQUIRED)

add_library(luaort SHARED ort.c)	# Создание статической библиотеки с именем foo
target_include_directories(luaort PRIVATE ${LUA_INCLUDE_DIR} "D:\\onnxruntime-training-win-x64-1.16.3\\include")
target_link_directories(luaort PRIVATE "D:\\onnxruntime-training-win-x64-1.16.3\\lib")
target_link_libraries(luaort PRIVATE ${LUA_LIBRARIES} onnxruntime)

add_library(pngutils SHARED pngutils.c)	# Создание статической библиотеки с именем foo
target_link_libraries(pngutils PRIVATE lua libpng16) # TODO libpng16 не находит