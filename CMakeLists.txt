cmake_minimum_required(VERSION 3.5)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.

project(luaort C)			# Название проекта

find_package(lua REQUIRED)
find_package(PNG REQUIRED)

add_library(luaort SHARED ort.c)	# Создание статической библиотеки с именем foo
target_include_directories(luaort PRIVATE "D:\\onnxruntime\\include\\onnxruntime\\core\\session") #"D:\\onnxruntime-training-win-x64-1.16.3\\include"
target_link_directories(luaort PRIVATE "D:\\onnxbuild\\Release\\Release") #"D:\\onnxruntime-training-win-x64-1.16.3\\lib"
target_link_libraries(luaort PRIVATE lua onnxruntime)

add_library(pngutils SHARED pngutils.c)	# Создание статической библиотеки с именем foo
target_link_libraries(pngutils PRIVATE lua PNG::PNG) # TODO libpng16 не находит