CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(hph)

include_directories(. ${CMAKE_SOURCE_DIR}/src ${JNI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/jni
	TBB_INCLUDE_DIRS
	)

set(NOSIMD_SOURCE_FILES
	${CMAKE_SOURCE_DIR}/src/jni/dr_inference_hawkes_NativeHPHSingleton.cpp
	${CMAKE_SOURCE_DIR}/src/factory.cpp
    ${CMAKE_SOURCE_DIR}/src/MemoryManagement.hpp
    ${CMAKE_SOURCE_DIR}/src/AbstractHawkes.hpp
    ${CMAKE_SOURCE_DIR}/src/NewHawkes.hpp
    ${CMAKE_SOURCE_DIR}/src/CDF.h
    ${CMAKE_SOURCE_DIR}/src/Distance.hpp
	)

add_library(hph_nosimd SHARED ${NOSIMD_SOURCE_FILES})
target_link_libraries(hph_nosimd hph_opencl)
target_link_libraries(hph_nosimd ${TBB_LIBRARIES})
set_target_properties(hph_nosimd PROPERTIES COMPILE_FLAGS "-fno-vectorize -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-avx")

add_executable(benchmark-nosimd ${CMAKE_SOURCE_DIR}/src/benchmark.cpp)
target_link_libraries(benchmark-nosimd hph_nosimd)
target_link_libraries(benchmark-nosimd ${TBB_LIBRARIES})
target_link_libraries(benchmark-nosimd boost_program_options)
