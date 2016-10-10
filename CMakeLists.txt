cmake_minimum_required(VERSION 2.8)
project( hsv_gpu )
#find_package( Threads REQUIRED)
find_package( OpenCV REQUIRED)
find_package( CUDA REQUIRED )
cuda_compile( HSV_GPU_O hsv_gpu.cu)
add_executable(
	hsv_gpu
	main.cpp
#	hsv_gpu.cu
#	hcsr04.cpp
#	jetsonGPIO.c
#	JHPWMPCA9685.cpp
#	line_element.cpp
#	DETECTED_LINE.cpp
#	CAR_table.cpp
	${HSV_GPU_O}
	)
target_link_libraries(hsv_gpu
	# ${CMAKE_THREAD_LIBS_INIT}
	${OpenCV_LIBS} ) 
