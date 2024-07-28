mask: mask.cpp
	$(CXX) -std=c++17 \
		   -Llib \
		   -o mask \
		   -Iinclude \
		   -Isrc \
		   -lglog \
		   -lvart-mem-manager -lxir -lunilog -lvart-buffer-object -lvart-runner -lvart-util -lvart-xrt-device-handle -lvart-dpu-runner -lvart-dpu-controller -lvart-runner-assistant -lvart-trace \
		   -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
		   -I/usr/include/opencv4 \
		   mask.cpp
