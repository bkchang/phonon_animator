CC = g++
CFLAGS = -Wall -g

CUDA_PATH       = /usr/local/cuda-12.1
CUDA_INC_PATH   = $(CUDA_PATH)/include
CUDA_BIN_PATH   = $(CUDA_PATH)/bin
CUDA_LIB_PATH   = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets
NVCC_INCLUDE =
NVCC_CUDA_LIBS = 
NVCC_GENCODES = -gencode arch=compute_50,code=sm_50 \
				-gencode arch=compute_52,code=sm_52 \
				-gencode arch=compute_60,code=sm_60 \
				-gencode arch=compute_61,code=sm_61

CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

all: clean test

test: animation
	./bin/animation

animation: materialsystem.o raytracer.o raytracer.cu.o kernel_utils.cu.o cuda.o
	$(CC) $(CFLAGS) -o ./bin/animation ./src/main.cpp ./obj/materialsystem.o ./obj/raytracer.o ./obj/raytracer.cu.o ./obj/kernel_utils.cu.o ./obj/cuda.o -I$(CUDA_INC_PATH) -L$(CUDA_LIB_PATH) -lcudart

materialsystem.o:
	$(CC) $(CFLAGS) -c ./src/materialsystem.cpp -o ./obj/materialsystem.o

raytracer.o:
	$(CC) $(CFLAGS) -c ./src/raytracer.cpp -o ./obj/raytracer.o

raytracer.cu.o:
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o ./obj/raytracer.cu.o $(NVCC_INCLUDE) ./src/raytracer.cu

kernel_utils.cu.o:
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o ./obj/kernel_utils.cu.o $(NVCC_INCLUDE) ./src/kernel_utils.cu

cuda.o: raytracer.cu.o kernel_utils.cu.o
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o ./obj/cuda.o $(NVCC_INCLUDE) ./obj/raytracer.cu.o ./obj/kernel_utils.cu.o

clean:
	rm -rf ./bin/* ./obj/*.o