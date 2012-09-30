NVCC = nvcc

main.o: main.cu kernel.cuh
	$(NVCC) -c %< -o $@

kernel.o: kernel.cu
	$(NVCC) -c %< -o $@

main: main.o kernel.o
    $(NVCC) %^ -o $@