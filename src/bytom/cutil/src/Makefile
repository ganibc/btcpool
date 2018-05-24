.PHONY: clean

TARGET=tensority

$(TARGET): GpuTs.a
	cd .. && go test -v

GpuTs.a: GpuTs.o BytomPoWGpu.o
	ar r $@ $^

%.o: %.cpp
	nvcc -O3 -ccbin g++-6 -o $@ -c $^ -std=c++11 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcublas

%.o: %.cu
	nvcc -O3 -ccbin g++-6 -o $@ -c $^ -std=c++11 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcublas

clean:
	rm -f *.o *.so *.a