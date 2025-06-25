run: gemm.cpp
	g++ -O3 -o a gemm.cpp
	./a

.PHONY: clean
clean:
	rm -f a