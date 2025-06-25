run: gemm.cpp
	g++ -O3 main.cpp gemm.cpp -o a
	./a

.PHONY: clean
clean:
	rm -f a