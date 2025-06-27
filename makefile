run: gemm.cpp
	g++ -O3 -mavx2 -mfma main.cpp gemm.cpp -o a
	./a

.PHONY: clean
clean:
	rm -f a