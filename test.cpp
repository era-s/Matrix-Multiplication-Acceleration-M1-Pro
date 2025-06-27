#include <iostream>
#include <immintrin.h>

int main() {
    std::cout << "start" << std::endl;
    alignas(64) float output[16];  // AVX-512는 64바이트 정렬 필요
    __m512 vec = _mm512_set1_ps(3.14f); // AVX-512 명령어
    _mm512_store_ps(output, vec);      // 메모리에 실제로 저장함

    std::cout << "AVX-512 executed, output[0] = " << output[0] << std::endl;
    return 0;
}