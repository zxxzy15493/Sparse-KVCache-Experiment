#include <immintrin.h>  // AVX512
#include <stdint.h>
#include<torch/extension.h>
#include<math.h>
#include <cstdio>
#include "fbgemm/FbgemmConvert.h"
#include<immintrin.h>
#include<vector>
#include<omp.h>

// ===================== /（）=====================
// ：LOG2E = 1/ln2 ≈1.4426950408889634，AVX512
static const __m512 LOG2E_VEC = _mm512_set1_ps(1.4426950408889634f);
// ： 2^f ，f ∈ [-0.5, 0.5]（frac_part）
// ：2^f ≈ a0 + a1*f + a2*f² + a3*f³ （4，≈1e-6，）
static const __m512 EXP_POLY_COEFFS[4] = {
    _mm512_set1_ps(1.0000000000f),  // a0
    _mm512_set1_ps(0.6931471805f),  // a1 = ln2
    _mm512_set1_ps(0.2402265069f),  // a2 = (ln2)²/2
    _mm512_set1_ps(0.0555041087f)   // a3 = (ln2)³/6
};

// ===================== avx512_exp_ps（）=====================
__m512 avx512_exp_ps(__m512 x) {
    // Step 1: Scale x to base-2 exponent (x * 1/ln2)，
    __m512 scaled = _mm512_mul_ps(x, LOG2E_VEC);

    // ：_mm512_cvttps_epi32（）→ ：，frac_part ∈ [-0.5, 0.5]
    __m512i int_part = _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 frac_part = _mm512_sub_ps(scaled, _mm512_cvtepi32_ps(int_part)); // ，[-0.5,0.5]

    // float：(1) + (8) + (23)，127，23
    int_part = _mm512_add_epi32(int_part, _mm512_set1_epi32(127));
    int_part = _mm512_slli_epi32(int_part, 23);                    // 23
    __m512 int_exp = _mm512_castsi512_ps(int_part);                // ，2^integer_part

    __m512 poly = EXP_POLY_COEFFS[3];
    poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[2]);
    poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[1]);
    poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[0]);

    __m512 res = _mm512_mul_ps(int_exp, poly); // ：2^k * 2^f = 2^(k+f) = exp(x)
    
    // ：x < -1000（exp(-100)≈3.72e-44，float，）
    __m512 mask = _mm512_cmp_ps(x, _mm512_set1_ps(-100.0f), _CMP_LT_OS); // -1001
    res = _mm512_blendv_ps(res, _mm512_setzero_ps(), mask);             // 0，-inf

    return res;
}

// avx512_exp_ps +  

int main() {
    float test_value[16];
    for(int j = 0; j < 16; j ++ ){
        test_value[j] = -88.945632935f; // f，
    }
    __m512 test_exp = _mm512_loadu_ps(test_value);
    __m512 test_exp_scores = avx512_exp_ps(test_exp);
    float test_exp_scores_copy[16];
    _mm512_storeu_ps(test_exp_scores_copy, test_exp_scores);
    
    printf("\n\n");
    printf("test_exp_scores value is: ");
    for(int j = 0; j < 16; j ++ ){
        printf(" %.9e ", test_exp_scores_copy[j]); // ，（0）
    }
    return 0;
}