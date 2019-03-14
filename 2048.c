/* This code comes from this paper: Ouyang M.  KNN in the Jaccard
 * Space.  Proceedings of IEEE High Performance Extreme Computing
 * Conference (HPEC), 2016, 1-6.
 *
 * Copyright (c) 2019 Ming Ouyang
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <x86intrin.h>

#define MILLION  1000000L
#define NumVec   268435456     // 2^28
#define VecLen   2048          // vector length in bits
#define NumWord  (VecLen >> 5) // # of unsigned in each vector
#define RandSeed 1283
#define ALIGN    64

uint32_t *data, *cnt1, *cnt2;

void init(void) {
  uint64_t i, j;

  srand48(RandSeed);
  posix_memalign((void **) &data, ALIGN, (size_t) NumVec * (VecLen >> 3));
  for (i = 0; i < NumVec; i++)
    for (j = 0; j < NumWord; j++)
      data[i * NumWord + j] = lrand48();
  posix_memalign((void **) &cnt1, ALIGN, sizeof(uint32_t) * NumVec);
  posix_memalign((void **) &cnt2, ALIGN, sizeof(uint32_t) * NumVec);
}

void verify(void) {
  uint32_t i;

  for (i = 0; i < NumVec; i++)
    if (cnt1[i] != cnt2[i]) {
      printf("mismatch: i %u, cnt1[i] %u, cnt2[i] %u\n", i, cnt1[i], cnt2[i]);
      exit(1);
    }
}

//Intel popcnt intrinsics
void popcnt64(const uint32_t *data, uint32_t *cnt) {
  uint64_t *ptr64, i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr64 = (uint64_t *)data;
  for (i = 0; i < NumVec; i++)
    cnt[i] =
      _mm_popcnt_u64(ptr64[32 * i +  0]) + _mm_popcnt_u64(ptr64[32 * i +  1]) +
      _mm_popcnt_u64(ptr64[32 * i +  2]) + _mm_popcnt_u64(ptr64[32 * i +  3]) +
      _mm_popcnt_u64(ptr64[32 * i +  4]) + _mm_popcnt_u64(ptr64[32 * i +  5]) +
      _mm_popcnt_u64(ptr64[32 * i +  6]) + _mm_popcnt_u64(ptr64[32 * i +  7]) +
      _mm_popcnt_u64(ptr64[32 * i +  8]) + _mm_popcnt_u64(ptr64[32 * i +  9]) +
      _mm_popcnt_u64(ptr64[32 * i + 10]) + _mm_popcnt_u64(ptr64[32 * i + 11]) +
      _mm_popcnt_u64(ptr64[32 * i + 12]) + _mm_popcnt_u64(ptr64[32 * i + 13]) +
      _mm_popcnt_u64(ptr64[32 * i + 14]) + _mm_popcnt_u64(ptr64[32 * i + 15]) +
      _mm_popcnt_u64(ptr64[32 * i + 16]) + _mm_popcnt_u64(ptr64[32 * i + 17]) +
      _mm_popcnt_u64(ptr64[32 * i + 18]) + _mm_popcnt_u64(ptr64[32 * i + 19]) +
      _mm_popcnt_u64(ptr64[32 * i + 20]) + _mm_popcnt_u64(ptr64[32 * i + 21]) +
      _mm_popcnt_u64(ptr64[32 * i + 22]) + _mm_popcnt_u64(ptr64[32 * i + 23]) +
      _mm_popcnt_u64(ptr64[32 * i + 24]) + _mm_popcnt_u64(ptr64[32 * i + 25]) +
      _mm_popcnt_u64(ptr64[32 * i + 26]) + _mm_popcnt_u64(ptr64[32 * i + 27]) +
      _mm_popcnt_u64(ptr64[32 * i + 28]) + _mm_popcnt_u64(ptr64[32 * i + 29]) +
      _mm_popcnt_u64(ptr64[32 * i + 30]) + _mm_popcnt_u64(ptr64[32 * i + 31]);
}

void lookup256(const uint32_t *data, uint32_t *cnt) {
  const __m256i table = _mm256_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m256i mask_0F = _mm256_set1_epi8(0x0F);
  __m256i v, low, high, *ptr, acc;
  uint64_t i, j;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m256i *)data;
  for (i = 0; i < NumVec; i++, ptr += 8){
    acc = _mm256_setzero_si256();
    for (j = 0; j < 8; j++) {
      v    = _mm256_load_si256((__m256i const*) (ptr + j));
      low  = _mm256_and_si256(mask_0F, v);
      high = _mm256_and_si256(mask_0F, _mm256_srli_epi16(v, 4));
      v    = _mm256_add_epi8(_mm256_shuffle_epi8(table, low),
			     _mm256_shuffle_epi8(table, high));
      acc  = _mm256_add_epi64(acc, _mm256_sad_epu8(v, _mm256_setzero_si256()));
    }
    cnt[i] = _mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 1) +
      _mm256_extract_epi64(acc, 2) + _mm256_extract_epi64(acc, 3);
  }
}

#define FULLADD256(cout,sum,cin,b1,b2) {	\
    sum  = _mm256_xor_si256(cin, b1);		\
    cin  = _mm256_and_si256(cin, b1);		\
    cout = _mm256_and_si256(sum, b2);		\
    sum  = _mm256_xor_si256(sum, b2);		\
    cout = _mm256_or_si256(cout, cin);		\
  }

void fullAdd256(const uint32_t *data, uint32_t *cnt) {
  const __m256i table = _mm256_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m256i mask_55 = _mm256_set1_epi8(0x55);
  const __m256i mask_33 = _mm256_set1_epi8(0x33);
  const __m256i mask_0F = _mm256_set1_epi8(0x0F);
  __m256i *ptr, b1, b2, inOut, outIn, c1, c2, c3;
  __m256i tmp, low, high;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m256i *)data;
  for (i = 0; i < NumVec; i++, ptr += 8) {
    inOut = _mm256_load_si256((__m256i const*) (ptr + 0));
    b1    = _mm256_load_si256((__m256i const*) (ptr + 1));
    b2    = _mm256_load_si256((__m256i const*) (ptr + 2));
    FULLADD256(c1, outIn, inOut, b1, b2);

    b1    = _mm256_load_si256((__m256i const*) (ptr + 3));
    b2    = _mm256_load_si256((__m256i const*) (ptr + 4));
    FULLADD256(c2, inOut, outIn, b1, b2);

    b1    = _mm256_load_si256((__m256i const*) (ptr + 5));
    b2    = _mm256_load_si256((__m256i const*) (ptr + 6));
    FULLADD256(c3, outIn, inOut, b1, b2);
    // outIn is the sum
    // c1, c2, and c3 are the carries

    FULLADD256(b2, b1, c1, c2, c3);

    tmp   = _mm256_srli_epi64(outIn, 1);
    tmp   = _mm256_and_si256(tmp, mask_55);
    outIn = _mm256_sub_epi64(outIn, tmp);
    tmp   = _mm256_srli_epi64(b1, 1);
    tmp   = _mm256_and_si256(tmp, mask_55);
    b1    = _mm256_sub_epi64(b1, tmp);
    tmp   = _mm256_srli_epi64(b2, 1);
    tmp   = _mm256_and_si256(tmp, mask_55);
    b2    = _mm256_sub_epi64(b2, tmp);
    // 0--2 in 2-bit subwords

    tmp   = _mm256_srli_epi64(outIn, 2);
    tmp   = _mm256_and_si256(tmp, mask_33);
    outIn = _mm256_and_si256(outIn, mask_33);
    outIn = _mm256_add_epi64(outIn, tmp);
    tmp   = _mm256_srli_epi64(b1, 2);
    tmp   = _mm256_and_si256(tmp, mask_33);
    b1    = _mm256_and_si256(b1, mask_33);
    b1    = _mm256_add_epi64(b1, tmp);
    tmp   = _mm256_srli_epi64(b2, 2);
    tmp   = _mm256_and_si256(tmp, mask_33);
    b2    = _mm256_and_si256(b2, mask_33);
    b2    = _mm256_add_epi64(b2, tmp);
    // 0--4 in 4-bit subwords
   
    b1    = _mm256_slli_epi64(b1, 1);
    outIn = _mm256_add_epi64(outIn, b1);
    // 0--12 in 4-bit subwords

    tmp   = _mm256_srli_epi64(outIn, 4);
    tmp   = _mm256_and_si256(tmp, mask_0F);
    outIn = _mm256_and_si256(outIn, mask_0F);
    outIn = _mm256_add_epi64(outIn, tmp);
    // 0--24 in 8-bit subwords

    tmp   = _mm256_srli_epi64(b2, 4);
    b2    = _mm256_add_epi64(b2, tmp);
    b2    = _mm256_and_si256(b2, mask_0F);
    // 0--8 in 8-bit subwords
    b2    = _mm256_slli_epi64(b2, 2);
    outIn = _mm256_add_epi64(outIn, b2);
    // 0--56 in 8-bit subwords

    outIn = _mm256_sad_epu8(outIn, _mm256_setzero_si256());

    b1    = _mm256_load_si256((__m256i const*) (ptr + 7));
    low   = _mm256_and_si256(mask_0F, b1);
    high  = _mm256_and_si256(mask_0F, _mm256_srli_epi16(b1, 4));
    b1    = _mm256_add_epi8(_mm256_shuffle_epi8(table, low),
			   _mm256_shuffle_epi8(table, high));

    b1 = _mm256_add_epi64(outIn, _mm256_sad_epu8(b1, _mm256_setzero_si256()));

    cnt[i] = _mm256_extract_epi64(b1, 0) + _mm256_extract_epi64(b1, 1) +
      _mm256_extract_epi64(b1, 2) + _mm256_extract_epi64(b1, 3);
  }
}

#define NumAlg 3

void (*algPtr[NumAlg])(const uint32_t *data, uint32_t *cnt) =
  {popcnt64, lookup256, fullAdd256};
char *algName[NumAlg] =
  {"popcnt64", "lookup256", "fullAdd256"};

int32_t main(int32_t argc, char* argv[]) {
  int32_t c, alg = NumAlg - 1, v = 0;
  struct timeval start, stop;
  float sec;

  while ((c = getopt(argc, argv, "a:v")) != -1)
    switch (c) {
    case 'a':
      sscanf(optarg, "%d", &alg);
      break;
    case 'v':
      v = 1;
      break;
    default:
      break;
    }
  if (alg < 0 || alg >= NumAlg)
    alg = NumAlg - 1;
  init();

  gettimeofday(&start, NULL);
  algPtr[alg](data, cnt1);
  gettimeofday(&stop, NULL);
  sec = (stop.tv_sec - start.tv_sec) +
    (stop.tv_usec - start.tv_usec) / (float) MILLION;
  printf("%s: %.3f sec\n", algName[alg], sec);

  if (v) {
    popcnt64(data, cnt2);
    verify();
  }
  return 0;
}
