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
#define VecLen   4096          // vector length in bits
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
      _mm_popcnt_u64(ptr64[64 * i +  0]) + _mm_popcnt_u64(ptr64[64 * i +  1]) +
      _mm_popcnt_u64(ptr64[64 * i +  2]) + _mm_popcnt_u64(ptr64[64 * i +  3]) +
      _mm_popcnt_u64(ptr64[64 * i +  4]) + _mm_popcnt_u64(ptr64[64 * i +  5]) +
      _mm_popcnt_u64(ptr64[64 * i +  6]) + _mm_popcnt_u64(ptr64[64 * i +  7]) +
      _mm_popcnt_u64(ptr64[64 * i +  8]) + _mm_popcnt_u64(ptr64[64 * i +  9]) +
      _mm_popcnt_u64(ptr64[64 * i + 10]) + _mm_popcnt_u64(ptr64[64 * i + 11]) +
      _mm_popcnt_u64(ptr64[64 * i + 12]) + _mm_popcnt_u64(ptr64[64 * i + 13]) +
      _mm_popcnt_u64(ptr64[64 * i + 14]) + _mm_popcnt_u64(ptr64[64 * i + 15]) +
      _mm_popcnt_u64(ptr64[64 * i + 16]) + _mm_popcnt_u64(ptr64[64 * i + 17]) +
      _mm_popcnt_u64(ptr64[64 * i + 18]) + _mm_popcnt_u64(ptr64[64 * i + 19]) +
      _mm_popcnt_u64(ptr64[64 * i + 20]) + _mm_popcnt_u64(ptr64[64 * i + 21]) +
      _mm_popcnt_u64(ptr64[64 * i + 22]) + _mm_popcnt_u64(ptr64[64 * i + 23]) +
      _mm_popcnt_u64(ptr64[64 * i + 24]) + _mm_popcnt_u64(ptr64[64 * i + 25]) +
      _mm_popcnt_u64(ptr64[64 * i + 26]) + _mm_popcnt_u64(ptr64[64 * i + 27]) +
      _mm_popcnt_u64(ptr64[64 * i + 28]) + _mm_popcnt_u64(ptr64[64 * i + 29]) +
      _mm_popcnt_u64(ptr64[64 * i + 30]) + _mm_popcnt_u64(ptr64[64 * i + 31]) +
      _mm_popcnt_u64(ptr64[64 * i + 32]) + _mm_popcnt_u64(ptr64[64 * i + 33]) +
      _mm_popcnt_u64(ptr64[64 * i + 34]) + _mm_popcnt_u64(ptr64[64 * i + 35]) +
      _mm_popcnt_u64(ptr64[64 * i + 36]) + _mm_popcnt_u64(ptr64[64 * i + 37]) +
      _mm_popcnt_u64(ptr64[64 * i + 38]) + _mm_popcnt_u64(ptr64[64 * i + 39]) +
      _mm_popcnt_u64(ptr64[64 * i + 40]) + _mm_popcnt_u64(ptr64[64 * i + 41]) +
      _mm_popcnt_u64(ptr64[64 * i + 42]) + _mm_popcnt_u64(ptr64[64 * i + 43]) +
      _mm_popcnt_u64(ptr64[64 * i + 44]) + _mm_popcnt_u64(ptr64[64 * i + 45]) +
      _mm_popcnt_u64(ptr64[64 * i + 46]) + _mm_popcnt_u64(ptr64[64 * i + 47]) +
      _mm_popcnt_u64(ptr64[64 * i + 48]) + _mm_popcnt_u64(ptr64[64 * i + 49]) +
      _mm_popcnt_u64(ptr64[64 * i + 50]) + _mm_popcnt_u64(ptr64[64 * i + 51]) +
      _mm_popcnt_u64(ptr64[64 * i + 52]) + _mm_popcnt_u64(ptr64[64 * i + 53]) +
      _mm_popcnt_u64(ptr64[64 * i + 54]) + _mm_popcnt_u64(ptr64[64 * i + 55]) +
      _mm_popcnt_u64(ptr64[64 * i + 56]) + _mm_popcnt_u64(ptr64[64 * i + 57]) +
      _mm_popcnt_u64(ptr64[64 * i + 58]) + _mm_popcnt_u64(ptr64[64 * i + 59]) +
      _mm_popcnt_u64(ptr64[64 * i + 60]) + _mm_popcnt_u64(ptr64[64 * i + 61]) +
      _mm_popcnt_u64(ptr64[64 * i + 62]) + _mm_popcnt_u64(ptr64[64 * i + 63]);
}

void lookup128(const uint32_t *data, uint32_t *cnt) {
  const __m128i table = _mm_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m128i mask_0F = _mm_set1_epi8(0x0F);
  __m128i v, low, high, *ptr, acc;
  uint64_t i, j;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m128i *)data;
  for (i = 0; i < NumVec; i++, ptr += 32) {
    acc = _mm_setzero_si128();
    for (j = 0; j < 32; j++) {
      v    = _mm_load_si128((__m128i const*) (ptr + j));
      low  = _mm_and_si128(mask_0F, v);
      high = _mm_and_si128(mask_0F, _mm_srli_epi16(v, 4));
      v    = _mm_add_epi8(_mm_shuffle_epi8(table, low),
			  _mm_shuffle_epi8(table, high));
      acc  = _mm_add_epi64(acc, _mm_sad_epu8(v, _mm_setzero_si128()));
    }
    cnt[i] = _mm_extract_epi64(acc, 0) + _mm_extract_epi64(acc, 1);
  }
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
  for (i = 0; i < NumVec; i++, ptr += 16) {
    acc = _mm256_setzero_si256();
    for (j = 0; j < 16; j++) {
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

void lookup512(const uint32_t *data, uint32_t *cnt) {
  const __m512i table = _mm512_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m512i mask4 = _mm512_set1_epi8(0x0F);
  __m512i v, low, high, *ptr;
  __m256i acc, tmp;
  uint64_t i, j;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m512i *)data;
  for (i = 0; i < NumVec; i++, ptr += 8) {
    acc = _mm256_setzero_si256();
    for (j = 0; j < 8; j++) {
      v    = _mm512_load_si512((__m512i const*) (ptr + j));
      low  = _mm512_and_si512(mask4, v);
      high = _mm512_and_si512(mask4, _mm512_srli_epi16(v, 4));
      v    = _mm512_add_epi8(_mm512_shuffle_epi8(table, low),
			     _mm512_shuffle_epi8(table, high));
      v    = _mm512_sad_epu8(v, _mm512_setzero_si512());
      tmp  = _mm512_extracti64x4_epi64(v, 0);
      acc = _mm256_add_epi64(acc, tmp);
      tmp  = _mm512_extracti64x4_epi64(v, 1);
      acc = _mm256_add_epi64(acc, tmp);
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
  __m256i *ptr, inOut, outIn, first, second, tmp, low, high;
  __m256i c1, c2, c3, c4, c5, c6, c7, b1, b2, b3;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m256i *)data;
  for (i=0; i<NumVec; i++, ptr+=16){
    inOut = _mm256_load_si256((__m256i const*) (ptr + 0));
    b1    = _mm256_load_si256((__m256i const*) (ptr + 1));
    b2    = _mm256_load_si256((__m256i const*) (ptr + 2));
    FULLADD256(c1, outIn, inOut, b1, b2);

    b1 = _mm256_load_si256((__m256i const*) (ptr + 3));
    b2 = _mm256_load_si256((__m256i const*) (ptr + 4));
    FULLADD256(c2, inOut, outIn, b1, b2);

    b1 = _mm256_load_si256((__m256i const*) (ptr + 5));
    b2 = _mm256_load_si256((__m256i const*) (ptr + 6));
    FULLADD256(c3, outIn, inOut, b1, b2);

    b1 = _mm256_load_si256((__m256i const*) (ptr + 7));
    b2 = _mm256_load_si256((__m256i const*) (ptr + 8));
    FULLADD256(c4, inOut, outIn, b1, b2);

    b1 = _mm256_load_si256((__m256i const*) (ptr + 9));
    b2 = _mm256_load_si256((__m256i const*) (ptr + 10));
    FULLADD256(c5, outIn, inOut, b1, b2);

    b1 = _mm256_load_si256((__m256i const*) (ptr + 11));
    b2 = _mm256_load_si256((__m256i const*) (ptr + 12));
    FULLADD256(c6, inOut, outIn, b1, b2);

    b1 = _mm256_load_si256((__m256i const*) (ptr + 13));
    b2 = _mm256_load_si256((__m256i const*) (ptr + 14));
    FULLADD256(c7, first, inOut, b1, b2);
    // first: first bit
    // c1, c2, c3, c4, c5, c6, and c7 are the carries

    FULLADD256(b1,  outIn,    c1, c2, c3);
    FULLADD256(b2,  inOut, outIn, c4, c5);
    FULLADD256(b3, second, inOut, c6, c7);
    // second: second bit
    // b1, b2, and b3 are the carries

    FULLADD256(inOut, outIn, b1, b2, b3);
    //outIn: third bit
    //inOut: fourth bit

    tmp    = _mm256_srli_epi64(first, 1);
    tmp    = _mm256_and_si256(tmp, mask_55);
    first  = _mm256_sub_epi64(first, tmp);
    tmp    = _mm256_srli_epi64(second, 1);
    tmp    = _mm256_and_si256(tmp, mask_55);
    second = _mm256_sub_epi64(second, tmp);
    tmp    = _mm256_srli_epi64(outIn, 1);
    tmp    = _mm256_and_si256(tmp, mask_55);
    outIn  = _mm256_sub_epi64(outIn, tmp);
    tmp    = _mm256_srli_epi64(inOut, 1);
    tmp    = _mm256_and_si256(tmp, mask_55);
    inOut  = _mm256_sub_epi64(inOut, tmp);
    // 0--2 in 2-bit subwords, 12 instructions

    tmp    = _mm256_srli_epi64(first, 2);
    tmp    = _mm256_and_si256(tmp, mask_33);
    first  = _mm256_and_si256(first, mask_33);
    first  = _mm256_add_epi64(first, tmp);
    tmp    = _mm256_srli_epi64(second, 2);
    tmp    = _mm256_and_si256(tmp, mask_33);
    second = _mm256_and_si256(second, mask_33);
    second = _mm256_add_epi64(second, tmp);
    tmp    = _mm256_srli_epi64(outIn, 2);
    tmp    = _mm256_and_si256(tmp, mask_33);
    outIn  = _mm256_and_si256(outIn, mask_33);
    outIn  = _mm256_add_epi64(outIn, tmp);
    tmp    = _mm256_srli_epi64(inOut, 2);
    tmp    = _mm256_and_si256(tmp, mask_33);
    inOut  = _mm256_and_si256(inOut, mask_33);
    inOut  = _mm256_add_epi64(inOut, tmp);
    // 0--4 in 4-bit subwords, 16 instructions
   
    second = _mm256_slli_epi64(second, 1);
    first  = _mm256_add_epi64(first, second);
    inOut  = _mm256_slli_epi64(inOut, 1);
    outIn  = _mm256_add_epi64(outIn, inOut);
    // 0--12 in 4-bit subwords, 4 instructions

    tmp   = _mm256_srli_epi64(first, 4);
    tmp   = _mm256_and_si256(tmp, mask_0F);
    first = _mm256_and_si256(first, mask_0F);
    first = _mm256_add_epi64(first, tmp);
    tmp   = _mm256_srli_epi64(outIn, 4);
    tmp   = _mm256_and_si256(tmp, mask_0F);
    outIn = _mm256_and_si256(outIn, mask_0F);
    outIn = _mm256_add_epi64(outIn, tmp);
    // 0--24 in 8-bit subwords, 8 instructions

    outIn = _mm256_slli_epi64(outIn, 2);
    first = _mm256_add_epi64(first, outIn);
    // 0--120 in 8-bit subwords, 2 instructions

    b1    = _mm256_load_si256((__m256i const*) (ptr + 15));
    low   = _mm256_and_si256(mask_0F, b1);
    high  = _mm256_and_si256(mask_0F, _mm256_srli_epi16(b1, 4));
    b1    = _mm256_add_epi8(_mm256_shuffle_epi8(table, low),
			    _mm256_shuffle_epi8(table, high));

    first = _mm256_add_epi64(first, b1);
    first = _mm256_sad_epu8(first, _mm256_setzero_si256());
    //9 instructions

    cnt[i] = _mm256_extract_epi64(first, 0) + _mm256_extract_epi64(first, 1) +
      _mm256_extract_epi64(first, 2) + _mm256_extract_epi64(first, 3);
  }
}

#define NumAlg 5

void (*algPtr[NumAlg])(const uint32_t *data, uint32_t *cnt) =
  {popcnt64, lookup128, lookup256, lookup512, fullAdd256};
char *algName[NumAlg] =
  {"popcnt64", "lookup128", "lookup256", "lookup512", "fullAdd256"};

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
