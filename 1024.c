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
#define VecLen   1024          // vector length in bits
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
      _mm_popcnt_u64(ptr64[16 * i +  0]) +
      _mm_popcnt_u64(ptr64[16 * i +  1]) +
      _mm_popcnt_u64(ptr64[16 * i +  2]) +
      _mm_popcnt_u64(ptr64[16 * i +  3]) +
      _mm_popcnt_u64(ptr64[16 * i +  4]) +
      _mm_popcnt_u64(ptr64[16 * i +  5]) +
      _mm_popcnt_u64(ptr64[16 * i +  6]) +
      _mm_popcnt_u64(ptr64[16 * i +  7]) +
      _mm_popcnt_u64(ptr64[16 * i +  8]) +
      _mm_popcnt_u64(ptr64[16 * i +  9]) +
      _mm_popcnt_u64(ptr64[16 * i + 10]) +
      _mm_popcnt_u64(ptr64[16 * i + 11]) +
      _mm_popcnt_u64(ptr64[16 * i + 12]) +
      _mm_popcnt_u64(ptr64[16 * i + 13]) +
      _mm_popcnt_u64(ptr64[16 * i + 14]) +
      _mm_popcnt_u64(ptr64[16 * i + 15]);
}

void merge2p2(const uint32_t *data, uint32_t *cnt) {
  const __m256i mask_0F = _mm256_set1_epi8(0x0F);
  const __m256i mask_33 = _mm256_set1_epi8(0x33);
  const __m256i mask_55 = _mm256_set1_epi8(0x55);
  __m256i u, v, *ptr, tmp;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m256i *)data;
  for (i = 0; i < NumVec; i++, ptr += 4) {
    u   = _mm256_load_si256((__m256i const*) (ptr + 0));
    tmp = _mm256_srli_epi64(u, 1);
    tmp = _mm256_and_si256(tmp, mask_55);
    u   = _mm256_sub_epi64(u, tmp);
    // 2-bit subwords, 0-2
    tmp = _mm256_srli_epi64(u, 2);
    tmp = _mm256_and_si256(tmp, mask_33);
    u   = _mm256_and_si256(u, mask_33);
    u   = _mm256_add_epi64(u, tmp);
    // 4-bit subwords, 0-4

    v   = _mm256_load_si256((__m256i const*) (ptr + 1));
    tmp = _mm256_srli_epi64(v, 1);
    tmp = _mm256_and_si256(tmp, mask_55);
    v   = _mm256_sub_epi64(v, tmp);
    // 2-bit subwords, 0-2
    tmp = _mm256_srli_epi64(v, 2);
    tmp = _mm256_and_si256(tmp, mask_33);
    v   = _mm256_and_si256(v, mask_33);
    v   = _mm256_add_epi64(v, tmp);
    // 4-bit subwords, 0-4

    v   = _mm256_add_epi64(u, v);
    // 4-bit subwords, 0-8
    tmp = _mm256_srli_epi64(v, 4);
    v   = _mm256_and_si256(v, mask_0F);
    tmp = _mm256_and_si256(tmp, mask_0F);
    v   = _mm256_add_epi64(v, tmp);
    // 8-bit subwords, 0-16
    v   = _mm256_sad_epu8(v, _mm256_setzero_si256());
 
    cnt[i] = _mm256_extract_epi64(v, 0) + _mm256_extract_epi64(v, 1) +
      _mm256_extract_epi64(v, 2) + _mm256_extract_epi64(v, 3);

    u   = _mm256_load_si256((__m256i const*) (ptr + 2));
    tmp = _mm256_srli_epi64(u, 1);
    tmp = _mm256_and_si256(tmp, mask_55);
    u   = _mm256_sub_epi64(u, tmp);
    // 2-bit subwords, 0-2
    tmp = _mm256_srli_epi64(u, 2);
    tmp = _mm256_and_si256(tmp, mask_33);
    u   = _mm256_and_si256(u, mask_33);
    u   = _mm256_add_epi64(u, tmp);
    // 4-bit subwords, 0-4

    v   = _mm256_load_si256((__m256i const*) (ptr + 3));
    tmp = _mm256_srli_epi64(v, 1);
    tmp = _mm256_and_si256(tmp, mask_55);
    v   = _mm256_sub_epi64(v, tmp);
    // 2-bit subwords, 0-2
    tmp = _mm256_srli_epi64(v, 2);
    tmp = _mm256_and_si256(tmp, mask_33);
    v   = _mm256_and_si256(v, mask_33);
    v   = _mm256_add_epi64(v, tmp);
    // 4-bit subwords, 0-4

    v   = _mm256_add_epi64(u, v);
    // 4-bit subwords, 0-8
    tmp = _mm256_srli_epi64(v, 4);
    v   = _mm256_and_si256(v, mask_0F);
    tmp = _mm256_and_si256(tmp, mask_0F);
    v   = _mm256_add_epi64(v, tmp);
    // 8-bit subwords, 0-16
    v   = _mm256_sad_epu8(v, _mm256_setzero_si256());
 
    cnt[i] += _mm256_extract_epi64(v, 0) + _mm256_extract_epi64(v, 1) +
      _mm256_extract_epi64(v, 2) + _mm256_extract_epi64(v, 3);
  }
}

void merge3p1(const uint32_t *data, uint32_t *cnt) {
  const __m256i mask_0F = _mm256_set1_epi8(0x0F);
  const __m256i mask_33 = _mm256_set1_epi8(0x33);
  const __m256i mask_55 = _mm256_set1_epi8(0x55);
  __m256i u, v, w, *ptr, tmp;
  uint64_t *ptr64, u64, v64, w64, i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr   = ( __m256i *)data;
  ptr64 = (uint64_t *)data;
  for (i = 0; i < NumVec; i++, ptr += 4) {
    u   = _mm256_load_si256((__m256i const*) (ptr + 0));
    tmp = _mm256_srli_epi64(u, 1);
    tmp = _mm256_and_si256(tmp, mask_55);
    u   = _mm256_sub_epi64(u, tmp);
    // 0--2 in 2-bit subwords
    v   = _mm256_load_si256((__m256i const*) (ptr + 1));
    tmp = _mm256_srli_epi64(v, 1);
    tmp = _mm256_and_si256(tmp, mask_55);
    v   = _mm256_sub_epi64(v, tmp);
    // 0--2 in 2-bit subwords

    w   = _mm256_load_si256((__m256i const*) (ptr + 2));
    tmp = _mm256_and_si256(w, mask_55);
    u   = _mm256_add_epi64(u, tmp);
    // 0--3 in 2-bit subwords
    w   = _mm256_srli_epi64(w, 1);
    w   = _mm256_and_si256(w, mask_55);
    v   = _mm256_add_epi64(v, w);
    // 0--3 in 2-bit subwords

    tmp = _mm256_srli_epi64(u, 2);
    tmp = _mm256_and_si256(tmp, mask_33);
    u   = _mm256_and_si256(u, mask_33);
    u   = _mm256_add_epi64(u, tmp);
    // 0--6 in 4-bit subwords
    tmp = _mm256_srli_epi64(v, 2);
    tmp = _mm256_and_si256(tmp, mask_33);
    v   = _mm256_and_si256(v, mask_33);
    v   = _mm256_add_epi64(v, tmp);
    // 0--6 in 4-bit subwords

    v   = _mm256_add_epi64(u, v);
    // 0--12 in 4-bit subwords
    tmp = _mm256_srli_epi64(v, 4);
    v   = _mm256_and_si256(v, mask_0F);
    tmp = _mm256_and_si256(tmp, mask_0F);
    v   = _mm256_add_epi64(v, tmp);
    // 0--24 in 8-bit subwords

    v = _mm256_sad_epu8(v, _mm256_setzero_si256());
    cnt[i] = _mm256_extract_epi64(v, 0) + _mm256_extract_epi64(v, 1) +
      _mm256_extract_epi64(v, 2) + _mm256_extract_epi64(v, 3);

    u64 = ptr64[16 * i + 12];
    v64 = ptr64[16 * i + 13];
    w64 = ptr64[16 * i + 14];

    u64 = u64 - (u64 >> 1 & 0x5555555555555555ULL);
    v64 = v64 - (v64 >> 1 & 0x5555555555555555ULL);

    u64 += w64 & 0x5555555555555555ULL;
    v64 += w64 >> 1 & 0x5555555555555555ULL;

    w64 = ptr64[16 * i + 15];
    w64 = w64 - (w64 >> 1 & 0x5555555555555555ULL);

    u64 = (u64 & 0x3333333333333333ULL) + (u64 >> 2 & 0x3333333333333333ULL);
    v64 = (v64 & 0x3333333333333333ULL) + (v64 >> 2 & 0x3333333333333333ULL);
    w64 = (w64 & 0x3333333333333333ULL) + (w64 >> 2 & 0x3333333333333333ULL);

    u64 += v64;

    u64 = ((u64 & 0x0F0F0F0F0F0F0F0FULL) + (u64 >> 4 & 0x0F0F0F0F0F0F0F0FULL))
      * 0x0101010101010101ULL >> 56;
    w64 = (w64 + (w64 >> 4) & 0x0F0F0F0F0F0F0F0FULL)
      * 0x0101010101010101ULL >> 56;
    cnt[i] += u64 + w64;
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
  for (i = 0; i < NumVec; i++, ptr += 4) {
    acc = _mm256_setzero_si256();
    for (j = 0; j < 4; j++) {
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
  for (i = 0; i < NumVec; i++, ptr += 2) {
    acc = _mm256_setzero_si256();
    for (j = 0; j < 2; j++) {
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

//full adder: carryOut, sum, carryIn, bit1, bit2
#define FULLADD(cout,sum,cin,b1,b2) {	\
    sum  = cin  ^ b1;			\
    cin  = cin  & b1;			\
    cout = sum  & b2;			\
    sum  = sum  ^ b2;			\
    cout = cout | cin;			\
  }

void fullAdd64(const uint32_t *data, uint32_t *cnt) {
  uint64_t t0, a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1;
  uint64_t h0, h1, i0, i1, j0, j1, k0, k1, i, *ptr64;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr64 = (uint64_t *)data;
  for (i = 0; i < NumVec; i++) {
    t0 = ptr64[16 * i];
    FULLADD(a1, a0, t0, ptr64[16 * i +  1], ptr64[16 * i +  2]);
    FULLADD(b1, b0, a0, ptr64[16 * i +  3], ptr64[16 * i +  4]);
    FULLADD(c1, c0, b0, ptr64[16 * i +  5], ptr64[16 * i +  6]);
    FULLADD(d1, d0, c0, ptr64[16 * i +  7], ptr64[16 * i +  8]);
    FULLADD(e1, e0, d0, ptr64[16 * i +  9], ptr64[16 * i + 10]);
    FULLADD(f1, f0, e0, ptr64[16 * i + 11], ptr64[16 * i + 12]);
    FULLADD(g1, g0, f0, ptr64[16 * i + 13], ptr64[16 * i + 14]);
    //g0: first bit; a1 -- g1 carries

    FULLADD(h1, h0, c1, b1, a1);
    FULLADD(i1, i0, h0, e1, d1);
    FULLADD(j1, j0, i0, g1, f1);
    //j0: second bit; h1 -- j1 carries
    
    FULLADD(k1, k0, j1, i1, h1);
    //k0: third bit
    //k1: fourth bit

    cnt[i] = _mm_popcnt_u64(g0) + (_mm_popcnt_u64(j0) << 1) +
      (_mm_popcnt_u64(k0) << 2) + (_mm_popcnt_u64(k1) << 3) +
      _mm_popcnt_u64(ptr64[16 * i + 15]);
  }
}

//full adder: carryOut, sum, carryIn, bit1, bit2
#define FULLADD128(cout,sum,cin,b1,b2) {	\
    sum  = _mm_xor_si128(cin, b1);		\
    cin  = _mm_and_si128(cin, b1);		\
    cout = _mm_and_si128(sum, b2);		\
    sum  = _mm_xor_si128(sum, b2);		\
    cout = _mm_or_si128(cout, cin);		\
  }

void fullAdd128(const uint32_t *data, uint32_t *cnt) {
  const __m128i table = _mm_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m128i mask_55 = _mm_set1_epi8(0x55);
  const __m128i mask_33 = _mm_set1_epi8(0x33);
  const __m128i mask_0F = _mm_set1_epi8(0x0F);
  __m128i *ptr, b1, b2, inOut, outIn, c1, c2, c3, tmp, low, high;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr = (__m128i *)data;
  for (i = 0; i < NumVec; i++, ptr += 8) {
    inOut = _mm_load_si128((__m128i const*) (ptr + 0));
    b1    = _mm_load_si128((__m128i const*) (ptr + 1));
    b2    = _mm_load_si128((__m128i const*) (ptr + 2));
    FULLADD128(c1, outIn, inOut, b1, b2);

    b1 = _mm_load_si128((__m128i const*) (ptr + 3));
    b2 = _mm_load_si128((__m128i const*) (ptr + 4));
    FULLADD128(c2, inOut, outIn, b1, b2);

    b1 = _mm_load_si128((__m128i const*) (ptr + 5));
    b2 = _mm_load_si128((__m128i const*) (ptr + 6));
    FULLADD128(c3, outIn, inOut, b1, b2);
    // outIn is the sum
    // c1, c2, and c3 are the carries

    FULLADD128(b2, b1, c1, c2, c3);

    tmp   = _mm_srli_epi64(outIn, 1);
    tmp   = _mm_and_si128(tmp, mask_55);
    outIn = _mm_sub_epi64(outIn, tmp);
    tmp   = _mm_srli_epi64(b1, 1);
    tmp   = _mm_and_si128(tmp, mask_55);
    b1    = _mm_sub_epi64(b1, tmp);
    tmp   = _mm_srli_epi64(b2, 1);
    tmp   = _mm_and_si128(tmp, mask_55);
    b2    = _mm_sub_epi64(b2, tmp);
    // 0--2 in 2-bit subwords

    tmp   = _mm_srli_epi64(outIn, 2);
    tmp   = _mm_and_si128(tmp, mask_33);
    outIn = _mm_and_si128(outIn, mask_33);
    outIn = _mm_add_epi64(outIn, tmp);
    tmp   = _mm_srli_epi64(b1, 2);
    tmp   = _mm_and_si128(tmp, mask_33);
    b1    = _mm_and_si128(b1, mask_33);
    b1    = _mm_add_epi64(b1, tmp);
    tmp   = _mm_srli_epi64(b2, 2);
    tmp   = _mm_and_si128(tmp, mask_33);
    b2    = _mm_and_si128(b2, mask_33);
    b2    = _mm_add_epi64(b2, tmp);
    // 0--4 in 4-bit subwords
   
    b1    = _mm_slli_epi64(b1, 1);
    outIn = _mm_add_epi64(outIn, b1);
    // 0--12 in 4-bit subwords

    tmp   = _mm_srli_epi64(outIn, 4);
    tmp   = _mm_and_si128(tmp, mask_0F);
    outIn = _mm_and_si128(outIn, mask_0F);
    outIn = _mm_add_epi64(outIn, tmp);
    // 0--24 in 8-bit subwords

    tmp   = _mm_srli_epi64(b2, 4);
    b2    = _mm_add_epi64(b2, tmp);
    b2    = _mm_and_si128(b2, mask_0F);
    // 0--8 in 8-bit subwords
    b2    = _mm_slli_epi64(b2, 2);
    outIn = _mm_add_epi64(outIn, b2);
    // 0--56 in 8-bit subwords

    outIn = _mm_sad_epu8(outIn, _mm_setzero_si128());
    cnt[i] = _mm_extract_epi16(outIn, 0) + _mm_extract_epi16(outIn, 4);

    b1   = _mm_load_si128((__m128i const*) (ptr + 7));
    low  = _mm_and_si128(mask_0F, b1);
    high = _mm_and_si128(mask_0F, _mm_srli_epi16(b1, 4));
    b1   = _mm_add_epi8(_mm_shuffle_epi8(table, low),
			_mm_shuffle_epi8(table, high));

    b1 = _mm_sad_epu8(b1, _mm_setzero_si128());
    cnt[i] += _mm_extract_epi64(b1, 0) + _mm_extract_epi64(b1, 1);
  }
}

#define NumAlg 7

void (*algPtr[NumAlg])(const uint32_t *data, uint32_t *cnt) =
  {popcnt64, merge2p2, merge3p1, lookup256, lookup512,
   fullAdd64, fullAdd128};
char *algName[NumAlg] =
  {"popcnt64", "merge2p2", "merge3p1", "lookup256", "lookup512",
   "fullAdd64", "fullAdd128"};

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
