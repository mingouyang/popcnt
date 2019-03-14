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
#define VecLen   256           // vector length in bits
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
    cnt[i] = _mm_popcnt_u64(ptr64[4 * i]) + _mm_popcnt_u64(ptr64[4 * i + 1]) +
      _mm_popcnt_u64(ptr64[4 * i + 2]) + _mm_popcnt_u64(ptr64[4 * i + 3]);
}

//Sean Anderson's Bit Twiddling Hacks
//https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
//merge 2 pairs of popcnts of 64-bit words
void merge2p2(const uint32_t *data, uint32_t *cnt) {
  uint64_t *ptr64, u, v, w, x, i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr64 = (uint64_t *)data;
  for (i = 0; i < NumVec; i++) {
    u = ptr64[4 * i + 0];
    v = ptr64[4 * i + 1];
    w = ptr64[4 * i + 2];
    x = ptr64[4 * i + 3];

    u = u - (u >> 1 & 0x5555555555555555ULL);
    v = v - (v >> 1 & 0x5555555555555555ULL);
    w = w - (w >> 1 & 0x5555555555555555ULL);
    x = x - (x >> 1 & 0x5555555555555555ULL);

    u = (u & 0x3333333333333333ULL) + (u >> 2 & 0x3333333333333333ULL);
    v = (v & 0x3333333333333333ULL) + (v >> 2 & 0x3333333333333333ULL);
    w = (w & 0x3333333333333333ULL) + (w >> 2 & 0x3333333333333333ULL);
    x = (x & 0x3333333333333333ULL) + (x >> 2 & 0x3333333333333333ULL);

    u += v;
    w += x;
    u = ((u & 0x0F0F0F0F0F0F0F0FULL) + (u >> 4 & 0x0F0F0F0F0F0F0F0FULL)) *
      0x0101010101010101ULL >> 56;
    w = ((w & 0x0F0F0F0F0F0F0F0FULL) + (w >> 4 & 0x0F0F0F0F0F0F0F0FULL)) *
      0x0101010101010101ULL >> 56;

    cnt[i] = u + w;
  }
}

//merge 3 plus 1 popcnts of 64-bit words
void merge3p1(const uint32_t *data, uint32_t *cnt) {
  uint64_t *ptr64, u, v, w, i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  ptr64 = (uint64_t *)data;
  for (i = 0; i < NumVec; i++) {
    u = ptr64[4 * i + 0];
    v = ptr64[4 * i + 1];
    w = ptr64[4 * i + 2];

    u = u - (u >> 1 & 0x5555555555555555ULL);
    v = v - (v >> 1 & 0x5555555555555555ULL);

    //merge w into u and v
    u += w & 0x5555555555555555ULL;
    v += w >> 1 & 0x5555555555555555ULL;

    //get a new w
    w = ptr64[4 * i + 3];
    w = w - (w >> 1 & 0x5555555555555555ULL);

    u = (u & 0x3333333333333333ULL) + (u >> 2 & 0x3333333333333333ULL);
    v = (v & 0x3333333333333333ULL) + (v >> 2 & 0x3333333333333333ULL);
    w = (w & 0x3333333333333333ULL) + (w >> 2 & 0x3333333333333333ULL);

    u += v;

    u = ((u & 0x0F0F0F0F0F0F0F0FULL) + (u >> 4 & 0x0F0F0F0F0F0F0F0FULL)) *
      0x0101010101010101ULL >> 56;
    w = (w + (w >> 4) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL >> 56;

    cnt[i] = u + w;
  }
}

void lookup256(const uint32_t *data, uint32_t *cnt) {
  const __m256i table = _mm256_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m256i mask4 = _mm256_set1_epi8(0x0F);
  __m256i v, low, high;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i++) {
    v = _mm256_load_si256((__m256i const*) (data + i * 8));
    low = _mm256_and_si256(mask4, v);
    high = _mm256_and_si256(mask4, _mm256_srli_epi16(v, 4));
    v = _mm256_add_epi8(_mm256_shuffle_epi8(table, low),
			_mm256_shuffle_epi8(table, high));

    v = _mm256_sad_epu8(v, _mm256_setzero_si256());
    cnt[i] = _mm256_extract_epi64(v, 0) + _mm256_extract_epi64(v, 1) +
      _mm256_extract_epi64(v, 2) + _mm256_extract_epi64(v, 3);
  }
}

void lookup512(const uint32_t *data, uint32_t *cnt) {
  const __m512i table = _mm512_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
					4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m512i mask4 = _mm512_set1_epi8(0x0F);
  __m512i v, low, high;
  __m256i sum;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i += 2) {
    v    = _mm512_load_si512((__m512i const*) (data + i * 8));
    low  = _mm512_and_si512(mask4, v);
    high = _mm512_and_si512(mask4, _mm512_srli_epi16(v, 4));
    v    = _mm512_add_epi8(_mm512_shuffle_epi8(table, low),
			   _mm512_shuffle_epi8(table, high));
    v    = _mm512_sad_epu8(v, _mm512_setzero_si512());

    sum = _mm512_extracti64x4_epi64(v, 0);
    cnt[i] = _mm256_extract_epi16(sum, 0) + _mm256_extract_epi16(sum, 4) +
      _mm256_extract_epi16(sum, 8) + _mm256_extract_epi16(sum, 12);

    sum = _mm512_extracti64x4_epi64(v, 1);
    cnt[i + 1] = _mm256_extract_epi16(sum, 0) + _mm256_extract_epi16(sum, 4) +
      _mm256_extract_epi16(sum, 8) + _mm256_extract_epi16(sum, 12);
  }
}

//Sean Anderson's Bit Twiddling Hacks
//https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
//with AVX intrinsics
void seander(const uint32_t *data, uint32_t *cnt) {
  const __m256i mask_55 = _mm256_set1_epi8(0x55);
  const __m256i mask_33 = _mm256_set1_epi8(0x33);
  const __m256i mask_0F = _mm256_set1_epi8(0x0F);
  __m256i v, tmp;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i++) {
    v = _mm256_load_si256((__m256i const*) (data + i * 8));

    // right-shift v 1 bit
    tmp = _mm256_srli_epi64(v, 1);
    // v >> 1 & 0x55555555
    tmp = _mm256_and_si256(tmp, mask_55);
    // v = v - (v >> 1 & 0x55555555)
    v = _mm256_sub_epi64(v, tmp);

    // right-shift v by 2 bits
    tmp = _mm256_srli_epi64(v, 2);
    // v >> 2 & 0x33333333
    tmp = _mm256_and_si256(tmp, mask_33);
    // v & 0x33333333
    v = _mm256_and_si256(v, mask_33);
    // v = (v & 0x33333333) + (v >> 2 & 0x33333333)
    v = _mm256_add_epi64(v, tmp);

    // right-shift v by 4 bits
    tmp = _mm256_srli_epi64(v, 4);
    // v + (v >> 4)
    v = _mm256_add_epi64(v, tmp);
    // v = v + (v >> 4) & 0x0F0F0F0F
    v = _mm256_and_si256(v, mask_0F);
    // sum up 8-bit unsigned integers in v
    v = _mm256_sad_epu8(v, _mm256_setzero_si256());
 
    cnt[i] = _mm256_extract_epi64(v, 0) + _mm256_extract_epi64(v, 1) +
      _mm256_extract_epi64(v, 2) + _mm256_extract_epi64(v, 3);
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

void fullAdd(const uint32_t *data, uint32_t *cnt) {
  uint32_t t0, c0, c1, d0, d1, e0, e1, f1, f2;
  uint64_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i++) {
    t0 = data[8 * i];
    FULLADD(c1, c0, t0, data[8 * i + 1], data[8 * i + 2]);
    FULLADD(d1, d0, c0, data[8 * i + 3], data[8 * i + 4]);
    FULLADD(e1, e0, d0, data[8 * i + 5], data[8 * i + 6]);
    FULLADD(f2, f1, c1, d1, e1);
    
    cnt[i] = _mm_popcnt_u32(e0) + _mm_popcnt_u32(f1) * 2 +
      _mm_popcnt_u32(f2) * 4 + _mm_popcnt_u32(data[8 * i + 7]);
  }
}

//carry-save adder
inline void CSA(uint32_t *h, uint32_t *l, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t u = a ^ b;
  *h = a & b | u & c;
  *l = u ^ c;
}

void HarleySeal(const uint32_t *data, uint32_t *cnt) {
  uint32_t onesA, onesB, twosA, twosB, twosC, foursA, foursB;
  uint32_t ones, twos, fours, eights, i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i++) {
    CSA(&twosA,  &onesA,     0, data[8 * i + 0], data[8 * i + 1]);
    CSA(&twosB,  &onesB, onesA, data[8 * i + 2], data[8 * i + 3]);
    CSA(&foursA, &twosC,     0,           twosA,           twosB);
    CSA(&twosA,  &onesA, onesB, data[8 * i + 4], data[8 * i + 5]);
    CSA(&twosB,   &ones, onesA, data[8 * i + 6], data[8 * i + 7]);
    CSA(&foursB,  &twos, twosC,           twosA,           twosB);
    CSA(&eights, &fours,     0,          foursA,          foursB);

    cnt[i] = (_mm_popcnt_u32(eights) << 3) + (_mm_popcnt_u32(fours) << 2) +
      (_mm_popcnt_u32(twos) << 1) + _mm_popcnt_u32(ones);
  }
}

#define NumAlg 8

void (*algPtr[NumAlg])(const uint32_t *data, uint32_t *cnt) =
  {popcnt64, merge2p2, merge3p1, lookup256, lookup512,
   seander, fullAdd, HarleySeal};
char *algName[NumAlg] =
  {"popcnt64", "merge2p2", "merge3p1", "lookup256", "lookup512",
   "seander", "fullAdd", "HarleySeal"};

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
