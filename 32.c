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
#include <sys/time.h>
#include <x86intrin.h>

#define MILLION  1000000L
#define NumVec   268435456     // 2^28
#define VecLen   32            // vector length in bits
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
void popcnt32(const uint32_t *data, uint32_t *cnt) {
  uint32_t i;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i++)
    cnt[i] = _mm_popcnt_u32(data[i]);
}

//Sean Anderson's Bit Twiddling Hacks
//https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
void seander(const uint32_t *data, uint32_t *cnt) {
  uint32_t i, v;

  __assume_aligned(data, ALIGN);
  __assume_aligned(cnt, ALIGN);

  for (i = 0; i < NumVec; i++) {
    v = data[i];
    v = v - (v >> 1 & 0x55555555);
    v = (v & 0x33333333) + (v >> 2 & 0x33333333);
    v = (v + (v >> 4) & 0x0F0F0F0F) * 0x01010101 >> 24;
    cnt[i] = v;
  }
}

#define NumAlg 2

void (*algPtr[NumAlg])(const uint32_t *data, uint32_t *cnt) =
  {popcnt32, seander};
char *algName[NumAlg] =
  {"popcnt32", "seander"};

int32_t main(int32_t argc, char *argv[]) {
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
    popcnt32(data, cnt2);
    verify();
  }
  return 0;
}
