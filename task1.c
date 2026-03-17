#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define DNA_SIZE (256 * 1024 * 1024)
#define NUM_THREADS 4

typedef struct {
  char* buffer;
  size_t start;
  size_t end;
  long* global_counts;
  pthread_mutex_t* mutex;
} ThreadData;

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void count_scalar(char* buffer, size_t size, long* counts) {
  for (size_t i = 0; i < size; ++i) {
    switch (buffer[i]) {
      case 'A': { ++counts[0]; break; }
      case 'C': { ++counts[1]; break; }
      case 'G': { ++counts[2]; break; }
      case 'T': { ++counts[3]; break; }
      default: {
        fprintf(stderr, "Error: Invalid character '%c'\n", buffer[i]);
        exit(1);
      }
    }
  }
}

void count_simd_core(char* buffer, size_t start, size_t end, long* counts) {
  __m256i v_a = _mm256_set1_epi8('A');
  __m256i v_c = _mm256_set1_epi8('C');
  __m256i v_g = _mm256_set1_epi8('G');
  __m256i v_t = _mm256_set1_epi8('T');

  size_t i = start;
  for (; i <= end - 32; i += 32) {
    __m256i chunk = _mm256_loadu_si256((__m256i*) &buffer[i]);
    counts[0] += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_a)));
    counts[1] += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_c)));
    counts[2] += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_g)));
    counts[3] += __builtin_popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_t)));
  }

  // remaining
  for (; i < end; ++i) {
    switch (buffer[i]) {
      case 'A': { ++counts[0]; break; }
      case 'C': { ++counts[1]; break; }
      case 'G': { ++counts[2]; break; }
      case 'T': { ++counts[3]; break; }
    }
  }
}

void* thread_scalar_func(void* arg) {
  ThreadData* data = (ThreadData*) arg;
  long local[4] = {0, 0, 0, 0};
  for (size_t i = data->start; i < data->end; ++i) {
    switch (data->buffer[i]) {
      case 'A': { ++local[0]; break; }
      case 'C': { ++local[1]; break; }
      case 'G': { ++local[2]; break; }
      case 'T': { ++local[3]; break; }
    }
  }

  pthread_mutex_lock(data->mutex);
  for (int i = 0; i < 4; ++i) {
    data->global_counts[i] += local[i];
  }
  pthread_mutex_unlock(data->mutex);

  return NULL;
}

void* thread_simd_func(void* arg) {
  ThreadData* data = (ThreadData*) arg;
  long local[4] = {0, 0, 0, 0};
  count_simd_core(data->buffer, data->start, data->end, local);

  pthread_mutex_lock(data->mutex);
  for (int i = 0; i < 4; ++i) {
    data->global_counts[i] += local[i];
  }
  pthread_mutex_unlock(data->mutex);

  return NULL;
}

long results[4] = {0, 0, 0, 0};

int main() {
  printf("DNA size: %d MB\n", DNA_SIZE / (1024 * 1024));
  char* dna = malloc(DNA_SIZE);
  const char map[] = "ACGT";
  srand(time(NULL));

  double start_time = get_time();
  printf("Generating DNA sequence...\n");
  for (size_t i = 0; i < DNA_SIZE; ++i) {
    dna[i] = map[rand() % 4];
  }
  printf("Generation complete. Took %.3f sec\n\n", get_time() - start_time);

  double t_scalar, t_mt, t_simd, t_simd_mt;
  pthread_t threads[NUM_THREADS];
  ThreadData tdata[NUM_THREADS];
  pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

  // Scalar
  start_time = get_time();
  count_scalar(dna, DNA_SIZE, results);
  t_scalar = get_time() - start_time;

  printf("Counts (A C G T):\n");
  printf("%ld %ld %ld %ld\n\n", results[0], results[1], results[2], results[3]);
  printf("Scalar time:\t\t\t%.3f sec\n", t_scalar);

  // Multithreading Version
  for (int j = 0; j < 4; ++j) {
    results[j] = 0;
  }
  start_time = get_time();
  for (int i = 0; i < NUM_THREADS; ++i) {
    tdata[i].buffer = dna;
    tdata[i].start = i * (DNA_SIZE / NUM_THREADS);
    tdata[i].end = (i == NUM_THREADS - 1) ? DNA_SIZE : (i + 1) * (DNA_SIZE / NUM_THREADS);
    tdata[i].global_counts = results;
    tdata[i].mutex = &lock;
    pthread_create(&threads[i], NULL, thread_scalar_func, &tdata[i]);
  }
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }
  t_mt = get_time() - start_time;
  printf("Multithreading time:\t\t%.3f sec\n", t_mt);

  // SIMD Version
  for (int j = 0; j < 4; ++j) {
    results[j] = 0;
  }
  start_time = get_time();
  count_simd_core(dna, 0, DNA_SIZE, results);
  t_simd = get_time() - start_time;
  printf("SIMD time:\t\t\t%.3f sec\n", t_simd);

  // SIMD + Multithreading Version
  for (int j = 0; j < 4; ++j) {
    results[j] = 0;
  }
  start_time = get_time();
  for (int i = 0; i < NUM_THREADS; ++i) {
    tdata[i].buffer = dna;
    tdata[i].start = i * (DNA_SIZE / NUM_THREADS);
    tdata[i].end = (i == NUM_THREADS - 1) ? DNA_SIZE : (i + 1) * (DNA_SIZE / NUM_THREADS);
    tdata[i].global_counts = results;
    tdata[i].mutex = &lock;
    pthread_create(&threads[i], NULL, thread_simd_func, &tdata[i]);
  }
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }
  t_simd_mt = get_time() - start_time;
  printf("SIMD + Multithreading time:\t%.3f sec\n", t_simd_mt);
  printf("Threads used: %d\n", NUM_THREADS);

  free(dna);
  return 0;
}
