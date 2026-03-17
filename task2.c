#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define BUFFER_SIZE (256 * 1024 * 1024)
#define NUM_THREADS 4

typedef struct {
  char* buffer;
  size_t start;
  size_t end;
} ThreadData;

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void process_scalar_core(char* buffer, size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    if (buffer[i] >= 'a' && buffer[i] <= 'z') {
      buffer[i] -= 32;
    }
  }
}

void process_simd_core(char* buffer, size_t start, size_t end) {
  __m256i v_low_bound = _mm256_set1_epi8('a' - 1);
  __m256i v_high_bound = _mm256_set1_epi8('z' + 1);
  __m256i v_diff = _mm256_set1_epi8(32);

  size_t i = start;
  for (; i <= end - 32; i += 32) {
    __m256i chunk = _mm256_loadu_si256((__m256i*)&buffer[i]);
    __m256i mask_low = _mm256_cmpgt_epi8(chunk, v_low_bound);
    __m256i mask_high = _mm256_cmpgt_epi8(v_high_bound, chunk);
    __m256i is_lowercase = _mm256_and_si256(mask_low, mask_high);
    __m256i sub_vec = _mm256_and_si256(is_lowercase, v_diff);
    chunk = _mm256_sub_epi8(chunk, sub_vec);
    _mm256_storeu_si256((__m256i*)&buffer[i], chunk);
  }

  process_scalar_core(buffer, i, end);
}

void* thread_scalar_func(void* arg) {
  ThreadData* data = (ThreadData*)arg;
  process_scalar_core(data->buffer, data->start, data->end);
  return NULL;
}

void* thread_simd_func(void* arg) {
  ThreadData* data = (ThreadData*)arg;
  process_simd_core(data->buffer, data->start, data->end);
  return NULL;
}

int main() {
  printf("Buffer size: %d MB\n", BUFFER_SIZE / (1024 * 1024));

  char* original = malloc(BUFFER_SIZE);
  char* work_buf = malloc(BUFFER_SIZE);

  // Fill buffer with random mix of chars
  const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&* ";
  srand(time(NULL));
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    original[i] = charset[rand() % (sizeof(charset) - 1)];
  }

  pthread_t threads[NUM_THREADS];
  ThreadData tdata[NUM_THREADS];
  double start_time;

  // Multithreading Version
  memcpy(work_buf, original, BUFFER_SIZE);
  start_time = get_time();
  for (int i = 0; i < NUM_THREADS; ++i) {
    tdata[i].buffer = work_buf;
    tdata[i].start = i * (BUFFER_SIZE / NUM_THREADS);
    tdata[i].end = (i == NUM_THREADS - 1) ? BUFFER_SIZE : (i + 1) * (BUFFER_SIZE / NUM_THREADS);
    pthread_create(&threads[i], NULL, thread_scalar_func, &tdata[i]);
  }
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }
  printf("Multithreading time:\t\t%.3f sec\n", get_time() - start_time);

  // SIMD Version (Single Threaded)
  memcpy(work_buf, original, BUFFER_SIZE);
  start_time = get_time();
  process_simd_core(work_buf, 0, BUFFER_SIZE);
  printf("SIMD time:\t\t\t%.3f sec\n", get_time() - start_time);

  // SIMD + Multithreading Version
  memcpy(work_buf, original, BUFFER_SIZE);
  start_time = get_time();
  for (int i = 0; i < NUM_THREADS; ++i) {
    tdata[i].buffer = work_buf;
    tdata[i].start = i * (BUFFER_SIZE / NUM_THREADS);
    tdata[i].end = (i == NUM_THREADS - 1) ? BUFFER_SIZE : (i + 1) * (BUFFER_SIZE / NUM_THREADS);
    pthread_create(&threads[i], NULL, thread_simd_func, &tdata[i]);
  }
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }
  printf("SIMD + Multithreading time:\t%.3f sec\n", get_time() - start_time);
  printf("Threads used: %d\n", NUM_THREADS);

  free(original);
  free(work_buf);
  return 0;
}
