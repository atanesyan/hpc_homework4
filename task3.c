#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

const float R_COEF = 0.299f;
const float G_COEF = 0.587f;
const float B_COEF = 0.114f;

#pragma pack(push, 1)
typedef struct {
  uint8_t r, g, b;
} Pixel;
#pragma pack(pop)

typedef struct {
  int width;
  int height;
  Pixel* data;
} Image;

typedef struct {
  const Image* input;
  Image* output;
  int start_row;
  int end_row;
  bool use_simd;
} ThreadData;

void skip_comments(FILE* fp) {
  int ch;
  while ((ch = fgetc(fp)) != EOF) {
    if (isspace(ch)) continue;
    if (ch == '#') {
      while ((ch = fgetc(fp)) != EOF && ch != '\n');
    } else {
      ungetc(ch, fp);
      break;
    }
  }
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void process_scalar(const Image* input, Image* output, int start_row, int end_row) {
  for (int y = start_row; y < end_row; ++y) {
    for (int x = 0; x < input->width; ++x) {
      int idx = y * input->width + x;
      Pixel p = input->data[idx];
      uint8_t gray = (uint8_t)(R_COEF * p.r + G_COEF * p.g + B_COEF * p.b);
      output->data[idx] = (Pixel){gray, gray, gray};
    }
  }
}

void process_simd(const Image* input, Image* output, int start_row, int end_row) {
  int width = input->width;
  __m256 r_w = _mm256_set1_ps(R_COEF);
  __m256 g_w = _mm256_set1_ps(G_COEF);
  __m256 b_w = _mm256_set1_ps(B_COEF);

  for (int y = start_row; y < end_row; ++y) {
    int x = 0;
    for (; x <= width - 8; x += 8) {
      int idx = y * width + x;
      float r_f[8], g_f[8], b_f[8];
      for (int i = 0; i < 8; ++i) {
        r_f[i] = (float)input->data[idx + i].r;
        g_f[i] = (float)input->data[idx + i].g;
        b_f[i] = (float)input->data[idx + i].b;
      }
      __m256 r_v = _mm256_loadu_ps(r_f);
      __m256 g_v = _mm256_loadu_ps(g_f);
      __m256 b_v = _mm256_loadu_ps(b_f);
      __m256 gray_v = _mm256_add_ps(_mm256_mul_ps(r_v, r_w),
                      _mm256_add_ps(_mm256_mul_ps(g_v, g_w),
                                    _mm256_mul_ps(b_v, b_w)));
      float res[8];
      _mm256_storeu_ps(res, gray_v);
      for (int i = 0; i < 8; ++i) {
        uint8_t g = (uint8_t)res[i];
        output->data[idx + i] = (Pixel){g, g, g};
      }
    }
    for (; x < width; ++x) {
      int idx = y * width + x;
      uint8_t g = (uint8_t)(R_COEF * input->data[idx].r + G_COEF * input->data[idx].g + B_COEF * input->data[idx].b);
      output->data[idx] = (Pixel){g, g, g};
    }
  }
}

void* thread_worker(void* arg) {
  ThreadData* td = (ThreadData*)arg;
  if (td->use_simd) {
    process_simd(td->input, td->output, td->start_row, td->end_row);
  } else {
    process_scalar(td->input, td->output, td->start_row, td->end_row);
  }
  return NULL;
}

Image load_ppm(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  Image img = {0, 0, NULL};
  if (!fp) return img;

  char magic[3];
  int max_val;
  if (fscanf(fp, "%2s", magic) != 1 || magic[1] != '6') { fclose(fp); return img; }

  skip_comments(fp);
  fscanf(fp, "%d %d", &img.width, &img.height);
  skip_comments(fp);
  fscanf(fp, "%d", &max_val);
  fgetc(fp); 

  size_t num_pixels = (size_t)img.width * img.height;
  img.data = (Pixel*)malloc(num_pixels * sizeof(Pixel));
  if (img.data) fread(img.data, sizeof(Pixel), num_pixels, fp);
  fclose(fp);
  return img;
}

bool verify_grayscale(const Image* img) {
  for (int i = 0; i < img->width * img->height; ++i) {
    if (img->data[i].r != img->data[i].g || img->data[i].g != img->data[i].b) return false;
  }
  return true;
}

int main() {
  const char* input_name = "input.ppm";
  Image input = load_ppm(input_name);
  if (!input.data) return 1;

  Image output = {input.width, input.height, malloc(input.width * input.height * sizeof(Pixel))};
  int num_threads = 4;
  pthread_t threads[4];
  ThreadData td[4];
  double t1, t2, t3, t4, start;

  // Scalar
  start = get_time();
  process_scalar(&input, &output, 0, input.height);
  t1 = get_time() - start;

  // SIMD only
  start = get_time();
  process_simd(&input, &output, 0, input.height);
  t2 = get_time() - start;

  // Multithreading only
  start = get_time();
  int step = input.height / num_threads;
  for (int i = 0; i < num_threads; i++) {
    td[i] = (ThreadData){&input, &output, i * step, (i == 3) ? input.height : (i + 1) * step, false};
    pthread_create(&threads[i], NULL, thread_worker, &td[i]);
  }
  for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
  t3 = get_time() - start;

  // Multithreading + SIMD
  start = get_time();
  for (int i = 0; i < num_threads; i++) {
    td[i].use_simd = true;
    pthread_create(&threads[i], NULL, thread_worker, &td[i]);
  }
  for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
  t4 = get_time() - start;

  printf("Image size: %d x %d\n", input.width, input.height);
  printf("Threads used: %d\n\n", num_threads);
  printf("Scalar time: %.6f sec\n", t1);
  printf("SIMD time: %.6f sec\n", t2);
  printf("Multithreading time: %.6f sec\n", t3);
  printf("Multithreading + SIMD time: %.6f sec\n\n", t4);
  printf("Verification: %s\n", verify_grayscale(&output) ? "PASSED" : "FAILED");
  
  FILE* fp = fopen("gray_output.ppm", "wb");
  fprintf(fp, "P6\n%d %d\n255\n", output.width, output.height);
  fwrite(output.data, sizeof(Pixel), output.width * output.height, fp);
  fclose(fp);
  printf("Output image: gray_output.ppm\n");

  free(input.data); free(output.data);
  return 0;
}
