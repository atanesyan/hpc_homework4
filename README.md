# Parallel Performance Analysis: SIMD and Multithreading

This report provides a comprehensive overview of three computational tasks optimized using **Data-Level Parallelism** (SIMD) and **Task-Level Parallelism** (Multithreading). 

---

## 1. Project Specifications
* **Compiler:** `gcc`
* **Compilation Flags:** `-pthread -mavx2 -O0`
* **Hardware Target:** x86_64 with AVX2 support.

---

## 2. Implementation Descriptions

### Task 1: DNA Nucleotide Statistics
* **Goal:** Count occurrences of 'A', 'C', 'G', and 'T' in a 256 MB buffer.
* **Scalar:** Uses a standard loop with a `switch` statement.
* **SIMD:** Loads 32 bytes (256 bits) into a `__m256i` register. It performs a parallel comparison using `_mm256_cmpeq_epi8` against four constant vectors (one for each nucleotide). The resulting mask is converted to an integer bitmask via `_mm256_movemask_epi8`, and bits are counted using the hardware-accelerated `__builtin_popcount`.
* **Multithreading:** The buffer is divided into $N$ chunks. Each thread processes its chunk using the SIMD core and updates the global result via a `pthread_mutex`.

### Task 2: Character Buffer Processing
* **Goal:** Convert lowercase English letters ('a'-'z') to uppercase in a 256 MB buffer.
* **SIMD Implementation:** Processes 32 characters per iteration. It creates a boolean mask identifying bytes within the range $[97, 122]$ (ASCII 'a'-'z'). Using bitwise AND, it creates a vector where only lowercase positions contain the value 32. This vector is subtracted from the original, converting lowercase to uppercase without any conditional branching.

### Task 3: Grayscale Image Conversion
* **Goal:** Convert a $512 \times 512$ color PPM image to grayscale.
* **Formula:** $gray = 0.299 \times R + 0.587 \times G + 0.114 \times B$
* **Implementation:** The image uses the "Canonical" $512 \times 512$ size. Because $512$ is a power of two and a multiple of 8, the SIMD registers (processing 8 pixels at a time) align perfectly with the row width, eliminating the need for complex "tail" handling for most of the image.

---

## 3. Technical Explanations

### Buffer Division Among Threads
The buffer is divided into contiguous segments to ensure cache efficiency:
* **Logic:** Each thread is assigned a range $[start, end)$.
* **Calculation:** `start = i * (total_size / num_threads)`. 
* **Edge Case:** To prevent data loss if the size is not perfectly divisible, the last thread takes the remainder: `end = (i == last) ? total_size : (i + 1) * chunk_size`.

### SIMD Processing (AVX2)
We utilize **AVX2** (Advanced Vector Extensions 2), which provides 256-bit registers.
* **Instruction Efficiency:** For 8-bit characters, a single instruction operates on **32 elements** simultaneously.
* **Alignment:** By using a $512 \times 512$ image, we ensure that every row starts on a boundary that is compatible with vector widths, maximizing memory throughput.

---

## 4. Analysis of Results

### The Impact of `-O0`
Compiling with `-O0` prevents the compiler from performing automatic vectorization. This highlights the power of **manual optimization**:
1. **The SIMD Gap:** In Task 1, SIMD is approximately **24 times faster** than Scalar. At `-O0`, the scalar code is extremely inefficient due to branch mispredictions in the `switch` statement and lack of loop unrolling.
2. **Task vs. Data Parallelism:** While Multithreading (4 threads) roughly halved the execution time, a single-threaded SIMD implementation was significantly faster than a 4-threaded scalar implementation. 
3. **Synergy:** The combination of both techniques yielded the best results. SIMD maximizes the efficiency of each core, while Multithreading utilizes the total core count of the CPU.

---

## 5. Output

### Task 1
```
DNA size: 256 MB
Generating DNA sequence...
Generation complete. Took 2.284 sec

Counts (A C G T):
67104560 67108389 67116114 67106393

Scalar time:                    2.140 sec
Multithreading time:            1.233 sec
SIMD time:                      0.089 sec
SIMD + Multithreading time:     0.052 sec
Threads used: 4
```

### Task 2
```
Buffer size: 256 MB
Multithreading time:            0.571 sec
SIMD time:                      0.130 sec
SIMD + Multithreading time:     0.073 sec
Threads used: 4
```

### Task 3
```
Image size: 512 x 512
Threads used: 4

Scalar time: 0.003203 sec
SIMD time: 0.003719 sec
Multithreading time: 0.001648 sec
Multithreading + SIMD time: 0.001960 sec

Verification: PASSED
Output image: gray_output.ppm
```
