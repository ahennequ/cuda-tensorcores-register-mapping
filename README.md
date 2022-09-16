# Cuda tensorcores register mapping

Since the Volta architecture, NVIDIA's GPUs include tensorcores that can be used to accelerate matrix multiplication.
Each warp is able to produce a 16x16 fragment of the output, stored in a distributed register cache. However, the layout of registers is unspecified. Because of this restriction, user code is limited to storing the fragment back to shared or global memory, using the provided API function, or applying pointwise operations that do not need to know the specific layout.

This tool aim to automatically extract the register mapping of the output fragment and deduce formulae for the row and the column of the element within the 16x16 fragment. The formulae depends on two variables:
* `int tid = threadIdx.x % 32;` the thread index in the warp
* `i` the id of the register in the thread (each thread contains 8 values of the fragment)

The layout also depends on the compute capability of the GPU and for Volta on the type of the fragment (float or half).

## What can be done with this mapping ?

* Apply rowwise or columnwise operations on registers
* Make the matrix upper or lower triangular
* Fill the matrix with procedural values

## How to run on your card

```bash
nvcc -arch=sm_75 tensorcore_mapping.cu -o mapping
./mapping
```

## Mapping on different compute capabilities

### A100 (sm_80), Turing (sm_75)

Mapping:
```
Element / ThreadIdx:
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5          0  0  1  1  2  2  3  3  0  0  1  1  2  2  3  3 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5          4  4  5  5  6  6  7  7  4  4  5  5  6  6  7  7 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5          8  8  9  9 10 10 11 11  8  8  9  9 10 10 11 11 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5         12 12 13 13 14 14 15 15 12 12 13 13 14 14 15 15 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5         16 16 17 17 18 18 19 19 16 16 17 17 18 18 19 19 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5         20 20 21 21 22 22 23 23 20 20 21 21 22 22 23 23 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5         24 24 25 25 26 26 27 27 24 24 25 25 26 26 27 27 
 0  1  0  1  0  1  0  1  4  5  4  5  4  5  4  5         28 28 29 29 30 30 31 31 28 28 29 29 30 30 31 31 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7          0  0  1  1  2  2  3  3  0  0  1  1  2  2  3  3 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7          4  4  5  5  6  6  7  7  4  4  5  5  6  6  7  7 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7          8  8  9  9 10 10 11 11  8  8  9  9 10 10 11 11 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7         12 12 13 13 14 14 15 15 12 12 13 13 14 14 15 15 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7         16 16 17 17 18 18 19 19 16 16 17 17 18 18 19 19 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7         20 20 21 21 22 22 23 23 20 20 21 21 22 22 23 23 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7         24 24 25 25 26 26 27 27 24 24 25 25 26 26 27 27 
 2  3  2  3  2  3  2  3  6  7  6  7  6  7  6  7         28 28 29 29 30 30 31 31 28 28 29 29 30 30 31 31
```

Formulae:
* Row: `return ((i & 2) << 2) + ((tid & 28) >> 2);`
* Col: `return (i & 1) + ((i & 4) << 1) + ((tid & 3) << 1);`

### Volta (sm_70)

Mapping for float:
```
Element / ThreadIdx:
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5          0  0  2  2  0  0  2  2  8  8 10 10  8  8 10 10 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5          1  1  3  3  1  1  3  3  9  9 11 11  9  9 11 11 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7          0  0  2  2  0  0  2  2  8  8 10 10  8  8 10 10 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7          1  1  3  3  1  1  3  3  9  9 11 11  9  9 11 11 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5         16 16 18 18 16 16 18 18 24 24 26 26 24 24 26 26 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5         17 17 19 19 17 17 19 19 25 25 27 27 25 25 27 27 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7         16 16 18 18 16 16 18 18 24 24 26 26 24 24 26 26 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7         17 17 19 19 17 17 19 19 25 25 27 27 25 25 27 27 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5          4  4  6  6  4  4  6  6 12 12 14 14 12 12 14 14 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5          5  5  7  7  5  5  7  7 13 13 15 15 13 13 15 15 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7          4  4  6  6  4  4  6  6 12 12 14 14 12 12 14 14 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7          5  5  7  7  5  5  7  7 13 13 15 15 13 13 15 15 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5         20 20 22 22 20 20 22 22 28 28 30 30 28 28 30 30 
 0  1  0  1  4  5  4  5  0  1  0  1  4  5  4  5         21 21 23 23 21 21 23 23 29 29 31 31 29 29 31 31 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7         20 20 22 22 20 20 22 22 28 28 30 30 28 28 30 30 
 2  3  2  3  6  7  6  7  2  3  2  3  6  7  6  7         21 21 23 23 21 21 23 23 29 29 31 31 29 29 31 31
```

Formulae for float:
* Row: `return (i & 2) + (tid & 1) + ((tid & 4) << 1) + ((tid & 16) >> 2);`
* Col: `return (i & 5) + (tid & 10);`

Mapping for half:
```
Element / ThreadIdx:
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          0  0  0  0  0  0  0  0  8  8  8  8  8  8  8  8 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          1  1  1  1  1  1  1  1  9  9  9  9  9  9  9  9 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          2  2  2  2  2  2  2  2 10 10 10 10 10 10 10 10 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          3  3  3  3  3  3  3  3 11 11 11 11 11 11 11 11 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         16 16 16 16 16 16 16 16 24 24 24 24 24 24 24 24 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         17 17 17 17 17 17 17 17 25 25 25 25 25 25 25 25 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         18 18 18 18 18 18 18 18 26 26 26 26 26 26 26 26 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         19 19 19 19 19 19 19 19 27 27 27 27 27 27 27 27 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          4  4  4  4  4  4  4  4 12 12 12 12 12 12 12 12 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          5  5  5  5  5  5  5  5 13 13 13 13 13 13 13 13 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          6  6  6  6  6  6  6  6 14 14 14 14 14 14 14 14 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7          7  7  7  7  7  7  7  7 15 15 15 15 15 15 15 15 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         20 20 20 20 20 20 20 20 28 28 28 28 28 28 28 28 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         21 21 21 21 21 21 21 21 29 29 29 29 29 29 29 29 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         22 22 22 22 22 22 22 22 30 30 30 30 30 30 30 30 
 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7         23 23 23 23 23 23 23 23 31 31 31 31 31 31 31 31
```

Formulae for half:
* Row: `return (tid & 3) + ((tid & 4) << 1) + ((tid & 16) >> 2);`
* Col: `return (i & 7) + (tid & 8);`

## References

* "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" https://arxiv.org/pdf/1804.06826.pdf
* "Accelerating Reduction and Scan Using Tensor Core Units" https://arxiv.org/pdf/1811.09736.pdf