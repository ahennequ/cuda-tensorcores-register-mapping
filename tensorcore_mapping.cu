#include <stdio.h>
#include <type_traits>
#include <mma.h>
using namespace nvcuda;

// Check tensor core's warp register layout
// nvcc -arch=sm_75 tensorcore_mapping.cu -o mapping
// ./mapping

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

template<typename scalar_t>
__device__ int getWarpRow(int i) {
  int tid = threadIdx.x % 32;
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
    if (std::is_same<scalar_t, half>::value) {
      return (tid & 3) + ((tid & 4) << 1) + ((tid & 16) >> 2);
    } else {
      return (tid & 16) / 4 + 2 * (tid & 4) + (tid & 1) + (i & 2);
    }
  #else
    return (i & 2) * 4 + tid / 4;
  #endif
}

template<typename scalar_t>
__device__ int getWarpCol(int i) {
  int tid = threadIdx.x % 32;
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
    if (std::is_same<scalar_t, half>::value) {
      return (i & 7) + (tid & 8);
    } else {
      return (tid & 10) + (i & 5);
    }
  #else
    return (tid % 4) * 2 + i % 2 + (i & 4) * 2;
  #endif
}

template<int N, int M, typename T>
__global__ void wmma_example(T *elem, T* thread, T* row, T* col) {
  #if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) printf("cuda %d\n", __CUDA_ARCH__);
  #endif

   wmma::fragment<wmma::accumulator, N, M, 16, T> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   for (int i=0 ; i<acc_frag.num_elements; i++) {
    acc_frag.x[i] = i;
   }
   wmma::store_matrix_sync(elem, acc_frag, M, wmma::mem_row_major);

   wmma::fill_fragment(acc_frag, 0.0f);
   for (int i=0 ; i<acc_frag.num_elements; i++) {
    acc_frag.x[i] = threadIdx.x;
   }
   wmma::store_matrix_sync(thread, acc_frag, M, wmma::mem_row_major);

   // row:
   wmma::fill_fragment(acc_frag, 0.0f);
   for (int i=0 ; i<acc_frag.num_elements; i++) {
    acc_frag.x[i] = getWarpRow<T>(i);
   }
   wmma::store_matrix_sync(row, acc_frag, M, wmma::mem_row_major);

   // col:
   wmma::fill_fragment(acc_frag, 0.0f);
   for (int i=0 ; i<acc_frag.num_elements; i++) {
    acc_frag.x[i] = getWarpCol<T>(i);
   }
   wmma::store_matrix_sync(col, acc_frag, M, wmma::mem_row_major);
}

#include <vector>
struct MaskShift {
  int var;
  int mask;
  int shift;

  void prettyPrint() {
    const char* varname;
    if (var == 0) {
      varname = "i";
    } else {
      varname = "tid";
    }
    if (shift < 0) {
      printf("((%s & %d) >> %d)", varname, mask, -shift);
    } else if (shift > 0) {
      printf("((%s & %d) << %d)", varname, mask, shift);
    } else {
      printf("(%s & %d)", varname, mask);
    }
  }
};

template<int N>
void appendVar(int var, std::vector<MaskShift>& formula, int* invariant) {
  int inv_mask = 0;
  for (int i = 0; i < N; i++) {
    inv_mask |= invariant[i];
  }

  for (int bit = 0; bit < 8; bit++) {
    if (((inv_mask >> bit) & 1) == 0) continue;
    for (int bit2 = 0; bit2 < 4; bit2++) {
      bool correlated = true;
      for (int i = 0; i < N; i++) {
        if (((i >> bit2) & 1) != ((invariant[i] >> bit) & 1)) {
          correlated = false;
          break;
        }
      }
      if (correlated) {
        bool added = false;
        int shift = bit2 - bit;
        for (auto& ms : formula) {
          if (ms.var == var && ms.shift == shift) {
            ms.mask |= 1 << bit;
            added = true;
            break;
          }
        }
        if (!added) {
          formula.push_back({var, 1 << bit, shift});
        }
      }
    }
  }
}

template<int N, int M, typename T>
void find_formulae(T* elem, T* thread) {
  std::vector<MaskShift> row_formula;
  std::vector<MaskShift> col_formula;

  int row_invariant_elem[N];
  int row_invariant_thread[N];
  for (int i=0; i<N ; i++) {
    row_invariant_elem[i] = -1;
    row_invariant_thread[i] = -1;
    for (int j=0; j<M; j++) {
      row_invariant_elem[i] &= (int)(float)elem[i*M+j];
      row_invariant_thread[i] &= (int)(float)thread[i*M+j];
    }
  }

  int col_invariant_elem[M];
  int col_invariant_thread[M];
  for (int i=0; i<M ; i++) {
    col_invariant_elem[i] = -1;
    col_invariant_thread[i] = -1;
    for (int j=0; j<N; j++) {
      col_invariant_elem[i] &= (int)(float)elem[j*M+i];
      col_invariant_thread[i] &= (int)(float)thread[j*M+i];
    }
  }

  appendVar<N>(0, row_formula, row_invariant_elem);
  appendVar<N>(1, row_formula, row_invariant_thread);

  printf("Row:\nreturn ");
  const char* pad = "";
  for (auto ms : row_formula) {
    printf("%s", pad);
    ms.prettyPrint();
    pad = " + ";
  }
  printf(";\n\n");

  appendVar<M>(0, col_formula, col_invariant_elem);
  appendVar<M>(1, col_formula, col_invariant_thread);

  printf("Col:\nreturn ");
  pad = "";
  for (auto ms : col_formula) {
    printf("%s", pad);
    ms.prettyPrint();
    pad = " + ";
  }
  printf(";\n\n");
}

template<int N, int M, typename T>
void print_matrix(T* mat) {
  for (int i=0; i<N ; i++) {
    for (int j=0; j<M; j++) {
      printf("%2d ", (int)(float) mat[i*M+j]);
    }
    printf("\n");
  }
  printf("\n");
}

template<int N, int M, typename T>
void print_matrices(T* mat, T* mat2) {
  for (int i=0; i<N ; i++) {
    for (int j=0; j<M; j++) {
      printf("%2d ", (int)(float) mat[i*M+j]);
    }
    printf("\t");
    for (int j=0; j<M; j++) {
      printf("%2d ", (int)(float) mat2[i*M+j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("%s\n", prop.name);

  using F = float;
  static const int N = 32;
  static const int M = 8;

  F *elem;
  F *thread;
  F *row;
  F *col;

  F *elem_host;
  F *thread_host;
  F *row_host;
  F *col_host;

  // Use tensor cores

  cudaErrCheck(cudaMalloc((void**)&elem, N * M * sizeof(F)));
  cudaErrCheck(cudaMalloc((void**)&thread, N * M * sizeof(F)));
  cudaErrCheck(cudaMalloc((void**)&row, N * M * sizeof(F)));
  cudaErrCheck(cudaMalloc((void**)&col, N * M * sizeof(F)));

  elem_host = (F*)malloc(N * M * sizeof(F));
  thread_host = (F*)malloc(N * M * sizeof(F));
  row_host = (F*)malloc(N * M * sizeof(F));
  col_host = (F*)malloc(N * M * sizeof(F));
  
  // First: using WMMA
  dim3 gridDim(1);
  dim3 blockDim(32);
  
  printf("Running with wmma...\n");
  wmma_example<N, M> <<< gridDim, blockDim >>> (elem, thread, row, col);

  // Error checking
  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(elem_host, elem, N * M * sizeof(F), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(thread_host, thread, N * M * sizeof(F), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(row_host, row, N * M * sizeof(F), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(col_host, col, N * M * sizeof(F), cudaMemcpyDeviceToHost));
  
  printf("Elem / ThreadIdx:\n");
  print_matrices<N, M>(elem_host, thread_host);

  find_formulae<N, M>(elem_host, thread_host);

  printf("Verify row / col:\n");
  print_matrices<N, M>(row_host, col_host);

  cudaErrCheck(cudaFree(elem));
  cudaErrCheck(cudaFree(thread));
  cudaErrCheck(cudaFree(row));
  cudaErrCheck(cudaFree(col));
  
  free(elem_host);
  free(thread_host);
  free(row_host);
  free(col_host);

  cudaErrCheck(cudaDeviceReset());
  return 0;
}