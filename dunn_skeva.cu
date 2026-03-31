#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <ctype.h>
#include <locale.h>

#define CUDA_CHECK(x) do { \
  cudaError_t err_ = (x); \
  if (err_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define ENABLE_TIMING 1
#if ENABLE_TIMING
  #include <time.h>
  static inline double wall_time_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
  }
#endif

// Número de streams para overlap (double-buffering)
#define NUM_STREAMS 2

// Modos de separação intercluster
#define INTER_MODE_CENTROIDS 0
#define INTER_MODE_POINTS    1

typedef struct {
  const char* file;
  const char* data_file;   // modo kmeans: arquivo de dados flat
  const char* labels_file; // modo kmeans: arquivo de labels
  int NF;
  int K;
  int sketch_size;    // S
  int validate_size;  // V
  int reps;           // R
  int threads;        // threads por bloco
  unsigned long long seed;
  int use_streams;    // flag para habilitar/desabilitar streams
  int inter_mode;     // INTER_MODE_CENTROIDS ou INTER_MODE_POINTS
  float sample_pct;   // % do cluster usado como pool SkeVa (0 = usar tudo)
} Args;

// --------- PRNG simples (xorshift64*) ----------
typedef struct { unsigned long long s; } Rng;
static inline void rng_seed(Rng* r, unsigned long long seed){ r->s = seed?seed:0x9e3779b97f4a7c15ULL; }
static inline unsigned long long rng_u64(Rng* r){
  unsigned long long x = r->s;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27; r->s = x;
  return x * 2685821657736338717ULL;
}
static inline unsigned int rng_u32(Rng* r){ return (unsigned int)(rng_u64(r) >> 32); }
static inline int rng_int(Rng* r, int lo, int hi){
  unsigned int range = (unsigned int)(hi - lo + 1);
  return lo + (int)(rng_u32(r) % (range?range:1));
}

// --------- parse args ----------
static void parse_args(int argc, char** argv, Args* a){
  a->file = NULL; a->data_file = NULL; a->labels_file = NULL;
  a->NF=-1; a->K=-1;
  a->sketch_size=512; a->validate_size=512; a->reps=8; a->threads=256;
  a->seed = 42ULL;
  a->use_streams = 1;
  a->inter_mode = INTER_MODE_CENTROIDS;  // padrão: centróides
  a->sample_pct = 0.f;                   // padrão: usar tudo
  for (int i=1;i<argc;i++){
    if (!strcmp(argv[i],"--file") && i+1<argc) a->file = argv[++i];
    else if (!strcmp(argv[i],"--data_file") && i+1<argc) a->data_file = argv[++i];
    else if (!strcmp(argv[i],"--labels_file") && i+1<argc) a->labels_file = argv[++i];
    else if (!strcmp(argv[i],"--nf") && i+1<argc) a->NF = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--k") && i+1<argc)  a->K  = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--sketch_size") && i+1<argc) a->sketch_size = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--validate_size") && i+1<argc) a->validate_size = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--reps") && i+1<argc) a->reps = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--threads") && i+1<argc) a->threads = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--seed") && i+1<argc) a->seed = (unsigned long long)atoll(argv[++i]);
    else if (!strcmp(argv[i],"--no-streams")) a->use_streams = 0;
    else if (!strcmp(argv[i],"--sample_pct") && i+1<argc){
      a->sample_pct = (float)atof(argv[++i]);
      if (a->sample_pct < 0.f || a->sample_pct > 100.f){
        fprintf(stderr,"--sample_pct deve estar entre 0 e 100 (recebido: %.2f)\n", a->sample_pct);
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i],"--inter_mode") && i+1<argc){
      ++i;
      if (!strcmp(argv[i],"centroids"))     a->inter_mode = INTER_MODE_CENTROIDS;
      else if (!strcmp(argv[i],"points"))   a->inter_mode = INTER_MODE_POINTS;
      else { fprintf(stderr,"--inter_mode inválido: %s (use 'centroids' ou 'points')\n", argv[i]); exit(EXIT_FAILURE); }
    }
    else { fprintf(stderr,"Arg inválido: %s\n", argv[i]); exit(EXIT_FAILURE); }
  }
  int mode_file   = (a->file != NULL);
  int mode_kmeans = (a->data_file != NULL && a->labels_file != NULL);

  if (!mode_file && !mode_kmeans) {
    fprintf(stderr,
      "Uso (modo padrão):  %s --file <path> --nf <NF> --k <K> [...]\n"
      "Uso (modo k-means): %s --data_file <path> --labels_file <path> --nf <NF> [...]\n"
      "\n"
      "Opções comuns:\n"
      "  --sketch_size S         tamanho fixo do sketch (padrão: 512)\n"
      "  --validate_size V       tamanho do validate (padrão: 512)\n"
      "  --reps R                repetições SkeVa (padrão: 8)\n"
      "  --threads T             threads por bloco CUDA (padrão: 256)\n"
      "  --seed N                semente RNG (padrão: 42)\n"
      "  --no-streams            desabilitar CUDA streams\n"
      "  --inter_mode centroids  separação = dist. mínima entre centróides (padrão)\n"
      "  --inter_mode points     separação = dist. mínima ponto-a-ponto (exato, O(N²))\n"
      "  --sample_pct P          sketch_size = P%% do cluster (0 = usar --sketch_size fixo)\n",
      argv[0], argv[0]);
    exit(EXIT_FAILURE);
  }
  if (mode_kmeans && a->NF <= 0) {
    fprintf(stderr, "Erro: --nf <NF> é obrigatório no modo --data_file/--labels_file\n");
    exit(EXIT_FAILURE);
  }
  if (mode_file && (a->NF <= 0 || a->K <= 0)) {
    fprintf(stderr, "Erro: --nf <NF> e --k <K> são obrigatórios no modo --file\n");
    exit(EXIT_FAILURE);
  }
}

// --------- leitura do dataset ----------
static void read_dataset(const Args* a, float** X_out, int** y_out, size_t* N_out, int* NF_out, int* K_out){
  FILE* fp = fopen(a->file, "r");
  if (!fp){ perror("fopen"); exit(EXIT_FAILURE); }

  int Kf=0, NFf=0;
  if (fscanf(fp, " %d %d", &Kf, &NFf) != 2) {
    fprintf(stderr, "Formato inesperado: não consegui ler 'K NF' na 1a linha.\n");
    exit(EXIT_FAILURE);
  }

  int K = Kf, NF = NFf;

  int* sizes = (int*)malloc(sizeof(int)*K);
  if (!sizes){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  for (int i=0;i<K;i++){
    if (fscanf(fp, " %d", &sizes[i]) != 1){
      fprintf(stderr,"Formato inesperado: não consegui ler sizes[%d] na 2a linha.\n", i);
      exit(EXIT_FAILURE);
    }
  }
  int ch; while ((ch=fgetc(fp))!='\n' && ch!=EOF){}

  size_t N = 0; for (int i=0;i<K;i++) N += (size_t)sizes[i];

  float* X = (float*)malloc(sizeof(float)*N*NF);
  int*   y = (int*)  malloc(sizeof(int)*N);
  if (!X || !y){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }

  size_t r = 0;
  for (int c=0;c<K;c++){
    for (int t=0; t<sizes[c]; t++){
      for (int f=0; f<NF; f++){
        if (fscanf(fp, " %f", &X[r*(size_t)NF + f]) != 1){
          fprintf(stderr,"Linha incompleta ao ler ponto %zu (cluster %d, pos %d), feature %d.\n",
                  r, c, t, f);
          exit(EXIT_FAILURE);
        }
      }
      y[r] = c;
      r++;
    }
  }

  fclose(fp);
  free(sizes);

  *X_out = X; *y_out = y; *N_out = N;
  *NF_out = NF; *K_out = K;

  fprintf(stderr, "Lidos %zu registros (NF=%d, K=%d) de %s\n", N, NF, K, a->file);
}


// --------- leitura no formato k-means (dataset flat + labels separados) ----------
// Formato data_file:   N linhas, cada linha com NF floats separados por espaço
// Formato labels_file: N linhas, cada linha com um inteiro (label 0-based)
// K é inferido como max(label)+1 (ou a->K se fornecido via --k)
static void read_dataset_kmeans(const Args* a, float** X_out, int** y_out,
                                size_t* N_out, int* NF_out, int* K_out)
{
  int NF = a->NF;

  // 1ª passagem no labels: contar N e encontrar K
  FILE* fl = fopen(a->labels_file, "r");
  if (!fl) { perror("fopen labels_file"); exit(EXIT_FAILURE); }

  size_t N = 0;
  int max_label = -1;
  int lv;
  while (fscanf(fl, " %d", &lv) == 1) {
    if (lv > max_label) max_label = lv;
    N++;
  }
  fclose(fl);

  if (N == 0) { fprintf(stderr, "Erro: labels_file vazio ou inválido\n"); exit(EXIT_FAILURE); }
  int K = (a->K > 0) ? a->K : (max_label + 1);

  float* X = (float*)malloc(sizeof(float) * N * NF);
  int*   y = (int*)  malloc(sizeof(int)   * N);
  if (!X || !y) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }

  // Ler data_file
  FILE* fd = fopen(a->data_file, "r");
  if (!fd) { perror("fopen data_file"); exit(EXIT_FAILURE); }
  for (size_t i = 0; i < N; i++) {
    for (int f = 0; f < NF; f++) {
      if (fscanf(fd, " %f", &X[i * (size_t)NF + f]) != 1) {
        fprintf(stderr, "Erro lendo ponto %zu feature %d em data_file\n", i, f);
        exit(EXIT_FAILURE);
      }
    }
  }
  fclose(fd);

  // 2ª passagem no labels: preencher array y
  fl = fopen(a->labels_file, "r");
  if (!fl) { perror("fopen labels_file"); exit(EXIT_FAILURE); }
  for (size_t i = 0; i < N; i++) {
    if (fscanf(fl, " %d", &y[i]) != 1) {
      fprintf(stderr, "Erro lendo label %zu em labels_file\n", i);
      exit(EXIT_FAILURE);
    }
    if (y[i] < 0 || y[i] >= K) {
      fprintf(stderr, "Erro: label[%zu]=%d fora do intervalo [0,%d)\n", i, y[i], K);
      exit(EXIT_FAILURE);
    }
  }
  fclose(fl);

  *X_out = X; *y_out = y; *N_out = N;
  *NF_out = NF; *K_out = K;

  fprintf(stderr, "Lidos %zu registros (NF=%d, K=%d) de '%s' + '%s'\n",
          N, NF, K, a->data_file, a->labels_file);
}

// --------- construir índice por cluster ----------
static void build_clusters(const int* y, size_t N, int K, int*** idx_out, int** sizes_out){
  int* sizes = (int*)calloc(K,sizeof(int));
  if (!sizes){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  for (size_t i=0;i<N;i++) sizes[y[i]]++;

  int** idx = (int**)malloc(sizeof(int*)*K);
  if (!idx){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  for (int c=0;c<K;c++){
    idx[c] = (int*)malloc(sizeof(int)*sizes[c]);
    if (!idx[c]){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  }
  int* pos = (int*)calloc(K,sizeof(int));
  if (!pos){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  for (size_t i=0;i<N;i++){
    int c = y[i];
    idx[c][pos[c]++] = (int)i;
  }
  free(pos);
  *idx_out=idx; *sizes_out=sizes;
}

// --------- centróides (CPU) ----------
static void compute_centroids_cpu(const float* X, int NF, int K, int** idx, int* sizes, float** C_out){
  float* C = (float*)calloc((size_t)K*NF,sizeof(float));
  if (!C){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  for (int c=0;c<K;c++){
    int sz = sizes[c];
    if (sz<=0) continue;
    for (int t=0;t<sz;t++){
      int id = idx[c][t];
      const float* xi = &X[(size_t)id*NF];
      for (int f=0;f<NF;f++) C[c*NF+f]+=xi[f];
    }
    float inv = 1.0f/(float)sz;
    for (int f=0;f<NF;f++) C[c*NF+f]*=inv;
  }
  *C_out=C;
}

// --------- separação mínima entre centróides (CPU, O(K²)) ----------
static float min_centroid_separation(const float* C, int K, int NF){
  if (K<2) return 0.f;
  float best = FLT_MAX;
  for (int i=0;i<K;i++){
    for (int j=i+1;j<K;j++){
      float s=0.f;
      for (int f=0;f<NF;f++){
        float d = C[i*NF+f]-C[j*NF+f];
        s += d*d;
      }
      if (s<best) best=s;
    }
  }
  return best;
}

// --------- comparador para qsort ----------
static int cmp_int(const void* a, const void* b){
  int ia = *(const int*)a, ib = *(const int*)b;
  return (ia<ib)?-1:(ia>ib)?1:0;
}

// --------- amostragem sem reposição ----------
static void sample_indices(const int* pool, int pool_sz, int need, Rng* rng, int** out, int* out_sz){
  if (pool_sz<=0 || need<=0){ *out=NULL; *out_sz=0; return; }
  int take = (need<pool_sz)?need:pool_sz;
  int* tmp = (int*)malloc(sizeof(int)*pool_sz);
  if (!tmp){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  memcpy(tmp,pool,sizeof(int)*pool_sz);
  for (int i=0;i<take;i++){
    int j = i + rng_int(rng,0,pool_sz-1-i);
    int t = tmp[i]; tmp[i]=tmp[j]; tmp[j]=t;
  }
  int* res = (int*)malloc(sizeof(int)*take);
  if (!res){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  memcpy(res,tmp,sizeof(int)*take);
  free(tmp);
  *out = res; *out_sz = take;
}

// --------- união única (concat + sort + unique) ----------
static void merge_unique(const int* a, int na, const int* b, int nb, int** out, int* out_sz){
  int total = na+nb;
  int* v = (int*)malloc(sizeof(int)*total);
  if (!v){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  memcpy(v,a,sizeof(int)*na);
  memcpy(v+na,b,sizeof(int)*nb);
  qsort(v,total,sizeof(int),cmp_int);
  int w=0;
  for (int i=0;i<total;i++){
    if (w==0 || v[i]!=v[w-1]) v[w++]=v[i];
  }
  int* res = (int*)malloc(sizeof(int)*w);
  if (!res){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  memcpy(res,v,sizeof(int)*w);
  free(v);
  *out=res; *out_sz=w;
}

// --------- gather: copia pontos amostrados para buffer contíguo ----------
static void gather_points(const float* X, const int* indices, int S, int NF, float* out){
  for (int i = 0; i < S; i++){
    int idx = indices[i];
    const float* src = &X[(size_t)idx * NF];
    float* dst = &out[(size_t)i * NF];
    memcpy(dst, src, NF * sizeof(float));
  }
}

#ifndef PARES_POR_THREAD
#define PARES_POR_THREAD 32
#endif

// =============================================================
// KERNEL: máximo pairwise intra-cluster (triângulo superior)
// =============================================================
// Cada thread cuida de um par (i,j), i<j, mapeado do índice linear gid.
// Redução por shared memory retorna o máximo por bloco em blockMax[].
__global__
void pairwise_max_kernel_simple(const float* __restrict__ d_sample,
                                int S, int NF, long long P,
                                float* __restrict__ blockMax)
{
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  long long gid = (long long)blockIdx.x * blockDim.x + tid;

  float val = 0.f;
  if (gid < P){
    // Mapeia gid → par (i,j) no triângulo superior
    double tmp = 2.0 * (double)(P - 1 - gid);
    int i_approx = (int)(S - 1 - sqrt(tmp));
    if (i_approx < 0) i_approx = 0;
    if (i_approx >= S-1) i_approx = S - 2;

    int i = i_approx;
    long long row_start = (long long)i * (2*S - i - 1) / 2;
    while (row_start > gid && i > 0){
      i--;
      row_start = (long long)i * (2*S - i - 1) / 2;
    }
    while (i < S - 1){
      long long next_row_start = (long long)(i+1) * (2*S - i - 2) / 2;
      if (next_row_start > gid) break;
      i++;
      row_start = next_row_start;
    }

    int j = (int)(i + 1 + (gid - row_start));

    const float* xi = d_sample + (size_t)i * NF;
    const float* xj = d_sample + (size_t)j * NF;
    float s = 0.f;
    for (int f = 0; f < NF; f++){
      float d = xi[f] - xj[f];
      s += d*d;
    }
    val = s;
  }

  sdata[tid] = val;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
    if (tid < stride){
      float a = sdata[tid], b = sdata[tid + stride];
      sdata[tid] = (a > b) ? a : b;
    }
    __syncthreads();
  }

  if (tid == 0) blockMax[blockIdx.x] = sdata[0];
}

__global__
void pairwise_max_kernel_optimized(const float* __restrict__ d_sample,
                                   int S, int NF, long long P,
                                   float* __restrict__ blockMax)
{
  long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x)
                   * (long long)PARES_POR_THREAD;

  // ── Nível 1: registrador ─────────────────────────────────────────────
  float local_max = 0.f;
  for (int k = 0; k < PARES_POR_THREAD; k++) {
    long long pid = base + k;
    if (pid >= P) break;

    // Decodifica pid → par (i,j) — mesma lógica do kernel simples
    double tmp = 2.0 * (double)(P - 1 - pid);
    int i_approx = (int)(S - 1 - sqrt(tmp));
    if (i_approx < 0)    i_approx = 0;
    if (i_approx >= S-1) i_approx = S - 2;

    int i = i_approx;
    long long row_start = (long long)i * (2*S - i - 1) / 2;
    while (row_start > pid && i > 0) {
      i--;
      row_start = (long long)i * (2*S - i - 1) / 2;
    }
    while (i < S - 1) {
      long long next_row_start = (long long)(i+1) * (2*S - i - 2) / 2;
      if (next_row_start > pid) break;
      i++;
      row_start = next_row_start;
    }
    int j = (int)(i + 1 + (pid - row_start));

    const float* xi = d_sample + (size_t)i * NF;
    const float* xj = d_sample + (size_t)j * NF;
    float dist = 0.f;
    for (int f = 0; f < NF; f++) {
      float d = xi[f] - xj[f];
      dist += d * d;
    }
    if (dist > local_max) local_max = dist;
  }

  // ── Nível 2a: warp shuffle ───────────────────────────────────────────
  unsigned mask = 0xFFFFFFFF;
  float warp_max = local_max;
  warp_max = fmaxf(warp_max, __shfl_down_sync(mask, warp_max, 16));
  warp_max = fmaxf(warp_max, __shfl_down_sync(mask, warp_max,  8));
  warp_max = fmaxf(warp_max, __shfl_down_sync(mask, warp_max,  4));
  warp_max = fmaxf(warp_max, __shfl_down_sync(mask, warp_max,  2));
  warp_max = fmaxf(warp_max, __shfl_down_sync(mask, warp_max,  1));

  // ── Nível 2b: shmem — apenas lane 0 de cada warp escreve ────────────
  extern __shared__ float sdata[];
  int lane     = threadIdx.x & 31;
  int warpId   = threadIdx.x >> 5;
  int numWarps = blockDim.x >> 5;

  if (lane == 0) sdata[warpId] = warp_max;
  __syncthreads();

  // ── Nível 2c: warp 0 reduz os numWarps valores restantes ────────────
  float block_max = 0.f;
  if (warpId == 0) {
    block_max = (lane < numWarps) ? sdata[lane] : 0.f;
    block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max, 16));
    block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max,  8));
    block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max,  4));
    block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max,  2));
    block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max,  1));
  }

  // ── Nível 3: memória global ──────────────────────────────────────────
  if (threadIdx.x == 0)
    blockMax[blockIdx.x] = block_max;
}

// =============================================================
// KERNEL: mínimo pairwise inter-cluster (grade SA x SB)
// =============================================================
// d_A: SA pontos do cluster A  (SA * NF floats)
// d_B: SB pontos do cluster B  (SB * NF floats)
// P = SA * SB  (grade completa, não triangular)
// gid → i = gid / SB  (ponto em A)
//       j = gid % SB  (ponto em B)
// Redução por shared memory retorna o mínimo por bloco em blockMin[].
__global__
void intercluster_min_kernel(const float* __restrict__ d_A, int SA,
                             const float* __restrict__ d_B, int SB,
                             int NF, long long P,
                             float* __restrict__ blockMin)
{
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  long long gid = (long long)blockIdx.x * blockDim.x + tid;

  float val = FLT_MAX;
  if (gid < P){
    int i = (int)(gid / SB);
    int j = (int)(gid % SB);

    const float* xi = d_A + (size_t)i * NF;
    const float* xj = d_B + (size_t)j * NF;
    float s = 0.f;
    for (int f = 0; f < NF; f++){
      float d = xi[f] - xj[f];
      s += d*d;
    }
    val = s;
  }

  sdata[tid] = val;
  __syncthreads();

  // Redução para mínimo
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
    if (tid < stride){
      float a = sdata[tid], b = sdata[tid + stride];
      sdata[tid] = (a < b) ? a : b;
    }
    __syncthreads();
  }

  if (tid == 0) blockMin[blockIdx.x] = sdata[0];
}

// =============================================================
// GPU pairwise MAX (sem streams) — para diâmetro intra-cluster
// =============================================================
static float gpu_pairwise_max_simple(const float* h_X,
                                     const int* h_indices, int S,
                                     int NF, int threads)
{
  if (S <= 1) return 0.f;
  long long P = (long long)S * (S - 1) / 2;

  size_t sample_bytes = (size_t)S * NF * sizeof(float);
  float* h_sample = (float*)malloc(sample_bytes);
  if (!h_sample){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  gather_points(h_X, h_indices, S, NF, h_sample);

  float* d_sample = NULL;
  CUDA_CHECK(cudaMalloc(&d_sample, sample_bytes));
  CUDA_CHECK(cudaMemcpy(d_sample, h_sample, sample_bytes, cudaMemcpyHostToDevice));

  int tpb = (threads > 0) ? threads : 256;
  long long threads_needed = (P + PARES_POR_THREAD - 1) / PARES_POR_THREAD;
  int blocks = (int)((threads_needed + tpb - 1) / tpb);
  if (blocks < 1) blocks = 1;

  float* d_blockMax = NULL;
  CUDA_CHECK(cudaMalloc(&d_blockMax, sizeof(float) * blocks));
  size_t shmem = (size_t)(tpb / 32) * sizeof(float);

  pairwise_max_kernel_optimized<<<blocks, tpb, shmem>>>(d_sample, S, NF, (long long)P, d_blockMax);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float* h_blockMax = (float*)malloc(sizeof(float) * blocks);
  if (!h_blockMax){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
  CUDA_CHECK(cudaMemcpy(h_blockMax, d_blockMax, sizeof(float) * blocks, cudaMemcpyDeviceToHost));

  float mx = 0.f;
  for (int i = 0; i < blocks; i++){
    if (h_blockMax[i] > mx) mx = h_blockMax[i];
  }

  free(h_blockMax);
  free(h_sample);
  CUDA_CHECK(cudaFree(d_blockMax));
  CUDA_CHECK(cudaFree(d_sample));

  return mx;
}

// =============================================================
// GPU mínimo inter-cluster para um par (A, B) — modo points, com tiling
// =============================================================
// Processa a grade SA×SB em tiles de TILE_A linhas de A por vez,
// mantendo d_blockMin com tamanho fixo (orçamento de memória controlado).
// Isso evita OOM mesmo quando SA*SB ultrapassa bilhões de pares.
//
// Orçamento: MAX_BLOCKMIN_FLOATS floats para d_blockMin (~64 MB).
#define MAX_BLOCKMIN_FLOATS (1 << 24)  // 16 M entradas = 64 MB

static float gpu_min_pair(const float* h_A, int SA,
                          const float* h_B, int SB,
                          int NF, int threads)
{
  if (SA <= 0 || SB <= 0) return FLT_MAX;

  int tpb = (threads > 0) ? threads : 256;

  // Quantas linhas de A cabem no orçamento de blocos?
  // blocks por tile = ceil(tile_A * SB / tpb) <= MAX_BLOCKMIN_FLOATS
  // tile_A <= MAX_BLOCKMIN_FLOATS * tpb / SB
  long long tile_A_ll = (long long)MAX_BLOCKMIN_FLOATS * tpb / (long long)SB;
  if (tile_A_ll < 1) tile_A_ll = 1;
  if (tile_A_ll > SA) tile_A_ll = SA;
  int tile_A = (int)tile_A_ll;

  // Blocos máximos por tile (para alocar d_blockMin uma vez só)
  long long max_pairs_tile = (long long)tile_A * SB;
  int max_blocks_tile = (int)((max_pairs_tile + tpb - 1) / tpb);
  if (max_blocks_tile < 1) max_blocks_tile = 1;

  // Aloca d_B inteiro na GPU (fica fixo durante todos os tiles)
  size_t bytes_B = (size_t)SB * NF * sizeof(float);
  float* d_B = NULL;
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  // Aloca d_A para o tile (tile_A linhas)
  size_t bytes_tile_A = (size_t)tile_A * NF * sizeof(float);
  float* d_A = NULL;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_tile_A));

  // Aloca d_blockMin com tamanho fixo (orçamento)
  float* d_blockMin = NULL;
  CUDA_CHECK(cudaMalloc(&d_blockMin, sizeof(float) * max_blocks_tile));

  float* h_blockMin = (float*)malloc(sizeof(float) * max_blocks_tile);
  if (!h_blockMin){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }

  size_t shmem = (size_t)tpb * sizeof(float);
  float global_min = FLT_MAX;

  // Itera sobre tiles de linhas de A
  for (int a_start = 0; a_start < SA; a_start += tile_A){
    int a_end  = a_start + tile_A;
    if (a_end > SA) a_end = SA;
    int cur_tile = a_end - a_start;

    // Copia tile atual de A para GPU
    size_t bytes_cur = (size_t)cur_tile * NF * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_A,
                          h_A + (size_t)a_start * NF,
                          bytes_cur,
                          cudaMemcpyHostToDevice));

    long long P_tile = (long long)cur_tile * SB;
    int blocks_tile  = (int)((P_tile + tpb - 1) / tpb);
    if (blocks_tile < 1) blocks_tile = 1;

    intercluster_min_kernel<<<blocks_tile, tpb, shmem>>>(
        d_A, cur_tile, d_B, SB, NF, P_tile, d_blockMin);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_blockMin, d_blockMin,
                          sizeof(float) * blocks_tile,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < blocks_tile; i++){
      if (h_blockMin[i] < global_min) global_min = h_blockMin[i];
    }
  }

  free(h_blockMin);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_blockMin));

  return global_min;
}

// =============================================================
// Separação mínima ponto-a-ponto (modo points) — itera K*(K-1)/2 pares
// =============================================================
static float gpu_min_intercluster_points(const float* X, int K, int NF,
                                         int** idx, int* sizes, int threads)
{
  float global_min = FLT_MAX;

  for (int ci = 0; ci < K; ci++){
    if (sizes[ci] <= 0) continue;

    // Gather cluster i no host
    size_t bytes_i = (size_t)sizes[ci] * NF * sizeof(float);
    float* h_A = (float*)malloc(bytes_i);
    if (!h_A){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
    gather_points(X, idx[ci], sizes[ci], NF, h_A);

    for (int cj = ci + 1; cj < K; cj++){
      if (sizes[cj] <= 0) continue;

      #if ENABLE_TIMING
      double t0 = wall_time_s();
      #endif

      // Gather cluster j no host
      size_t bytes_j = (size_t)sizes[cj] * NF * sizeof(float);
      float* h_B = (float*)malloc(bytes_j);
      if (!h_B){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }
      gather_points(X, idx[cj], sizes[cj], NF, h_B);

      float d2 = gpu_min_pair(h_A, sizes[ci], h_B, sizes[cj], NF, threads);

      #if ENABLE_TIMING
      double t1 = wall_time_s();
      fprintf(stderr, "[time] intercluster pair (%d,%d) [%d x %d pts]: %.6f s  dist²=%.6f\n",
              ci, cj, sizes[ci], sizes[cj], t1 - t0, d2);
      #endif

      if (d2 < global_min) global_min = d2;

      free(h_B);
    }

    free(h_A);
  }

  return global_min;  // distância² mínima
}

// ================= ESTRUTURA PARA STREAMS =================
typedef struct {
  cudaStream_t stream;
  float* h_sample;
  float* d_sample;
  float* d_blockMax;
  float* h_blockMax;
  int max_blocks;
} StreamContext;

static void init_stream_context(StreamContext* ctx, size_t max_sample_size, int NF, int threads){
  CUDA_CHECK(cudaStreamCreate(&ctx->stream));

  size_t sample_bytes = max_sample_size * NF * sizeof(float);

  CUDA_CHECK(cudaMallocHost(&ctx->h_sample, sample_bytes));
  CUDA_CHECK(cudaMalloc(&ctx->d_sample, sample_bytes));

  long long max_pairs   = (long long)max_sample_size * ((long long)max_sample_size - 1) / 2;
  long long max_threads = (max_pairs + PARES_POR_THREAD - 1) / PARES_POR_THREAD;
  ctx->max_blocks = (int)((max_threads + threads - 1) / threads);
  if (ctx->max_blocks < 1) ctx->max_blocks = 1;

  CUDA_CHECK(cudaMallocHost(&ctx->h_blockMax, sizeof(float) * ctx->max_blocks));
  CUDA_CHECK(cudaMalloc(&ctx->d_blockMax, sizeof(float) * ctx->max_blocks));
}

static void destroy_stream_context(StreamContext* ctx){
  CUDA_CHECK(cudaStreamDestroy(ctx->stream));
  CUDA_CHECK(cudaFreeHost(ctx->h_sample));
  CUDA_CHECK(cudaFree(ctx->d_sample));
  CUDA_CHECK(cudaFreeHost(ctx->h_blockMax));
  CUDA_CHECK(cudaFree(ctx->d_blockMax));
}

// ================= SKEVA COM STREAMS =================
static float skeva_cluster_with_streams(
    const float* h_X,
    const int* pool, int pool_sz,
    int NF, int sketch_size, int validate_size,
    int reps, int threads,
    Rng* rng,
    StreamContext* contexts)
{
  if (pool_sz <= 1) return 0.f;

  float best = 0.f;
  int tpb = (threads > 0) ? threads : 256;
  // shmem: kernel otimizado usa apenas (tpb/32) floats (um por warp)
  size_t shmem = (size_t)(tpb / 32) * sizeof(float);

  for (int r = 0; r < reps; r++){
    int s_idx = r % NUM_STREAMS;
    StreamContext* ctx = &contexts[s_idx];

    if (r >= NUM_STREAMS){
      CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    int *sk = NULL, sk_sz = 0;
    sample_indices(pool, pool_sz, sketch_size, rng, &sk, &sk_sz);

    if (sk_sz > 1){
      long long P_sk = (long long)sk_sz * (sk_sz - 1) / 2;
      long long thr_sk = (P_sk + PARES_POR_THREAD - 1) / PARES_POR_THREAD;
      int blocks_sk = (int)((thr_sk + tpb - 1) / tpb);
      if (blocks_sk < 1) blocks_sk = 1;
      size_t sample_bytes_sk = (size_t)sk_sz * NF * sizeof(float);

      gather_points(h_X, sk, sk_sz, NF, ctx->h_sample);

      CUDA_CHECK(cudaMemcpyAsync(ctx->d_sample, ctx->h_sample, sample_bytes_sk,
                                  cudaMemcpyHostToDevice, ctx->stream));

      pairwise_max_kernel_optimized<<<blocks_sk, tpb, shmem, ctx->stream>>>(
          ctx->d_sample, sk_sz, NF, P_sk, ctx->d_blockMax);

      CUDA_CHECK(cudaMemcpyAsync(ctx->h_blockMax, ctx->d_blockMax,
                                  sizeof(float) * blocks_sk,
                                  cudaMemcpyDeviceToHost, ctx->stream));

      CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

      float d1 = 0.f;
      for (int i = 0; i < blocks_sk; i++){
        if (ctx->h_blockMax[i] > d1) d1 = ctx->h_blockMax[i];
      }

      int *vl = NULL, vl_sz = 0;
      sample_indices(pool, pool_sz, validate_size, rng, &vl, &vl_sz);

      int *jn = NULL, jn_sz = 0;
      merge_unique(sk, sk_sz, vl, vl_sz, &jn, &jn_sz);
      free(vl);

      if (jn_sz > 1){
        long long P_jn = (long long)jn_sz * (jn_sz - 1) / 2;
        long long thr_jn = (P_jn + PARES_POR_THREAD - 1) / PARES_POR_THREAD;
        int blocks_jn = (int)((thr_jn + tpb - 1) / tpb);
        if (blocks_jn < 1) blocks_jn = 1;
        size_t sample_bytes_jn = (size_t)jn_sz * NF * sizeof(float);

        gather_points(h_X, jn, jn_sz, NF, ctx->h_sample);

        CUDA_CHECK(cudaMemcpyAsync(ctx->d_sample, ctx->h_sample, sample_bytes_jn,
                                    cudaMemcpyHostToDevice, ctx->stream));

        pairwise_max_kernel_optimized<<<blocks_jn, tpb, shmem, ctx->stream>>>(
            ctx->d_sample, jn_sz, NF, P_jn, ctx->d_blockMax);

        CUDA_CHECK(cudaMemcpyAsync(ctx->h_blockMax, ctx->d_blockMax,
                                    sizeof(float) * blocks_jn,
                                    cudaMemcpyDeviceToHost, ctx->stream));

        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        float d2 = 0.f;
        for (int i = 0; i < blocks_jn; i++){
          if (ctx->h_blockMax[i] > d2) d2 = ctx->h_blockMax[i];
        }

        float d = (d1 > d2) ? d1 : d2;
        if (d > best) best = d;
      } else {
        if (d1 > best) best = d1;
      }

      free(jn);
    }

    free(sk);
  }

  return best;
}

// ================= SKEVA SEM STREAMS (fallback) =================
static float skeva_cluster_simple(
    const float* h_X,
    const int* pool, int pool_sz,
    int NF, int sketch_size, int validate_size,
    int reps, int threads,
    Rng* rng)
{
  if (pool_sz <= 1) return 0.f;

  float best = 0.f;

  for (int r = 0; r < reps; r++){
    int *sk = NULL, sk_sz = 0;
    sample_indices(pool, pool_sz, sketch_size, rng, &sk, &sk_sz);

    float d1 = gpu_pairwise_max_simple(h_X, sk, sk_sz, NF, threads);

    int *vl = NULL, vl_sz = 0;
    sample_indices(pool, pool_sz, validate_size, rng, &vl, &vl_sz);

    int *jn = NULL, jn_sz = 0;
    merge_unique(sk, sk_sz, vl, vl_sz, &jn, &jn_sz);
    free(vl);

    float d2 = gpu_pairwise_max_simple(h_X, jn, jn_sz, NF, threads);

    float d = (d1 > d2) ? d1 : d2;
    if (d > best) best = d;

    free(jn);
    free(sk);
  }

  return best;
}

// ================= PRINT HELPERS =================
static void print_memory_info(void){
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  fprintf(stderr, "GPU Memory: %.2f MB free / %.2f MB total\n",
          (double)free_mem / (1024*1024),
          (double)total_mem / (1024*1024));
}

static void print_centroids(const char* tag, const float* C, int K, int NF){
  printf("%s\n", tag);
  int max_features = (NF > 10) ? 10 : NF;
  for (int c = 0; c < K; ++c){
    printf("  c%-3d:", c);
    for (int f = 0; f < max_features; ++f){
      printf(" %.4g", C[(size_t)c*NF + f]);
    }
    if (NF > 10) printf(" ...");
    printf("\n");
  }
}

// ================= MAIN =================
int main(int argc, char** argv){
  Args A;
  parse_args(argc, argv, &A);

  print_memory_info();

  // 1) Ler dados (modo selecionado pelos args)
  float* X = NULL;
  int* y = NULL;
  size_t N = 0;
  int NF = 0, K = 0;

  if (A.data_file != NULL && A.labels_file != NULL)
    read_dataset_kmeans(&A, &X, &y, &N, &NF, &K);
  else
    read_dataset(&A, &X, &y, &N, &NF, &K);

  A.NF = NF;
  A.K  = K;

  fprintf(stderr, "Dataset: N=%zu, NF=%d, K=%d\n", N, NF, K);
  if (A.sample_pct > 0.f){
    fprintf(stderr, "Parâmetros SkeVa: sketch=%.1f%% do cluster, validate=%d, reps=%d\n",
            A.sample_pct, A.validate_size, A.reps);
  } else {
    fprintf(stderr, "Parâmetros SkeVa: sketch=%d (fixo), validate=%d, reps=%d\n",
            A.sketch_size, A.validate_size, A.reps);
  }
  fprintf(stderr, "Streams: %s\n", A.use_streams ? "habilitado" : "desabilitado");
  fprintf(stderr, "Modo inter-cluster: %s\n",
          A.inter_mode == INTER_MODE_POINTS ? "points (ponto-a-ponto, exato)" : "centroids (centróides)");

  if (A.inter_mode == INTER_MODE_POINTS){
    fprintf(stderr, "AVISO: modo 'points' é O(N²) — pode ser muito lento para datasets grandes.\n");
  }

  // max_sample calculado após build_clusters (ver abaixo)
  size_t sample_mem_approx = ((size_t)A.sketch_size + A.validate_size) * NF * sizeof(float);
  fprintf(stderr, "Memória por amostra (estimada): %.2f MB (vs %.2f MB do dataset completo)\n",
          (double)sample_mem_approx / (1024*1024),
          (double)(N * NF * sizeof(float)) / (1024*1024));

  #if ENABLE_TIMING
  double t_total0 = wall_time_s();
  #endif

  // 2) Índices por cluster
  #if ENABLE_TIMING
  double t_idx0 = wall_time_s();
  #endif

  int** idx = NULL;
  int* sizes = NULL;
  build_clusters(y, N, K, &idx, &sizes);

  for (int c = 0; c < K; c++){
    fprintf(stderr, "Cluster %d: %d pontos\n", c, sizes[c]);
  }

  #if ENABLE_TIMING
  double t_idx1 = wall_time_s();
  fprintf(stderr, "[time] build_clusters           : %.6f s\n", t_idx1 - t_idx0);
  #endif

  // max_sample: tamanho máximo do buffer de amostra para os streams
  // Se sample_pct ativo, o sketch efetivo pode ser maior que sketch_size fixo
  size_t max_sample;
  if (A.sample_pct > 0.f){
    int max_cluster_sz = 0;
    for (int c = 0; c < K; c++)
      if (sizes[c] > max_cluster_sz) max_cluster_sz = sizes[c];
    int max_eff_sketch = (int)(max_cluster_sz * A.sample_pct / 100.f);
    if (max_eff_sketch < 2) max_eff_sketch = 2;
    max_sample = (size_t)max_eff_sketch + A.validate_size;
  } else {
    max_sample = (size_t)A.sketch_size + A.validate_size;
  }

  // 3) Centróides na CPU (sempre calculados — usados para exibição)
  #if ENABLE_TIMING
  double t_cent0 = wall_time_s();
  #endif

  float* C = NULL;
  compute_centroids_cpu(X, NF, K, idx, sizes, &C);

  #if ENABLE_TIMING
  double t_cent1 = wall_time_s();
  fprintf(stderr, "[time] centroids (CPU)          : %.6f s\n", t_cent1 - t_cent0);
  #endif

  print_centroids("Centroids:", C, K, NF);

  // 4) Separação mínima — modo selecionado pelo usuário
  #if ENABLE_TIMING
  double t_sep0 = wall_time_s();
  #endif

  float minsep;
  if (A.inter_mode == INTER_MODE_CENTROIDS){
    minsep = min_centroid_separation(C, K, NF);
    fprintf(stderr, "Min centroid separation (sq)    : %.6f\n", minsep);
  } else {
    // Modo points: O(N²) na GPU
    fprintf(stderr, "Calculando separação ponto-a-ponto na GPU...\n");
    minsep = gpu_min_intercluster_points(X, K, NF, idx, sizes, A.threads);
    fprintf(stderr, "Min point separation (sq)       : %.6f\n", minsep);
  }

  #if ENABLE_TIMING
  double t_sep1 = wall_time_s();
  fprintf(stderr, "[time] inter-cluster separation : %.6f s\n", t_sep1 - t_sep0);
  #endif

  // 5) Inicializa streams
  StreamContext stream_contexts[NUM_STREAMS];
  if (A.use_streams){
    #if ENABLE_TIMING
    double t_stream0 = wall_time_s();
    #endif

    for (int i = 0; i < NUM_STREAMS; i++){
      init_stream_context(&stream_contexts[i], max_sample, NF, A.threads);
    }

    #if ENABLE_TIMING
    double t_stream1 = wall_time_s();
    fprintf(stderr, "[time] init streams             : %.6f s\n", t_stream1 - t_stream0);
    #endif
  }

  // 6) SkeVa por cluster (diâmetro intra-cluster)
  #if ENABLE_TIMING
  double t_skeva0 = wall_time_s();
  #endif

  Rng rng;
  rng_seed(&rng, A.seed);

  float* diam_est = (float*)calloc(K, sizeof(float));
  if (!diam_est){ fprintf(stderr,"OOM\n"); exit(EXIT_FAILURE); }

  for (int c = 0; c < K; c++){
    int pool_sz = sizes[c];
    if (pool_sz <= 1){ diam_est[c] = 0.f; continue; }

    #if ENABLE_TIMING
    double t_c0 = wall_time_s();
    #endif

    const int* pool = idx[c];

    // sketch_size efetivo: fixo ou percentual do cluster
    int eff_sketch;
    if (A.sample_pct > 0.f){
      eff_sketch = (int)(pool_sz * A.sample_pct / 100.f);
      if (eff_sketch < 2) eff_sketch = 2;
      if (eff_sketch > pool_sz) eff_sketch = pool_sz;
      fprintf(stderr, "Cluster %d: sketch_size efetivo = %d/%d (%.1f%%)\n",
              c, eff_sketch, pool_sz, A.sample_pct);
    } else {
      eff_sketch = A.sketch_size;
    }

    if (A.use_streams){
      diam_est[c] = skeva_cluster_with_streams(
          X, pool, pool_sz, NF,
          eff_sketch, A.validate_size, A.reps, A.threads,
          &rng, stream_contexts);
    } else {
      diam_est[c] = skeva_cluster_simple(
          X, pool, pool_sz, NF,
          eff_sketch, A.validate_size, A.reps, A.threads,
          &rng);
    }

    fprintf(stderr, "Cluster %d: SkeVa diameter (sq) ≈ %.6f\n", c, diam_est[c]);

    #if ENABLE_TIMING
    double t_c1 = wall_time_s();
    fprintf(stderr, "[time] SkeVa cluster %d          : %.6f s\n", c, t_c1 - t_c0);
    #endif
  }

  #if ENABLE_TIMING
  double t_skeva1 = wall_time_s();
  fprintf(stderr, "[time] SkeVa (todos clusters)   : %.6f s\n", t_skeva1 - t_skeva0);
  #endif

  // 7) Cleanup streams
  if (A.use_streams){
    for (int i = 0; i < NUM_STREAMS; i++){
      destroy_stream_context(&stream_contexts[i]);
    }
  }

  // 8) Máximo diâmetro e Dunn Index
  float maxdiam = 0.f;
  for (int c = 0; c < K; c++){
    if (diam_est[c] > maxdiam) maxdiam = diam_est[c];
  }

  float dunn = (maxdiam > 0.f) ? sqrtf(minsep / maxdiam) : 0.f;

  printf("\n========== RESULTADOS ==========\n");
  if (A.inter_mode == INTER_MODE_CENTROIDS){
    printf("Min centroid separation : %.9f\n", sqrtf(minsep));
  } else {
    printf("Min point separation    : %.9f\n", sqrtf(minsep));
  }
  printf("Max (SkeVa) diameter    : %.9f\n", sqrtf(maxdiam));
  printf("Dunn Index (SkeVa)      : %.9f\n", dunn);

  #if ENABLE_TIMING
  double t_total1 = wall_time_s();
  fprintf(stderr, "\n[time] TOTAL (pós-leitura)      : %.6f s\n", t_total1 - t_total0);
  #endif

  // Cleanup
  free(diam_est);
  free(C);
  for (int c = 0; c < K; c++) free(idx[c]);
  free(idx);
  free(sizes);
  free(X);
  free(y);

  print_memory_info();

  return 0;
}
