// kmeanspp_init.c
// Gera centróides iniciais k-means++ (Arthur & Vassilvitskii, 2007) para o k-means.
//
// O k-means++ é incremental: os K primeiros centróides da sequência gerada para
// Kmax são uma inicialização k-means++ válida para K. Assim, o dataset é lido uma
// única vez e é gravado um arquivo por K:
//   <out_dir>/init_k<K>.txt   (K linhas x NF colunas)   para K em [kmin, kmax]
//
// Uso:
//   ./kmeanspp_init --file <dataset> --npoints N --ndims NF
//                   --kmin Kmin --kmax Kmax --out_dir DIR [--seed S]
//
// Compilar:
//   gcc -O3 -fopenmp kmeanspp_init.c -o kmeanspp_init -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// --------- PRNG simples (xorshift64*), mesmo do dunn_skeva ----------
typedef struct { unsigned long long s; } Rng;
static inline void rng_seed(Rng* r, unsigned long long seed){ r->s = seed?seed:0x9e3779b97f4a7c15ULL; }
static inline unsigned long long rng_u64(Rng* r){
  unsigned long long x = r->s;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27; r->s = x;
  return x * 2685821657736338717ULL;
}
// uniforme em [0, 1)
static inline double rng_unif(Rng* r){ return (double)(rng_u64(r) >> 11) * (1.0 / 9007199254740992.0); }

int main(int argc, char** argv){
  const char* file = NULL;
  const char* out_dir = NULL;
  long long N = -1;
  int NF = -1, kmin = -1, kmax = -1;
  unsigned long long seed = 1ULL;

  for (int i=1;i<argc;i++){
    if      (!strcmp(argv[i],"--file")    && i+1<argc) file    = argv[++i];
    else if (!strcmp(argv[i],"--out_dir") && i+1<argc) out_dir = argv[++i];
    else if (!strcmp(argv[i],"--npoints") && i+1<argc) N    = atoll(argv[++i]);
    else if (!strcmp(argv[i],"--ndims")   && i+1<argc) NF   = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--kmin")    && i+1<argc) kmin = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--kmax")    && i+1<argc) kmax = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--seed")    && i+1<argc) seed = (unsigned long long)atoll(argv[++i]);
    else { fprintf(stderr,"Arg inválido: %s\n", argv[i]); return EXIT_FAILURE; }
  }
  if (!file || !out_dir || N <= 0 || NF <= 0 || kmin <= 0 || kmax < kmin){
    fprintf(stderr,
      "Uso: %s --file <dataset> --npoints N --ndims NF --kmin Kmin --kmax Kmax --out_dir DIR [--seed S]\n",
      argv[0]);
    return EXIT_FAILURE;
  }
  if (kmax > N){ fprintf(stderr,"Erro: kmax (%d) > npoints (%lld)\n", kmax, N); return EXIT_FAILURE; }

  // --------- leitura do dataset (N linhas x NF floats) ----------
  float* X = (float*)malloc(sizeof(float) * (size_t)N * NF);
  double* D = (double*)malloc(sizeof(double) * (size_t)N);   // dist² ao centróide mais próximo
  float* C = (float*)malloc(sizeof(float) * (size_t)kmax * NF);
  if (!X || !D || !C){ fprintf(stderr,"OOM\n"); return EXIT_FAILURE; }

  FILE* fp = fopen(file, "r");
  if (!fp){ perror("fopen dataset"); return EXIT_FAILURE; }
  for (long long i=0;i<N;i++){
    for (int f=0;f<NF;f++){
      if (fscanf(fp, " %f", &X[(size_t)i*NF + f]) != 1){
        fprintf(stderr,"Erro lendo ponto %lld feature %d\n", i, f);
        return EXIT_FAILURE;
      }
    }
  }
  fclose(fp);

  Rng rng;
  rng_seed(&rng, seed);

  // --------- k-means++ ----------
  // 1º centróide: ponto uniforme
  long long pick = (long long)(rng_unif(&rng) * (double)N);
  memcpy(&C[0], &X[(size_t)pick*NF], sizeof(float)*NF);
  for (long long i=0;i<N;i++) D[i] = DBL_MAX;

  for (int k=1;k<=kmax;k++){
    // Atualiza D com o centróide k-1 recém-escolhido e soma os pesos
    const float* c = &C[(size_t)(k-1)*NF];
    double total = 0.0;
    #pragma omp parallel for reduction(+:total)
    for (long long i=0;i<N;i++){
      const float* xi = &X[(size_t)i*NF];
      double s = 0.0;
      for (int f=0;f<NF;f++){
        double d = (double)xi[f] - (double)c[f];
        s += d*d;
      }
      if (s < D[i]) D[i] = s;
      total += D[i];
    }
    if (k == kmax) break;

    // Próximo centróide: probabilidade proporcional a D(x)²
    if (total > 0.0){
      double r = rng_unif(&rng) * total;
      double acc = 0.0;
      pick = N - 1;
      for (long long i=0;i<N;i++){
        acc += D[i];
        if (acc > r){ pick = i; break; }
      }
    } else {
      // Todos os pontos coincidem com centróides já escolhidos
      pick = (long long)(rng_unif(&rng) * (double)N);
    }
    memcpy(&C[(size_t)k*NF], &X[(size_t)pick*NF], sizeof(float)*NF);
  }

  // --------- grava init_k<K>.txt (prefixo da sequência) ----------
  for (int K=kmin;K<=kmax;K++){
    char path[4096];
    snprintf(path, sizeof(path), "%s/init_k%d.txt", out_dir, K);
    FILE* fo = fopen(path, "w");
    if (!fo){ perror("fopen saída"); return EXIT_FAILURE; }
    for (int c=0;c<K;c++){
      for (int f=0;f<NF;f++) fprintf(fo, "%.9g ", C[(size_t)c*NF + f]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  fprintf(stderr, "k-means++: centróides para K=%d..%d gravados em %s\n", kmin, kmax, out_dir);

  free(C); free(D); free(X);
  return 0;
}
