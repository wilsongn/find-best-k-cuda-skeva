#!/usr/bin/env bash
# kmeans_dunn_eval.sh
# Avalia Dunn Index (via SkeVa CUDA) para uma faixa de K usando o k-means k-means.
#
# Para cada K em [Kmin, Kmax]:
#   1. Copia os fontes do k-means para um diretório temporário
#   2. Patcha main.h com os valores corretos (NbPoints, NbDims, NbClusters, INPUT_DATA)
#   3. Compila e executa o k-means  →  Labels.txt + FinalCentroids.txt
#   4. Executa dunn_skeva (modo --data_file / --labels_file)
#   5. Coleta o Dunn Index
# Ao final imprime tabela e destaca o melhor K.
#
# Uso:
#   ./kmeans_dunn_eval.sh --file <dataset> --npoints N --ndims NF \
#                         --kmin Kmin --kmax Kmax [opções]
#
# Opções k-means:
#   --npackages P           NbPackages [100]
#   --kmeans_target GPU|CPU alvo de execução [GPU]
#   --kmeans_threads T      threads OpenMP (só CPU) [1]
#   --kmeans_iters I        max iterações [200]
#   --kmeans_tol TOL        tolerância [1e-4]
#   --init random|kmeanspp  inicialização dos centróides [random]
#                           random   = sorteio uniforme do próprio k-means
#                           kmeanspp = k-means++ (via kmeanspp_init)
#   --init_seed N           semente do k-means++ [1]
#   --init_centroids_dir D  usa D/init_k<K>.txt como centróides iniciais
#                           (K linhas x NF colunas); incompatível com --init kmeanspp
#
# Opções Dunn/SkeVa:
#   --sketch_size S         [512]
#   --validate_size V       [512]
#   --reps R                [8]
#   --dunn_threads T        threads por bloco CUDA [256]
#   --seed N                [42]
#   --inter_mode M          centroids|points [centroids]
#   --sample_pct P          [0]
#   --no-streams
#   --prune                 branch-and-bound: calcula a cota sep_min/raio_max
#                           (tempo linear) para todos os K, roda o SkeVa no K
#                           de maior cota (cota inicial) e descarta os K cuja
#                           cota não supera o melhor Dunn já obtido
#
# Caminhos (detectados automaticamente se estiverem no mesmo diretório do script):
#   --kmeans_dir DIR        diretório dos fontes do k-means [<script_dir>/k-means]
#   --dunn_bin BIN          binário dunn [<script_dir>/dunn_skeva]
#   --kmeanspp_bin BIN      binário k-means++ [<script_dir>/kmeanspp_init]

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FILE=""
N=""
NF=""
KMIN=""
KMAX=""
NPACKAGES=100
KMEANS_TARGET="GPU"
KMEANS_THREADS=1
KMEANS_ITERS=200
KMEANS_TOL="1.0E-4"
INIT_CENTROIDS_DIR=""
INIT_MODE="random"
INIT_SEED=1

SKETCH_SIZE=512
VALIDATE_SIZE=512
REPS=8
DUNN_THREADS=256
SEED=42
INTER_MODE="centroids"
SAMPLE_PCT=0
USE_STREAMS=1
PRUNE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KMEANS_DIR="${SCRIPT_DIR}/k-means"
DUNN_BIN="${SCRIPT_DIR}/dunn_skeva"
KMEANSPP_BIN="${SCRIPT_DIR}/kmeanspp_init"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)          FILE="$2";            shift 2 ;;
    --npoints)       N="$2";               shift 2 ;;
    --ndims)         NF="$2";              shift 2 ;;
    --kmin)          KMIN="$2";            shift 2 ;;
    --kmax)          KMAX="$2";            shift 2 ;;
    --npackages)     NPACKAGES="$2";       shift 2 ;;
    --kmeans_target) KMEANS_TARGET="$2";   shift 2 ;;
    --kmeans_threads)KMEANS_THREADS="$2";  shift 2 ;;
    --kmeans_iters)  KMEANS_ITERS="$2";    shift 2 ;;
    --kmeans_tol)    KMEANS_TOL="$2";      shift 2 ;;
    --init_centroids_dir) INIT_CENTROIDS_DIR="$(realpath "$2")"; shift 2 ;;
    --init)          INIT_MODE="$2";       shift 2 ;;
    --init_seed)     INIT_SEED="$2";       shift 2 ;;
    --kmeanspp_bin)  KMEANSPP_BIN="$2";    shift 2 ;;
    --sketch_size)   SKETCH_SIZE="$2";     shift 2 ;;
    --validate_size) VALIDATE_SIZE="$2";   shift 2 ;;
    --reps)          REPS="$2";            shift 2 ;;
    --dunn_threads)  DUNN_THREADS="$2";    shift 2 ;;
    --seed)          SEED="$2";            shift 2 ;;
    --inter_mode)    INTER_MODE="$2";      shift 2 ;;
    --sample_pct)    SAMPLE_PCT="$2";      shift 2 ;;
    --no-streams)    USE_STREAMS=0;        shift   ;;
    --prune)         PRUNE=1;              shift   ;;
    --kmeans_dir)    KMEANS_DIR="$2";      shift 2 ;;
    --dunn_bin)      DUNN_BIN="$2";        shift 2 ;;
    *)
      echo "Argumento desconhecido: $1" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Validações básicas
# ---------------------------------------------------------------------------
if [[ -z "$FILE" || -z "$N" || -z "$NF" || -z "$KMIN" || -z "$KMAX" ]]; then
  echo "Uso: $0 --file <dataset> --npoints N --ndims NF --kmin Kmin --kmax Kmax [opções]" >&2
  exit 1
fi

# Converter path do dataset para absoluto
FILE="$(realpath "$FILE")"

if [[ ! -f "$FILE" ]]; then
  echo "Erro: dataset não encontrado: $FILE" >&2
  exit 1
fi
if [[ ! -d "$KMEANS_DIR" ]]; then
  echo "Erro: diretório do k-means não encontrado: $KMEANS_DIR" >&2
  exit 1
fi
if [[ ! -f "$DUNN_BIN" ]]; then
  echo "Erro: binário dunn não encontrado: $DUNN_BIN" >&2
  echo "  Compile com: nvcc -O3 -arch=sm_75 dunn_skeva.cu -o dunn_skeva -lm" >&2
  exit 1
fi
if [[ "$KMIN" -gt "$KMAX" ]]; then
  echo "Erro: kmin ($KMIN) > kmax ($KMAX)" >&2
  exit 1
fi
if [[ "$INIT_MODE" != "random" && "$INIT_MODE" != "kmeanspp" ]]; then
  echo "Erro: --init inválido: $INIT_MODE (use 'random' ou 'kmeanspp')" >&2
  exit 1
fi
if [[ "$INIT_MODE" == "kmeanspp" ]]; then
  if [[ -n "$INIT_CENTROIDS_DIR" ]]; then
    echo "Erro: --init kmeanspp e --init_centroids_dir são incompatíveis" >&2
    exit 1
  fi
  if [[ ! -f "$KMEANSPP_BIN" ]]; then
    echo "Erro: binário k-means++ não encontrado: $KMEANSPP_BIN" >&2
    echo "  Compile com: gcc -O3 -fopenmp kmeanspp_init.c -o kmeanspp_init -lm" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Resultados por K (arrays associativos indexados por K)
# ---------------------------------------------------------------------------
declare -a K_LIST=()
declare -A LABELS_OF=()
declare -A UB_OF=()
declare -A UB_TIME_OF=()
declare -A DUNN_OF=()
declare -A DUNN_TIME_OF=()

# Tempo de parede somado de todas as chamadas ao dunn_skeva (inclui leitura)
DUNN_WALL_TOTAL=0

# Executa dunn_skeva para um K.  Uso: run_dunn <K> <labels> [args extras]
# Define DUNN_INDEX, DUNN_UB e DUNN_TIME (ou "N/A").
run_dunn() {
  local k="$1" labels="$2"; shift 2
  local args="--data_file ${FILE} --labels_file ${labels} --nf ${NF}"
  args="$args --sketch_size ${SKETCH_SIZE} --validate_size ${VALIDATE_SIZE}"
  args="$args --reps ${REPS} --threads ${DUNN_THREADS} --seed ${SEED}"
  args="$args --inter_mode ${INTER_MODE}"
  if [[ "$SAMPLE_PCT" != "0" ]]; then
    args="$args --sample_pct ${SAMPLE_PCT}"
  fi
  if [[ "$USE_STREAMS" -eq 0 ]]; then
    args="$args --no-streams"
  fi

  # Captura stdout (resultados) e stderr (timings) separadamente
  # O || true impede que set -e mate o script caso o dunn retorne exit code != 0
  local stderr_file="${TMPBASE}/k${k}/dunn_stderr.txt"
  local out
  local t0 t1
  t0=$(date +%s.%N)
  out=$("$DUNN_BIN" $args "$@" 2>"$stderr_file") || true
  t1=$(date +%s.%N)
  DUNN_WALL_TOTAL=$(awk -v a="$DUNN_WALL_TOTAL" -v b="$t0" -v c="$t1" 'BEGIN { printf "%.3f", a + (c - b) }')

  DUNN_INDEX=$(echo "$out" | grep "Dunn Index (SkeVa)" | awk '{print $NF}') || true
  DUNN_UB=$(echo "$out" | grep "Dunn upper bound" | awk '{print $NF}') || true
  DUNN_TIME=$(grep "TOTAL" "$stderr_file" | awk '{print $(NF-1)}') || true

  if [[ -z "$DUNN_UB" ]]; then
    echo "Aviso: dunn_skeva falhou para K=$k. Erro:" >&2
    cat "$stderr_file" >&2
    DUNN_UB="N/A"
  fi
  [[ -z "$DUNN_INDEX" ]] && DUNN_INDEX="N/A"
  [[ -z "$DUNN_TIME"  ]] && DUNN_TIME="N/A"
  return 0
}

# ---------------------------------------------------------------------------
# Fase 1: k-means para cada K (+ cota superior, se --prune)
# ---------------------------------------------------------------------------
echo "========================================================"
echo " kmeans_dunn_eval  —  K de $KMIN a $KMAX"
echo " Dataset : $FILE"
echo " N=$N  NF=$NF  alvo_kmeans=$KMEANS_TARGET"
echo " Descarte por cota superior: $([[ "$PRUNE" -eq 1 ]] && echo habilitado || echo desabilitado)"
echo " Inicialização dos centróides: $INIT_MODE"
echo "========================================================"

TMPBASE=$(mktemp -d /tmp/kmeans_dunn_XXXXXX)
trap 'rm -rf "$TMPBASE"' EXIT

# Detectar compute capability da GPU para a flag --gpu-architecture
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [[ -z "$GPU_CC" ]]; then
  GPU_CC="75"  # fallback
  echo "Aviso: não foi possível detectar compute capability, usando sm_75" >&2
fi
GPU_ARCH="sm_${GPU_CC}"
echo " GPU arch detectada: ${GPU_ARCH}"

# k-means++: uma única passada gera os centróides iniciais de todos os K
if [[ "$INIT_MODE" == "kmeanspp" ]]; then
  INIT_CENTROIDS_DIR="${TMPBASE}/init_kmeanspp"
  mkdir -p "$INIT_CENTROIDS_DIR"
  echo " [k-means++] Gerando centróides iniciais para K=$KMIN..$KMAX (seed=$INIT_SEED)..."
  T_INIT0=$(date +%s.%N)
  if ! "$KMEANSPP_BIN" --file "$FILE" --npoints "$N" --ndims "$NF" \
       --kmin "$KMIN" --kmax "$KMAX" --seed "$INIT_SEED" --out_dir "$INIT_CENTROIDS_DIR"; then
    echo "Erro: kmeanspp_init falhou" >&2
    exit 1
  fi
  T_INIT1=$(date +%s.%N)
  INIT_TIME=$(awk -v a="$T_INIT0" -v b="$T_INIT1" 'BEGIN { printf "%.3f", b - a }')
  echo " [k-means++] Tempo: ${INIT_TIME}s"
fi

for K in $(seq "$KMIN" "$KMAX"); do
  echo ""
  echo "-------- K = $K ----------------------------------------"

  K_WORKDIR="${TMPBASE}/k${K}"
  mkdir -p "$K_WORKDIR"

  # ---- 1) Copiar fontes do k-means para temp dir ----
  cp "${KMEANS_DIR}"/*.cc "${KMEANS_DIR}"/*.cu \
     "${KMEANS_DIR}"/*.h  "${KMEANS_DIR}/Makefile" \
     "$K_WORKDIR"/

  # ---- 2) Patchar main.h ----
  # Escapa o path para uso seguro no sed (substitui / por \/)
  FILE_ESC="${FILE//\//\\/}"

  sed -i \
    -e "s|^#define NbPoints[[:space:]].*|#define NbPoints    ${N}|" \
    -e "s|^#define NbDims[[:space:]].*|#define NbDims      ${NF}|" \
    -e "s|^#define NbClusters[[:space:]].*|#define NbClusters  ${K}|" \
    -e "s|^#define NbPackages[[:space:]].*|#define NbPackages  ${NPACKAGES}|" \
    -e "s|^#define INPUT_DATA[[:space:]].*|#define INPUT_DATA  \"${FILE_ESC}\"|" \
    "${K_WORKDIR}/main.h"

  if [[ -n "$INIT_CENTROIDS_DIR" ]]; then
    INIT_FILE="${INIT_CENTROIDS_DIR}/init_k${K}.txt"
    if [[ ! -f "$INIT_FILE" ]]; then
      echo "Erro: centróides iniciais não encontrados: $INIT_FILE" >&2
      exit 1
    fi
    sed -i "s|^#define INPUT_INITIAL_CENTROIDS[[:space:]].*|#define INPUT_INITIAL_CENTROIDS  \"${INIT_FILE}\"|" \
      "${K_WORKDIR}/main.h"
  fi

  # ---- 2b) Patchar Makefile do k-means (adaptar ao ambiente local) ----
  # - Atualiza GPU arch
  # - Remove -lopenblas (não utilizado no código, apenas no Makefile original)
  # - Substitui -march=skylake-avx512 por -march=native (compatível com qualquer CPU)
  NVCC_PATH=$(which nvcc)
  sed -i \
    -e "s|GPUCC = .*|GPUCC = ${NVCC_PATH}|" \
    -e "s|--gpu-architecture=sm_[0-9]*|--gpu-architecture=${GPU_ARCH}|g" \
    -e "s|-lopenblas||g" \
    -e "s|-march=skylake-avx512|-march=native|g" \
    "${K_WORKDIR}/Makefile"

  # ---- 3) Compilar k-means ----
  echo "[kmeans] Compilando para K=$K..."
  if ! make -C "$K_WORKDIR" -f Makefile EXECNAME="kmeans" --silent 2>&1; then
    echo "Erro: compilação do k-means falhou para K=$K" >&2
    continue
  fi

  # ---- 4) Executar k-means ----
  echo "[kmeans] Executando..."
  KMEANS_ARGS="-max-iters ${KMEANS_ITERS} -tol ${KMEANS_TOL}"
  if [[ "$KMEANS_TARGET" == "CPU" ]]; then
    KMEANS_ARGS="$KMEANS_ARGS -t CPU -cpu-nt ${KMEANS_THREADS}"
  else
    KMEANS_ARGS="$KMEANS_ARGS -t GPU"
  fi

  # Roda no K_WORKDIR para que Labels.txt e demais arquivos de saída fiquem lá
  if ! (cd "$K_WORKDIR" && ./kmeans $KMEANS_ARGS); then
    echo "Erro: execução do k-means falhou para K=$K" >&2
    continue
  fi

  LABELS_FILE="${K_WORKDIR}/Labels.txt"
  if [[ ! -f "$LABELS_FILE" ]]; then
    echo "Erro: Labels.txt não gerado para K=$K" >&2
    continue
  fi

  # ---- 5) Cota superior (com --prune) ou Dunn completo ----
  K_LIST+=("$K")
  LABELS_OF[$K]="$LABELS_FILE"

  if [[ "$PRUNE" -eq 1 ]]; then
    echo "[dunn] Calculando cota superior sep_min/raio_max para K=$K..."
    run_dunn "$K" "$LABELS_FILE" --bound_only
    UB_OF[$K]="$DUNN_UB"
    UB_TIME_OF[$K]="$DUNN_TIME"
    echo "[cota] K=$K  cota superior = $DUNN_UB  Tempo = ${DUNN_TIME}s"
  else
    echo "[dunn] Calculando Dunn Index para K=$K..."
    run_dunn "$K" "$LABELS_FILE"
    UB_OF[$K]="$DUNN_UB"
    UB_TIME_OF[$K]="-"
    DUNN_OF[$K]="$DUNN_INDEX"
    DUNN_TIME_OF[$K]="$DUNN_TIME"
    echo "[resultado] K=$K  Dunn Index = $DUNN_INDEX  (cota sup. = $DUNN_UB)  Tempo DI = ${DUNN_TIME}s"
  fi
done

# ---------------------------------------------------------------------------
# Fase 2 (--prune): SkeVa em ordem decrescente de cota superior.
# O K de maior cota (mais promissor) é avaliado primeiro e fornece a cota
# inicial. Como a lista está ordenada, assim que a cota de um K não supera o
# melhor Dunn obtido, todos os K restantes também são descartados.
# ---------------------------------------------------------------------------
if [[ "$PRUNE" -eq 1 ]]; then
  echo ""
  echo "-------- Fase 2: SkeVa por ordem de cota superior ----------"

  RUN_BEST="-1"
  ORDER=$(for K in "${K_LIST[@]}"; do
            [[ "${UB_OF[$K]}" != "N/A" ]] && echo "${UB_OF[$K]} $K"
          done | sort -g -r -k1,1 | awk '{print $2}')

  for K in "${K_LIST[@]}"; do
    if [[ "${UB_OF[$K]}" == "N/A" ]]; then
      DUNN_OF[$K]="N/A"; DUNN_TIME_OF[$K]="N/A"
    fi
  done

  for K in $ORDER; do
    UB="${UB_OF[$K]}"
    if [[ "$RUN_BEST" != "-1" ]]; then
      CAN_BEAT=$(awk -v u="$UB" -v b="$RUN_BEST" 'BEGIN { print (u > b) ? 1 : 0 }')
      if [[ "$CAN_BEAT" -eq 0 ]]; then
        echo "[descarte] K=$K  cota superior = $UB <= melhor Dunn = $RUN_BEST"
        DUNN_OF[$K]="PRUNED"; DUNN_TIME_OF[$K]="-"
        continue
      fi
    fi

    if [[ "$RUN_BEST" == "-1" ]]; then
      echo "[dunn] K=$K (maior cota = $UB) → SkeVa define a cota inicial"
    else
      echo "[dunn] K=$K  cota superior = $UB > melhor = $RUN_BEST → SkeVa"
    fi
    run_dunn "$K" "${LABELS_OF[$K]}"
    DUNN_OF[$K]="$DUNN_INDEX"
    DUNN_TIME_OF[$K]="$DUNN_TIME"
    echo "[resultado] K=$K  Dunn Index = $DUNN_INDEX  Tempo DI = ${DUNN_TIME}s"

    if [[ "$DUNN_INDEX" != "N/A" ]]; then
      IS_BETTER=$(awk -v a="$DUNN_INDEX" -v b="$RUN_BEST" 'BEGIN { print (a > b) ? 1 : 0 }')
      if [[ "$IS_BETTER" -eq 1 ]]; then
        RUN_BEST="$DUNN_INDEX"
      fi
    fi
  done
fi

# ---------------------------------------------------------------------------
# Tabela final e melhor K
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " RESULTADOS FINAIS"
echo "========================================================"
printf "%-6s  %-20s  %-16s  %-14s  %s\n" "K" "Dunn Index (SkeVa)" "Cota superior" "Tempo cota (s)" "Tempo DI (s)"
printf "%-6s  %-20s  %-16s  %-14s  %s\n" "------" "--------------------" "----------------" "--------------" "------------"

BEST_K=""
BEST_DUNN="-1"
BEST_LABELS=""
N_PRUNED=0

for K in "${K_LIST[@]}"; do
  DI="${DUNN_OF[$K]}"
  printf "%-6s  %-20s  %-16s  %-14s  %s\n" "$K" "$DI" "${UB_OF[$K]}" "${UB_TIME_OF[$K]}" "${DUNN_TIME_OF[$K]}"

  if [[ "$DI" == "PRUNED" ]]; then
    N_PRUNED=$((N_PRUNED + 1))
  elif [[ "$DI" != "N/A" ]]; then
    # Compara com awk (bc-less, funciona com floats)
    IS_BETTER=$(awk -v a="$DI" -v b="$BEST_DUNN" 'BEGIN { print (a > b) ? 1 : 0 }')
    if [[ "$IS_BETTER" -eq 1 ]]; then
      BEST_DUNN="$DI"
      BEST_K="$K"
      BEST_LABELS="${LABELS_OF[$K]}"
    fi
  fi
done

echo "--------------------------------------------------------"
if [[ "$PRUNE" -eq 1 ]]; then
  echo " K descartados sem SkeVa: ${N_PRUNED} de ${#K_LIST[@]}"
fi
echo " Tempo total dunn_skeva (com leitura): ${DUNN_WALL_TOTAL}s"
if [[ -n "$BEST_K" ]]; then
  echo " Melhor K = ${BEST_K}  (Dunn Index = ${BEST_DUNN})"
  cp "$BEST_LABELS" "${SCRIPT_DIR}/Labels_best_k${BEST_K}.txt"
  echo " Labels salvo em: Labels_best_k${BEST_K}.txt"
else
  echo " Não foi possível determinar o melhor K."
fi
echo "========================================================"
