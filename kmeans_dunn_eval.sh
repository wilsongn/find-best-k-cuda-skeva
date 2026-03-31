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
#
# Caminhos (detectados automaticamente se estiverem no mesmo diretório do script):
#   --kmeans_dir DIR        diretório dos fontes do k-means [<script_dir>/k-means]
#   --dunn_bin BIN          binário dunn [<script_dir>/dunn_skeva]

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

SKETCH_SIZE=512
VALIDATE_SIZE=512
REPS=8
DUNN_THREADS=256
SEED=42
INTER_MODE="centroids"
SAMPLE_PCT=0
USE_STREAMS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KMEANS_DIR="${SCRIPT_DIR}/k-means"
DUNN_BIN="${SCRIPT_DIR}/dunn_skeva"

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
    --sketch_size)   SKETCH_SIZE="$2";     shift 2 ;;
    --validate_size) VALIDATE_SIZE="$2";   shift 2 ;;
    --reps)          REPS="$2";            shift 2 ;;
    --dunn_threads)  DUNN_THREADS="$2";    shift 2 ;;
    --seed)          SEED="$2";            shift 2 ;;
    --inter_mode)    INTER_MODE="$2";      shift 2 ;;
    --sample_pct)    SAMPLE_PCT="$2";      shift 2 ;;
    --no-streams)    USE_STREAMS=0;        shift   ;;
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

# ---------------------------------------------------------------------------
# Arrays de resultado
# ---------------------------------------------------------------------------
declare -a RESULT_K=()
declare -a RESULT_DUNN=()
declare -a RESULT_DUNN_TIME=()
declare -a RESULT_LABELS=()

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
echo "========================================================"
echo " kmeans_dunn_eval  —  K de $KMIN a $KMAX"
echo " Dataset : $FILE"
echo " N=$N  NF=$NF  alvo_kmeans=$KMEANS_TARGET"
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

  # ---- 5) Executar Dunn Index ----
  echo "[dunn] Calculando Dunn Index para K=$K..."
  DUNN_ARGS="--data_file ${FILE} --labels_file ${LABELS_FILE} --nf ${NF}"
  DUNN_ARGS="$DUNN_ARGS --sketch_size ${SKETCH_SIZE} --validate_size ${VALIDATE_SIZE}"
  DUNN_ARGS="$DUNN_ARGS --reps ${REPS} --threads ${DUNN_THREADS} --seed ${SEED}"
  DUNN_ARGS="$DUNN_ARGS --inter_mode ${INTER_MODE}"
  if [[ "$SAMPLE_PCT" != "0" ]]; then
    DUNN_ARGS="$DUNN_ARGS --sample_pct ${SAMPLE_PCT}"
  fi
  if [[ "$USE_STREAMS" -eq 0 ]]; then
    DUNN_ARGS="$DUNN_ARGS --no-streams"
  fi

  # Captura stdout (resultados) e stderr (timings) separadamente
  # O || true impede que set -e mate o script caso o dunn retorne exit code != 0
  DUNN_STDERR_FILE="${K_WORKDIR}/dunn_stderr.txt"
  DUNN_OUTPUT=$("$DUNN_BIN" $DUNN_ARGS 2>"$DUNN_STDERR_FILE") || true

  DUNN_INDEX=$(echo "$DUNN_OUTPUT" | grep "Dunn Index (SkeVa)" | awk '{print $NF}') || true
  DUNN_TIME=$(grep "TOTAL" "$DUNN_STDERR_FILE" | awk '{print $(NF-1)}') || true

  if [[ -z "$DUNN_INDEX" ]]; then
    echo "Aviso: Dunn falhou para K=$K. Erro:" >&2
    cat "$DUNN_STDERR_FILE" >&2
    DUNN_INDEX="N/A"
  fi
  if [[ -z "$DUNN_TIME" ]]; then
    DUNN_TIME="N/A"
  fi

  echo "[resultado] K=$K  Dunn Index = $DUNN_INDEX  Tempo DI = ${DUNN_TIME}s"

  RESULT_K+=("$K")
  RESULT_DUNN+=("$DUNN_INDEX")
  RESULT_DUNN_TIME+=("$DUNN_TIME")
  RESULT_LABELS+=("$LABELS_FILE")
done

# ---------------------------------------------------------------------------
# Tabela final e melhor K
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " RESULTADOS FINAIS"
echo "========================================================"
printf "%-6s  %-20s  %s\n" "K" "Dunn Index (SkeVa)" "Tempo DI (s)"
printf "%-6s  %-20s  %s\n" "------" "--------------------" "------------"

BEST_K=""
BEST_DUNN="-1"
BEST_LABELS=""

for i in "${!RESULT_K[@]}"; do
  K="${RESULT_K[$i]}"
  DI="${RESULT_DUNN[$i]}"
  DT="${RESULT_DUNN_TIME[$i]}"
  printf "%-6s  %-20s  %s\n" "$K" "$DI" "$DT"

  if [[ "$DI" != "N/A" ]]; then
    # Compara com awk (bc-less, funciona com floats)
    IS_BETTER=$(awk -v a="$DI" -v b="$BEST_DUNN" 'BEGIN { print (a > b) ? 1 : 0 }')
    if [[ "$IS_BETTER" -eq 1 ]]; then
      BEST_DUNN="$DI"
      BEST_K="$K"
      BEST_LABELS="${RESULT_LABELS[$i]}"
    fi
  fi
done

echo "--------------------------------------------------------"
if [[ -n "$BEST_K" ]]; then
  echo " Melhor K = ${BEST_K}  (Dunn Index = ${BEST_DUNN})"
  cp "$BEST_LABELS" "${SCRIPT_DIR}/Labels_best_k${BEST_K}.txt"
  echo " Labels salvo em: Labels_best_k${BEST_K}.txt"
else
  echo " Não foi possível determinar o melhor K."
fi
echo "========================================================"
