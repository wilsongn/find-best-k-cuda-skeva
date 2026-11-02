#!/usr/bin/env bash
# Roda o pipeline sem e com --prune para os 5 datasets (K = 2..10).
# Uso: experimentos/rodar_benchmark.sh <aleatorio|kmeanspp> <dir_saida>
set -u
INIT="$1"; OUT="$(realpath -m "$2")"; mkdir -p "$OUT"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
declare -A NPTS=([d1_4cl_original]=1000000 [d2_6cl_separados]=1200000 [d3_8cl_compactos]=1000000 [d4_3cl_desiguais]=1000000 [d5_5cl_sobrepostos]=1000000)
for D in d1_4cl_original d2_6cl_separados d3_8cl_compactos d4_3cl_desiguais d5_5cl_sobrepostos; do
  if [[ "$INIT" == "kmeanspp" ]]; then
    INIT_ARGS="--init kmeanspp --init_seed 1"
  else
    INIT_ARGS="--init_centroids_dir datasets/$D/init_centroids"
  fi
  for MODE in sem com; do
    EXTRA=""; [[ $MODE == com ]] && EXTRA="--prune"
    T0=$(date +%s.%N)
    ./kmeans_dunn_eval.sh --file datasets/$D/SyntheticDataset.txt --npoints ${NPTS[$D]} --ndims 4 \
      --kmin 2 --kmax 10 $INIT_ARGS \
      --sample_pct 10 --validate_size 0 --reps 8 $EXTRA > "$OUT/${D}_${MODE}.log" 2>&1
    T1=$(date +%s.%N)
    echo "WALL $(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.2f", b-a}')" >> "$OUT/${D}_${MODE}.log"
    echo "$D $MODE done"
  done
done
rm -f Labels_best_k*.txt
