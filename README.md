# find-best-k-cuda-skeva

> **K-means com Seleção Automática de K via Dunn-SkeVa em GPU**
> Wilson G. N. Junior, Wellington S. Martins — Instituto de Informática, UFG

---

> [!IMPORTANT]
> **Nota ao avaliador do artigo**
>
> Por equívoco, enviamos o artigo no formato *blind* (sem identificação dos autores).
> A versão completa e correta do artigo — com autores, filiação e demais informações —
> está disponível na pasta [`artigo/wperformance.pdf`](artigo/wperformance.pdf)
> deste repositório.

---

## Visão Geral

Este repositório implementa uma pipeline para **seleção automática do melhor K no K-means** usando o **Índice de Dunn acelerado por GPU via amostragem SkeVa** (*Sketch-and-Validate*).

Para cada K em um intervalo `[Kmin, Kmax]`, a pipeline:
1. Executa o K-means (CPU ou GPU) para obter os rótulos dos clusters.
2. Calcula o Índice de Dunn via `dunn_skeva` (CUDA).
3. Seleciona o K com maior Índice de Dunn.

---

## Dependências

- CUDA Toolkit (nvcc)
- GPU com suporte a CUDA (Compute Capability ≥ 7.5 recomendado)
- `nvidia-smi` disponível no PATH
- Python 3 (para geração de datasets sintéticos)
- `make`, `g++`, `gcc`

---

## Estrutura do Repositório

```
find-best-k-cuda-skeva/
├── dunn_skeva.cu              # Implementação CUDA do Índice de Dunn com SkeVa
├── kmeans_dunn_eval.sh        # Script principal da pipeline
├── Synthetic_Data_Generator.py# Gerador de datasets sintéticos
├── k-means/                   # Fontes do K-means (ver seção abaixo)
└── artigo/
    └── wperformance.pdf       # Artigo completo (versão com autores)
```

---

## 1. Obter o código do K-means

O script depende do K-means de He, G., Vialle, S., & Baboulin, M. (2021) — *Parallelization of the k-means algorithm in a spectral clustering chain on CPU-GPU platforms* (Euro-Par 2020 Workshops, LNCS 12480, pp. 135–147). Clone-o diretamente na pasta `k-means`:

```bash
git clone https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans k-means
```

Após o clone, a pasta `k-means/` deve conter os arquivos `main.h`, `Makefile`, os fontes `.cc` e `.cu` do K-means.

> **Importante:** O script `kmeans_dunn_eval.sh` espera os fontes exatamente em `k-means/`
> (relativo ao diretório do script). Não renomeie a pasta.

---

## 2. Compilar o `dunn_skeva`

```bash
# Detecte a compute capability da sua GPU (ex.: 75 para RTX série 20, 86 para RTX série 30)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compile (substitua sm_75 pela sua GPU)
nvcc -O3 -arch=sm_75 dunn_skeva.cu -o dunn_skeva -lm
```

---

## 3. Gerar datasets sintéticos com `Synthetic_Data_Generator.py`

O script Python gera datasets com clusters gaussianos para testes. Exemplo de uso:

```bash
# Gera um dataset com 50000 pontos, 8 dimensões e 5 clusters reais
python3 Synthetic_Data_Generator.py \
    --npoints 50000 \
    --ndims 8 \
    --nclusters 5 \
    --output dataset_50k_8d_5k.txt
```

O arquivo de saída é um texto com `N` linhas, cada linha contendo `NF` valores separados por espaço (formato esperado pelo K-means e pelo `dunn_skeva`).

---

## 4. Executar a pipeline completa

### Uso básico

```bash
./kmeans_dunn_eval.sh \
    --file <dataset> \
    --npoints N \
    --ndims NF \
    --kmin Kmin \
    --kmax Kmax
```

### Exemplo completo

```bash
./kmeans_dunn_eval.sh \
    --file dataset_50k_8d_5k.txt \
    --npoints 50000 \
    --ndims 8 \
    --kmin 2 \
    --kmax 10 \
    --kmeans_target GPU \
    --sketch_size 512 \
    --validate_size 512 \
    --reps 8 \
    --inter_mode centroids
```

Ao final, o script imprime uma tabela com o Índice de Dunn para cada K e destaca o melhor K. O arquivo de rótulos do melhor K é salvo como `Labels_best_k<K>.txt`.

### Todas as opções

| Opção | Padrão | Descrição |
|---|---|---|
| `--file PATH` | — | **Obrigatório.** Caminho do dataset |
| `--npoints N` | — | **Obrigatório.** Número de pontos |
| `--ndims NF` | — | **Obrigatório.** Número de dimensões |
| `--kmin K` | — | **Obrigatório.** K mínimo |
| `--kmax K` | — | **Obrigatório.** K máximo |
| `--npackages P` | 100 | NbPackages do K-means |
| `--kmeans_target GPU\|CPU` | GPU | Alvo de execução do K-means |
| `--kmeans_threads T` | 1 | Threads OpenMP (somente modo CPU) |
| `--kmeans_iters I` | 200 | Máximo de iterações do K-means |
| `--kmeans_tol TOL` | 1e-4 | Tolerância de convergência |
| `--sketch_size S` | 512 | Tamanho fixo do sketch SkeVa |
| `--validate_size V` | 512 | Tamanho do validate SkeVa |
| `--reps R` | 8 | Repetições SkeVa |
| `--dunn_threads T` | 256 | Threads por bloco CUDA |
| `--seed N` | 42 | Semente do gerador aleatório |
| `--inter_mode centroids\|points` | centroids | Modo de separação inter-cluster |
| `--sample_pct P` | 0 | sketch\_size = P% do cluster (0 = fixo) |
| `--no-streams` | — | Desabilitar CUDA streams |
| `--kmeans_dir DIR` | `./k-means` | Diretório dos fontes do K-means |
| `--dunn_bin BIN` | `./dunn_skeva` | Caminho do binário dunn\_skeva |

---

## 5. Usar o `dunn_skeva` diretamente (sem pipeline)

**Modo padrão** (dataset + K explícito):
```bash
./dunn_skeva --file dataset.txt --nf 8 --k 5 \
    --sketch_size 512 --validate_size 512 --reps 8
```

**Modo K-means** (dataset + arquivo de rótulos gerado pelo K-means):
```bash
./dunn_skeva --data_file dataset.txt --labels_file Labels.txt --nf 8 \
    --sketch_size 512 --validate_size 512 --reps 8 \
    --inter_mode centroids
```

---

## Referência

He, G., Vialle, S., & Baboulin, M. (2021). Parallelization of the k-means algorithm in a spectral clustering chain on CPU-GPU platforms. In *Euro-Par 2020: Parallel Processing Workshops* (Vol. 12480, LNCS, pp. 135–147). Springer.
Código fonte: https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans
