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
├── k-means/                   # Fontes do K-means (ver seção abaixo)
│   ├── Synthetic_Data_Generator.py  # Gerador de datasets sintéticos
│   ├── main.h, main.cc, gpu.cu, ...
│   └── Makefile
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

O script gera um dataset com clusters circulares em **4 dimensões**. Ele não aceita argumentos de linha de comando — edite as variáveis no topo do arquivo antes de executar:

```python
# Raio de cada cluster
cluster_r = [9, 9, 9, 9]

# Centros dos clusters por dimensão (um valor por cluster)
cluster_dim1 = [40, 40, 60, 60]
cluster_dim2 = [40, 60, 40, 60]
cluster_dim3 = [60, 60, 40, 40]
cluster_dim4 = [60, 40, 60, 40]

# Número de pontos por cluster (len define K)
nb_points = [12500000, 12500000, 12500000, 12500000]
```

Para alterar o dataset, ajuste essas variáveis diretamente:
- **K**: adicione/remova entradas em `cluster_r`, `cluster_dim*` e `nb_points` (todos devem ter o mesmo comprimento)
- **N**: altere os valores em `nb_points`
- **Separação entre clusters**: ajuste as coordenadas em `cluster_dim*`
- **Compacidade dos clusters**: ajuste `cluster_r`

```bash
# Execute a partir da raiz do repositório
python3 k-means/Synthetic_Data_Generator.py
```

A saída é gerada no diretório de trabalho atual (raiz do repositório):
- `SyntheticDataset.txt` — coordenadas (4 colunas separadas por tab, formato float32)
- `Labels.txt` — rótulo de cluster de cada ponto (um por linha)

O arquivo `SyntheticDataset.txt` é compatível com o K-means e o `dunn_skeva`. O `Labels.txt` pode ser usado diretamente com o `dunn_skeva` no modo `--labels_file`.

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
    --file SyntheticDataset.txt \
    --npoints 50000000 \
    --ndims 4 \
    --kmin 2 \
    --kmax 8 \
    --kmeans_target GPU \
    --reps 8 \
    --sample_pct 30 \
    --validate_size 0
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
./dunn_skeva --file SyntheticDataset.txt \
    --nf 4 --k 4 \
    --reps 8 --sample_pct 30 --validate_size 0
```

**Modo K-means** (dataset + arquivo de rótulos gerado pelo K-means):
```bash
./dunn_skeva --data_file SyntheticDataset.txt \
    --labels_file Labels.txt --nf 4 \
    --reps 8 --sample_pct 30 --validate_size 0 \
    --inter_mode centroids
```

---

## Referência

He, G., Vialle, S., & Baboulin, M. (2021). Parallelization of the k-means algorithm in a spectral clustering chain on CPU-GPU platforms. In *Euro-Par 2020: Parallel Processing Workshops* (Vol. 12480, LNCS, pp. 135–147). Springer.
Código fonte: https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans
