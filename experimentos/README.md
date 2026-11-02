# Experimento: descarte de K por cota superior (`--prune`)

Compara o pipeline `kmeans_dunn_eval.sh` **sem** e **com** `--prune` em 5 datasets
sintéticos, com os **mesmos centróides iniciais** nas duas execuções.

## Método de descarte

Para cada K, a cota superior do Dunn é calculada em tempo linear:

```
cota = separação mínima / maior raio      (raio = max ||x − centróide||)
```

Como raio ≤ diâmetro, `Dunn ≤ cota`. Com `--prune`:

1. Calcula a cota de todos os K (`dunn_skeva --bound_only`).
2. Roda o SkeVa no K de maior cota (o mais promissor); seu Dunn vira o limiar inicial.
3. Segue em ordem decrescente de cota; um K só roda o SkeVa se sua cota superar o
   melhor Dunn obtido. Como a lista está ordenada, ao primeiro descarte todos os
   restantes também são descartados.

O diâmetro estimado de cada cluster é `max(diâmetro SkeVa, raio)`, o que garante que o
Dunn calculado nunca ultrapassa a cota.

## Datasets

Gerados com `k-means/Synthetic_Data_Generator.py` (bolas uniformes em 4D), 1M pontos
(d2: 1,2M). Os arquivos `.txt` não são versionados (~200 MB); são regenerados de forma
idêntica (ver `datasets.md5`).

| Dataset | K real | Descrição |
|---|---|---|
| d1_4cl_original | 4 | layout original do gerador, r = 9 |
| d2_6cl_separados | 6 | clusters bem separados, r = 5 |
| d3_8cl_compactos | 8 | clusters compactos, r = 4 |
| d4_3cl_desiguais | 3 | raios 4/8/12, tamanhos 100k/300k/600k |
| d5_5cl_sobrepostos | 5 | clusters sobrepostos, r = 10 |

## Parâmetros

K = 2..10, `--sample_pct 10 --validate_size 0 --reps 8`, k-means em GPU.
Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU.

## Resultados — inicialização k-means++ (`--init kmeanspp --init_seed 1`)

Tempo = cálculo do Dunn (`dunn_skeva`, incluindo leitura do dataset).

| Dataset | K real | Melhor K | Dunn | Tempo sem descarte | Tempo com descarte | Variação |
|---|---|---|---|---|---|---|
| d1: 4 clusters (original) | 4 | 4 | 1,118 | 41,2 s | 11,3 s | −73% |
| d2: 6 clusters separados | 6 | 6 | 3,550 | 60,6 s | 12,2 s | −80% |
| d3: 8 clusters compactos | 8 | 8 | 5,027 | 42,0 s | 9,4 s | −78% |
| d4: 3 clusters desiguais | 3 | 2 | 0,849 | 40,3 s | 15,6 s | −61% |
| d5: 5 clusters sobrepostos | 5 | 4 | 0,490 | 38,7 s | 45,0 s | +16% |
| **Total** | | | | **222,8 s** | **93,5 s** | **−58%** |

K descartados: 32 de 45. Logs: `resultados/init_kmeanspp/`.

## Resultados — inicialização aleatória fixa (`--init_centroids_dir`)

| Dataset | K real | Melhor K | Dunn | Tempo sem descarte | Tempo com descarte | Variação |
|---|---|---|---|---|---|---|
| d1: 4 clusters (original) | 4 | 4 | 1,118 | 43,6 s | 11,6 s | −73% |
| d2: 6 clusters separados | 6 | 4 | 0,740 | 60,6 s | 43,4 s | −28% |
| d3: 8 clusters compactos | 8 | 8 | 5,031 | 49,5 s | 9,3 s | −81% |
| d4: 3 clusters desiguais | 3 | 2 | 0,849 | 45,4 s | 18,8 s | −59% |
| d5: 5 clusters sobrepostos | 5 | 4 | 0,490 | 38,6 s | 45,0 s | +16% |
| **Total** | | | | **237,7 s** | **128,1 s** | **−46%** |

K descartados: 28 de 45. Logs: `resultados/init_aleatorio/`.

## Observações

- Em todos os casos o melhor K foi o mesmo com e sem descarte, e todo Dunn calculado
  ficou abaixo da sua cota.
- Dunn/cota ≈ 0,5 em todos os casos: em bolas uniformes o diâmetro ≈ 2 × raio, o pior
  caso da cota. Um K só é descartado se seu Dunn for menor que ~metade do melhor.
- d5 (sobrepostos): nenhum K domina, nada é descartado e sobra o custo da 1ª fase
  (releitura do dataset por K). Com rótulos verdadeiros o Dunn do K=5 é 0,351, menor
  que o do K=4 — o próprio índice prefere K=4.
- d4: com rótulos verdadeiros o Dunn do K=3 é 2,05, mas o k-means (inclusive k-means++
  com uma semente) cai em mínimo local para K=3.
- d2 com inicialização aleatória: o k-means não encontra os 6 clusters (Dunn do K=6 =
  0,067 contra 3,55 com k-means++), por isso o descarte rende menos.
- O k-means em GPU não é totalmente determinístico: mesmo com os mesmos centróides
  iniciais, d5 apresentou diferenças na 3ª casa decimal entre as execuções.

## Reproduzir

```bash
# Na raiz do repositório
git clone https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans k-means
nvcc -O3 -arch=sm_86 dunn_skeva.cu -o dunn_skeva -lm
gcc -O3 -fopenmp kmeanspp_init.c -o kmeanspp_init -lm

python3 experimentos/gerar_datasets.py
(cd datasets && md5sum -c ../experimentos/datasets.md5)
python3 experimentos/gerar_init_aleatorio.py      # só para a rodada aleatória

experimentos/rodar_benchmark.sh kmeanspp  experimentos/resultados/init_kmeanspp
experimentos/rodar_benchmark.sh aleatorio experimentos/resultados/init_aleatorio
```
