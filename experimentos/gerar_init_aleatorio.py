# Gera centróides iniciais aleatórios fixos (K pontos sorteados do dataset)
# para K = 2..10 em datasets/<nome>/init_centroids/init_k<K>.txt.
# Usados na rodada com inicialização aleatória (--init_centroids_dir).
import random, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ds = os.path.join(root, 'datasets')
for name in sorted(os.listdir(ds)):
    d = os.path.join(ds, name)
    lines = open(os.path.join(d, 'SyntheticDataset.txt')).read().splitlines()
    out = os.path.join(d, 'init_centroids'); os.makedirs(out, exist_ok=True)
    for K in range(2, 11):
        rng = random.Random(1000 + K)
        idx = rng.sample(range(len(lines)), K)
        open(os.path.join(out, f'init_k{K}.txt'), 'w').write('\n'.join(lines[i].replace('\t', ' ') for i in idx) + '\n')
    print(name, 'ok')
