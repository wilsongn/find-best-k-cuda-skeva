# Gera os 5 datasets sintéticos (4D) usando k-means/Synthetic_Data_Generator.py
# com as variáveis do topo substituídas e random.seed(2026) (saída determinística).
# Saída: datasets/<nome>/{gen.py,SyntheticDataset.txt,Labels.txt}
import os, re, subprocess, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = open(os.path.join(root, 'k-means/Synthetic_Data_Generator.py')).read()
D = {
 'd1_4cl_original': dict(r=[9]*4, d1=[40,40,60,60], d2=[40,60,40,60], d3=[60,60,40,40], d4=[60,40,60,40], n=[250000]*4),
 'd2_6cl_separados': dict(r=[5]*6, d1=[20,50,80,20,50,80], d2=[20,20,20,60,60,60], d3=[30,70,30,70,30,70], d4=[30,30,70,70,50,50], n=[200000]*6),
 'd3_8cl_compactos': dict(r=[4]*8, d1=[20,60,20,60,20,60,20,60], d2=[20,20,60,60,20,20,60,60], d3=[20,20,20,20,60,60,60,60], d4=[20,60,60,20,60,20,20,60], n=[125000]*8),
 'd4_3cl_desiguais': dict(r=[4,8,12], d1=[20,60,30], d2=[20,60,80], d3=[20,20,80], d4=[20,60,30], n=[100000,300000,600000]),
 'd5_5cl_sobrepostos': dict(r=[10]*5, d1=[40,55,40,55,47], d2=[40,40,55,55,47], d3=[50]*5, d4=[50]*5, n=[200000]*5),
}
for name, p in D.items():
    out = os.path.join(root, 'datasets', name); os.makedirs(out, exist_ok=True)
    s = src
    for var, key in [('cluster_r','r'),('cluster_dim1','d1'),('cluster_dim2','d2'),('cluster_dim3','d3'),('cluster_dim4','d4'),('nb_points','n')]:
        s, c = re.subn(r'^%s = \[.*\]$' % var, '%s = %s' % (var, p[key]), s, flags=re.M); assert c == 1
    s = s.replace('import random\n', 'import random\nrandom.seed(2026)\n', 1)
    open(os.path.join(out, 'gen.py'), 'w').write(s)
    subprocess.run([sys.executable, 'gen.py'], cwd=out, check=True)
    print(name, sum(p['n']), flush=True)
