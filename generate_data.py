import numpy as np
import os

def generate_graph(v_num, avg_degree=10):
    edges = []
    for i in range(v_num):
        edges.append((i, i))
    
    e_num = v_num * avg_degree
    for _ in range(e_num):
        src = np.random.randint(0, v_num)
        dst = np.random.randint(0, v_num)
        edges.append((src, dst))
        edges.append((dst, src))
    
    edges = sorted(list(set(edges)))
    return v_num, len(edges), edges

def save_graph(v_num, e_num, edges, filename):
    with open(filename, 'w') as f:
        f.write(f"{v_num} {e_num}\n")
        for src, dst in edges:
            f.write(f"{src} {dst}\n")

def save_embedding(v_num, f0, filename):
    embedding = np.random.randn(v_num, f0).astype(np.float32)
    with open(filename, 'wb') as f:
        embedding.tofile(f)

def save_weight(in_dim, out_dim, filename):
    weight = np.random.randn(in_dim, out_dim).astype(np.float32)
    with open(filename, 'wb') as f:
        weight.tofile(f)

if __name__ == "__main__":
    np.random.seed(42)
    
    graph_dir = "graph"
    embedding_dir = "embedding"
    weight_dir = "weight"
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    
    sizes = [10000, 100000]
    f0 = 64
    f1 = 16
    f2 = 8
    
    for size in sizes:
        print(f"Generating dataset for {size} vertices...")
        v_num, e_num, edges = generate_graph(size, avg_degree=20)
        print(f"  Generated: {v_num} vertices, {e_num} edges")
        
        graph_file = f"{graph_dir}/{size}_graph.txt"
        save_graph(v_num, e_num, edges, graph_file)
        print(f"  Saved: {graph_file}")
        
        embedding_file = f"{embedding_dir}/{size}.bin"
        save_embedding(size, f0, embedding_file)
        print(f"  Saved: {embedding_file}")
    
    save_weight(f0, f1, f"{weight_dir}/W_{f0}_{f1}.bin")
    save_weight(f1, f2, f"{weight_dir}/W_{f1}_{f2}.bin")
    print(f"Saved weights: W_{f0}_{f1}.bin, W_{f1}_{f2}.bin")
