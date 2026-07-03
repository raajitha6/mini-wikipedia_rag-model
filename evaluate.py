from rag import ds, retrieve

def evaluate_hit_rate(k=3, n_samples=100):
    hits = 0
    sample = ds.select(range(min(n_samples, len(ds))))

    for i, row in enumerate(sample):
        results = retrieve(row['question'], top_n=k)
        retrieved_indices = [idx for idx, chunk, similarity in results]
        if i in retrieved_indices:
            hits += 1

    hit_rate = hits / len(sample)
    print(f"\nHit Rate@{k}: {hit_rate:.2%} ({hits}/{len(sample)})")
    return hit_rate

if __name__ == "__main__":
    evaluate_hit_rate(k=3, n_samples=100)