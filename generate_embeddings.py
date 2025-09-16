from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm, trange
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(model, ds, batch_size=32):
    all_embeddings = []
    for i in trange(0, len(ds), batch_size):
        batch_texts = ds[i:i+batch_size]['text']
        embeddings = model.encode(batch_texts, convert_to_tensor=True, device=device)
        all_embeddings.append(embeddings)

    embeddings_tensor = torch.cat(all_embeddings)
    return embeddings_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save sentence embeddings.")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save the generated embeddings.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generating embeddings.")
    args = parser.parse_args()

    output_dir = args.output_dir
    batch_size = args.batch_size

    ds = load_dataset("jxm/nq_corpus_dpr", split="train").shuffle(seed=42).select(range(60_000))
    sent_models = {}
    sent_embeds = {}

    sent_keys = {
        'e5': 'intfloat/e5-base-v2',
        'stella': 'infgrad/stella-base-en-v2',
        'granite': 'ibm-granite/granite-embedding-278m-multilingual',
        'gtr': 'sentence-transformers/gtr-t5-base'
        # 'granite-small': 'ibm-granite/granite-embedding-30m-english',
        # 'e5-small': 'intfloat/e5-small',
        # 'all-mini': 'all-MiniLM-L6-v2'
    }
    for key in sent_keys:
        if key not in sent_models:
            sent_models[key] = SentenceTransformer(sent_keys[key])
            sent_embeds[key] = get_embeddings(sent_models[key], ds, batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)

    for key, tensor in sent_embeds.items():
        torch.save(tensor, f"{output_dir}/{key}.pt")