import socket
import os
import urllib.request

# Force IPv4 because Windows IPv6 route resets connections
old_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4

def fetch_hf_repo(repo_id: str, files: list[str], cache_dir: str):
    folder_name = "models--" + repo_id.replace("/", "--")
    snap_dir = os.path.join(cache_dir, folder_name, "snapshots", "main")
    refs_dir = os.path.join(cache_dir, folder_name, "refs")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w", encoding="utf-8") as f:
        f.write("main")

    for rel_path in files:
        target_path = os.path.join(snap_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"Already cached: {rel_path}")
            continue

        url = f"https://huggingface.co/{repo_id}/resolve/main/{rel_path}"
        print(f"Downloading {repo_id}/{rel_path} -> {target_path}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Wget/1.20"})
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_f:
            copied = 0
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                out_f.write(buf)
                copied += len(buf)
                if copied % (10 * 1024 * 1024) < len(buf):
                    print(f"  {copied / (1024*1024):.1f} MB", flush=True)
        print(f"Downloaded {rel_path} ({os.path.getsize(target_path)} bytes)")

if __name__ == "__main__":
    cache_dir = os.path.abspath(r"C:\Users\nikla\Documents\Programming\ws26\SoftwareArchitecture\SemanticProjectArchitecture\earlybird-arch\.fastembed_cache")
    print(f"Populating cache: {cache_dir}")
    
    # 1. BGE-Small-EN-v1.5
    fetch_hf_repo(
        "Xenova/bge-small-en-v1.5",
        ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "onnx/model.onnx"],
        cache_dir
    )

    # 2. Qwen3-Embedding-0.6B
    fetch_hf_repo(
        "Qwen/Qwen3-Embedding-0.6B",
        ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
        cache_dir
    )
