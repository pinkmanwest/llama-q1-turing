# Bonsai-27B Q1_0 on a GTX 1660 SUPER: 19 -> 56.6 tok/s in llama.cpp

Unofficial CUDA patch for llama.cpp `a518119d3` (not affiliated with the llama.cpp project). It makes decoding of the 1-bit Bonsai-27B (`Q1_0`, Qwen3.5 hybrid) about 3x faster on Turing. The weights do not change.

| llama-bench tg512 @ depth 2048, GTX 1660 SUPER, Windows 10, CUDA 12.8 | tok/s |
|---|---:|
| stock llama.cpp | 19.0 |
| patched | 56.6 |

Measured A-B-B-A in one session (`llama-bench -p 0 -n 512 -d 2048 -r 5 -ngl 99 -fa on -ctk f16 -ctv f16 -t 4`): stock 18.39 +- 0.65, patched 56.47 +- 0.67, patched 56.74 +- 0.18, stock 19.71 +- 0.05 tok/s.

Correctness vs stock: WikiText-2 perplexity 18.1129 -> 18.1040, KL divergence 0.000195, same top token in 99.0% of positions, byte-identical greedy replies from llama-server.

What changed: a new Q1_0 mat-vec kernel for Turing, a single-token attention kernel for this model, kernel fusions for the Gated DeltaNet layers, and fewer CPU-GPU syncs. The new kernels only turn on for Q1_0 on compute capability 7.5. Tested only on a GTX 1660 SUPER under Windows. Made with AI coding agents (Codex, Claude Code).

## Download (Windows)

Get `llama-q1-turing-win-cuda12.8-sm75.zip` from Releases, unzip, then:

```
llama-server.exe -m path\to\Bonsai-27B-Q1_0.gguf -ngl 99 -fa on -c 8192
```

Open http://127.0.0.1:8080. Needs a Turing GPU (GTX 16xx / RTX 20xx), NVIDIA driver 570+, and a CPU with AVX2.

## Build from source

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout a518119d3
git apply q1-turing.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release --target llama-server llama-bench
```
