# Fast 1-bit Q1_0 models on Turing GPUs (GTX 16xx / RTX 20xx)

Unofficial CUDA patch for [llama.cpp](https://github.com/ggml-org/llama.cpp) `a518119d3`, not affiliated with the llama.cpp project. It makes `Q1_0` models generate text faster on Turing GPUs. Example: [Bonsai-27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) on a GTX 1660 SUPER goes from **19 to 56 tok/s**. Same model, same answers: perplexity 18.11 vs 18.10, greedy replies byte-identical to stock.

![llama-bench: stock vs patched](bench.png)

## Which models

Only the **`Q1_0`** format (binary: every weight is +1 or -1, 1.125 bits per weight). Not `TQ1_0` / `TQ2_0` (ternary) and not `IQ1_S` / `IQ1_M`. Check: `llama-server` prints `file type   = Q1_0` while loading.

- Any Q1_0 model gets the fast matrix kernel (2.7x faster per matrix).
- Qwen3.5-based models like Bonsai-27B (hybrid: 48 Gated DeltaNet + 16 attention layers, head size 256) also get the attention and DeltaNet speedups, so about 3x overall. Only Bonsai-27B was tested end to end.

## Which GPUs

Only **Turing (sm_75, compute capability 7.5)**: GTX 1660 / 1660 SUPER / 1660 Ti, RTX 2060-2080 Ti, Quadro RTX, T4. Tested on a GTX 1660 SUPER. Bonsai-27B (3.53 GB) needs 6 GB of video memory.

Does not work on GTX 10xx (Pascal). On RTX 30/40/50 it runs at stock speed: use official llama.cpp there.

## Quick start (Windows)

Needs NVIDIA driver 570+ and a CPU with AVX2. CUDA is included in the zip.

1. Download `llama-q1-turing-win-cuda12.8-sm75.zip` from [Releases](https://github.com/pinkmanwest/llama-q1-turing/releases) and unzip it.
2. Download a Q1_0 model, for example `Bonsai-27B-Q1_0.gguf` from [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf).
3. In the unzipped folder run (PowerShell):
   ```
   .\llama-server.exe -m C:\models\Bonsai-27B-Q1_0.gguf -ngl 99 -c 4096 -np 1 -fa on -ctk f16 -ctv f16 -t 4 -b 512 -ub 256 --jinja --alias bonsai --host 127.0.0.1 --port 8080
   ```
   Keep the KV cache at `f16`: the fast attention kernel does not work with `q8_0`.
4. Open http://127.0.0.1:8080.

| Problem | Fix |
|---|---|
| `VCRUNTIME140.dll` / `VCOMP140.DLL` not found | install [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `no kernel image is available` | your GPU is not Turing |
| out of memory | use a smaller `-c`, close other GPU apps |
| ~19 tok/s instead of ~55 | check `-ngl 99 -fa on` and `f16` KV cache |

## Build from source

Needs Git, CMake, Visual Studio 2022 and CUDA Toolkit 12.x.

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout a518119d3
curl -L -o q1-turing.patch https://raw.githubusercontent.com/pinkmanwest/llama-q1-turing/main/q1-turing.patch
git apply q1-turing.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release --target llama-server llama-bench
```

All code changes are in [q1-turing.patch](q1-turing.patch): a Q1_0 matrix kernel for Turing, a single-token attention kernel, Gated DeltaNet fusions and fewer CPU-GPU syncs. Made with AI coding agents (Codex, Claude Code). MIT license, same as llama.cpp.
