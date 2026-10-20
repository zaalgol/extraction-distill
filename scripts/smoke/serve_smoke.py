"""Smoke test: vLLM loads a small model on the local GPU and generates."""
import time

from vllm import LLM, SamplingParams


def main() -> None:
    t0 = time.time()
    llm = LLM("Qwen/Qwen3-0.6B", max_model_len=2048, gpu_memory_utilization=0.5)
    out = llm.generate(["Name three colors:"] * 8, SamplingParams(max_tokens=32, temperature=0))
    print(repr(out[0].outputs[0].text))
    print(f"{time.time() - t0:.0f}s")
    print("SERVE SMOKE OK")


# vLLM starts its engine core with multiprocessing "spawn", which re-imports this module.
if __name__ == "__main__":
    main()
