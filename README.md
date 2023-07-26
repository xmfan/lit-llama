```
Current benchmark results
7B, 6 prompt/200 tokens, bf16
eager:
Time for inference 2: 9.29 sec total, 21.53 tokens/sec
Bandwidth achieved: 290.09 GB/s

torch.compile per block
Time for inference 2: 5.03 sec total, 39.74 tokens/sec
Bandwidth achieved: 535.52 GB/s

Note: Still seems primarily overhead-bound (also, cudagraphs is segfaulting for me right now...)

- torch.compile on whole model generates gibberish right now... My suspicion is that it's because we're not guarding on attributes of modules. Yes indeed, it's now fixed)

- cudagraphs now doesn't work because I needed to add in-place mutation to fix whole graph compilation issues

torch.compile on whole model
Time for inference 3: 2.66 sec total, 75.13 tokens/sec
Bandwidth achieved: 1012.55 GB/s
```
