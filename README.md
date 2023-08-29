# attention-bottleneck
 Calculating attention bottleneck for LLM(GPT)

### Set the config of the GPU and LLM model

```json
{
    "GPU" : {
        "model" : "A100",
        "fp16" : 312,
        "fp32" : 19.5,
        "memory" : 80,
        "bandwidth" : 1935,
        "pcie" : 64
    },
    "Model" : {
        "n_layer" : 96,
        "n_head" : 96,
        "dim_head" : 128
    }
}
```

### Run the program with arguments

```bash
attn-bottleneck.py [-h] [--ngpu NGPU] [--gen GEN] [--fp16] [--seq SEQ]
```

optional arguments:

  -h, --help   show this help message and exit
  
  --ngpu NGPU  Number of gpu
  
  --gen GEN    Generation phase
  
  --fp16       Using fp16 or not
  
  --seq SEQ    Length of sequence
