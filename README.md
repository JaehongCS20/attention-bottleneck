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
  
  --ngpu NGPU  Number of gpu (INT, default 1)
  
  --gen GEN    Generation phase (BOOL, default True)
  
  --fp16       Using fp16 or not (default fp32)
  
  --seq SEQ    Length of sequence (INT, default 1024)

### Output
```bash
python3 attn-bottleneck.py --ngpu 10
```
```txt
--------------------------------------
Working with 10 A100 GPUs
--------------------------------------
QKV size per head
Q: 1 x 128
K (including cache): 1024 x 128
V (including cache): 1024 x 128
Cached KV size (FP32): 9 GB
--------------------------------------
In Attention Layer (per 96 head)
--------------------------------------
S = Q x K^T
96 x ((1 x 128) x (128 x 1024))
--------------------------------------
Total FLOPS: 25067520 FLOPS
Computation time (FP32):  1.286e-07s

Read Key Matrix (FP32): 96 x (128 x 1023) x 4 = 50282496 bytes
Memory time:  2.599e-06s

Synchronizing S (FP32): 96 x (1 x 1024) x 4 = 393216 bytes
Sync time:  6.144e-06s

Bandwidth Requirement (with sync): 6.768 TB/s
Bandwidth Requirement (without sync): 39.115 TB/s
--------------------------------------
Attn = S x V
96 x ((1 x 1024) x (1024 x 128))
--------------------------------------
Total FLOPS: 25153536 FLOPS
Computation time (FP32):  1.290e-07s

Read Key Matrix (FP32): 96 x (128 x 1023) x 4 = 50282496 bytes
Memory time:  2.599e-06s

Synchronizing Attn (FP32): 96 x (1 x 128) x 4 = 49152 bytes
Sync time:  7.680e-07s

Bandwidth Requirement (with sync): 24.434 TB/s
Bandwidth Requirement (without sync): 38.981 TB/s
```
