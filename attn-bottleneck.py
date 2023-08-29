import argparse
import json

parser = argparse.ArgumentParser(description='Compute bottleneck of attention layer')
parser.add_argument('--ngpu', default=1, help='Number of gpu')
parser.add_argument('--gen', default=True, help='Generation phase')
parser.add_argument('--fp16', action="store_true")
parser.add_argument('--seq', default=1024, help='Length of sequence')

args = parser.parse_args()
# print(args)
ngpu = int(args.ngpu)
gen = args.gen
seq_len = args.seq
fp16 = args.fp16

with open('config.json') as json_file:
    data = json.load(json_file)
    # print(data)
    n_layer = data["Model"]["n_layer"]
    n_head = data["Model"]["n_head"]
    dim_head = data["Model"]["dim_head"]
    
    print("--------------------------------------")
    print("Working with %d %s GPUs" % (ngpu, data["GPU"]["model"]))
    print("--------------------------------------")

    if gen:
        # print("In Generation Phase")
        # print()
        print("QKV size per head")
        print("Q: %d x %d" % (1, dim_head))
        print("K (including cache): %d x %d" % (seq_len, dim_head))
        print("V (including cache): %d x %d" % (seq_len, dim_head))
        print("Cached KV size (FP32): %d GB" % (2 * n_layer * n_head * dim_head * (seq_len - 1) * 4 / 10**9))
        print("--------------------------------------")
        print("In Attention Layer (per %d head)" % n_head)
        print("--------------------------------------")
        print("S = Q x K^T")
        print("%d x ((%d x %d) x (%d x %d))" % (n_head, 1, dim_head, dim_head, seq_len))
        print("--------------------------------------")

        flops = (n_head * seq_len * ((2 * dim_head) - 1))
        print("Total FLOPS: %d FLOPS" % flops)
        if not fp16:
            print("Computation time (FP32): %ss" % format(flops/(data["GPU"]["fp32"]*10**12)/ngpu, '10.3e'))
            print()
            memory = (n_head * dim_head * (seq_len - 1) * 4)
            print("Read Key Matrix (FP32): %d x (%d x %d) x %d = %d bytes" % (n_head, dim_head, (seq_len - 1), 4, memory))
            print("Memory time: %ss" % format(memory/(data["GPU"]["bandwidth"]*10**9)/ngpu, '10.3e'))
            print()
            latency = flops/(data["GPU"]["fp32"])

            if ngpu > 1: # larger than 1, need synchronizing
                sync_mem = n_head * seq_len * 4
                print("Synchronizing S (FP32): %d x (%d x %d) x %d = %d bytes" % (n_head, 1, seq_len, 4, sync_mem))
                print("Sync time: %ss" % format(sync_mem / (data["GPU"]["pcie"]*10**9), '10.3e'))
                sync = latency + sync_mem / (data["GPU"]["pcie"]/10**3)
                print()
                print("Bandwidth Requirement (with sync): %.3f TB/s" % (memory/sync))

        else:
            print("Computation time (FP16): %ss" % format(flops/(data["GPU"]["fp16"]*10**12)/ngpu, '10.3e'))
            print()
            memory = (n_head * dim_head * (seq_len - 1) * 2)
            print("Read Key Matrix (FP16): %d x (%d x %d) x %d = %d bytes" % (n_head, dim_head, (seq_len - 1), 2, memory))
            print("Memory time: %ss" % format(memory/(data["GPU"]["bandwidth"]*10**9)/ngpu, '10.3e'))
            print()
            latency = flops/(data["GPU"]["fp16"])
            
            if ngpu > 1: # larger than 1, need synchronizing
                sync_mem = n_head * seq_len * 2
                print("Synchronizing S (FP16): %d x (%d x %d) x %d = %d bytes" % (n_head, 1, seq_len, 2, sync_mem))
                print("Sync time: %ss" % format(sync_mem / (data["GPU"]["pcie"]*10**9), '10.3e'))
                sync = latency + sync_mem / (data["GPU"]["pcie"]/10**3)
                print()
                print("Bandwidth Requirement (with sync): %.3f TB/s" % (memory/sync))

        print("Bandwidth Requirement (without sync): %.3f TB/s" % (memory/latency))
        print("--------------------------------------")

        print("Attn = S x V")
        print("%d x ((%d x %d) x (%d x %d))" % (n_head, 1, seq_len, seq_len, dim_head))
        print("--------------------------------------")
        flops = (n_head * dim_head * ((2 * seq_len) - 1))
        print("Total FLOPS: %d FLOPS" % flops)

        if not fp16:
            print("Computation time (FP32): %ss" % format(flops/(data["GPU"]["fp32"]*10**12)/ngpu, '10.3e'))
            print()
            memory = (n_head * dim_head * (seq_len - 1) * 4)
            print("Read Key Matrix (FP32): %d x (%d x %d) x %d = %d bytes" % (n_head, dim_head, (seq_len - 1), 4, memory))
            print("Memory time: %ss" % format(memory/(data["GPU"]["bandwidth"]*10**9)/ngpu, '10.3e'))
            print()
            latency = flops/(data["GPU"]["fp32"])

            if ngpu > 1: # larger than 1, need synchronizing
                sync_mem = n_head * dim_head * 4
                print("Synchronizing S (FP32): %d x (%d x %d) x %d = %d bytes" % (n_head, 1, dim_head, 4, sync_mem))
                print("Sync time: %ss" % format(sync_mem / (data["GPU"]["pcie"]*10**9), '10.3e'))
                sync = latency + sync_mem / (data["GPU"]["pcie"]/10**3)
                print()
                print("Bandwidth Requirement (with sync): %.3f TB/s" % (memory/sync))

        else:
            print("Computation time (FP16): %ss" % format(flops/(data["GPU"]["fp16"]*10**12)/ngpu, '10.3e'))
            print()
            memory = (n_head * dim_head * (seq_len - 1) * 2)
            print("Read Key Matrix (FP16): %d x (%d x %d) x %d = %d bytes" % (n_head, dim_head, (seq_len - 1), 2, memory))
            print("Memory time: %ss" % format(memory/(data["GPU"]["bandwidth"]*10**9), '10.3e'))
            print()
            latency = flops/(data["GPU"]["fp16"])

            if ngpu > 1: # larger than 1, need synchronizing
                sync_mem = n_head * dim_head * 2
                print("Synchronizing S (FP16): %d x (%d x %d) x %d = %d bytes" % (n_head, 1, dim_head, 2, sync_mem))
                print("Sync time: %ss" % format(sync_mem / (data["GPU"]["pcie"]*10**9)/ngpu, '10.3e'))
                sync = latency + sync_mem / (data["GPU"]["pcie"]/10**3)
                print()
                print("Bandwidth Requirement (with sync): %.3f TB/s" % (memory/sync))

        print("Bandwidth Requirement (without sync): %.3f TB/s" % (memory/latency))