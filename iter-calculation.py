import csv

#################################
#          MODEL CONFIG         #
#################################
emb = 12288
head = 96
dim = 128
ffn = emb * 4

#################################
#           GPU CONFIG          #
#################################
T_flops = 19.5
GB_bandwidth = 1935
n_gpu = 20

T_flops *= n_gpu
GB_bandwidth *= n_gpu

def calculate_print(batch, seq, out):

    flops = 0
    memory = 0

    # initialization phase
    print("==========INIT PAHSE==========")
    print()
    # QKV
    flops_qkv = batch * seq * emb * (2*emb - 1) * 3
    memory_qkv = batch * (seq * emb) * 4 + (emb * emb) * 4 * 3
    print("QKV")
    print("Computation time (FP32): %.3fms" % (flops_qkv/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_qkv/(2**30)/GB_bandwidth*(10**3)))
    print()

    # Q * K
    flops_qk = batch * seq * seq * (2*dim - 1) * head
    memory_qk = batch * ((seq * dim) * 4 * head + (dim * seq) * 4 * head)
    print("QK")
    print("Computation time (FP32): %.3fms" % (flops_qk/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_qk/(2**30)/GB_bandwidth*(10**3)))
    print()

    # S * V
    flops_sv =  batch * seq * dim * (2*seq - 1) * head
    memory_sv = batch * ((seq * seq) * 4 * head + (seq * dim) * 4 * head)
    print("SV")
    print("Computation time (FP32): %.3fms" % (flops_sv/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_sv/(2**30)/GB_bandwidth*(10**3)))
    print()

    # add norm
    flops_add = batch * (seq * emb * (2*emb - 1))
    memory_add = batch * (seq * emb) * 4 + (emb * emb) * 4
    print("Add Norm")
    print("Computation time (FP32): %.3fms" % (flops_add/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_add/(2**30)/GB_bandwidth*(10**3)))
    print()

    # ffn
    flops_ffn = batch * (seq * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1))
    memory_ffn = batch * (seq * emb) * 4 + (emb * ffn) * 4 + batch * (seq * ffn) * 4 + (ffn * emb) * 4

    print("FFN")
    print("Computation time (FP32): %.3fms" % (flops_ffn/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_ffn/(2**30)/GB_bandwidth*(10**3)))
    print()

    flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
    memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

    ct = flops/(T_flops*10**9)
    print("Init Computation time (FP32): %.3fms" % ct)
    memory /= 2**30
    mt = memory/GB_bandwidth*(10**3)
    print("Init Memory time: %.3fms" % mt)

    # print("Total Computation time (FP32): %ss" % format(ct * 96, '10.3e'))
    # print("Total Memory time: %ss" % format(mt * 96, '10.3e'))

    print("Init Memory qkv + qk + sv : %.2f GB" % ((memory_qkv + memory_qk + memory_sv)/(2**30)))
    print("Init Memory ffn : %.2f GB" % ((memory_ffn)/(2**30)))

    print()
    print("==========GEN PAHSE==========")
    print()
    flops_qkv = 0
    flops_qk = 0
    flops_sv = 0
    flops_add = 0
    flops_ffn = 0
    memory_qkv = 0
    memory_qk = 0
    memory_sv = 0
    memory_add = 0
    memory_ffn = 0
    for i in range(1, out):
        seq = 512 + i
        # QKV
        flops_qkv += batch * (1 * emb * (2*emb - 1) * 3)
        memory_qkv += batch * (1 * emb) * 4 + (emb * emb) * 4 * 3

        # Q * K
        flops_qk += batch * (1 * seq * (2*dim - 1) * head)
        memory_qk += batch * ((1 * dim) * 4 * head + (dim * seq) * 4 * head )

        # S * V
        flops_sv += batch * (1 * dim * (2*seq - 1) * head)
        memory_sv += batch * ((1 * seq) * 4 * head + (seq * dim) * 4 * head)

        # add norm
        flops_add += batch * (1 * emb * (2*emb - 1))
        memory_add += batch * (1 * emb) * 4 + (emb * emb) * 4

        # ffn
        flops_ffn += batch * (1 * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1))
        memory_ffn += batch * (1 * emb) * 4 + (emb * ffn) * 4 + batch * (1 * ffn) * 4 + (ffn * emb) * 4

    print("QKV")
    print("Computation time (FP32): %.3fms" % (flops_qkv/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_qkv/(2**30)/GB_bandwidth*(10**3)))
    print()
    print("QK")
    print("Computation time (FP32): %.3fms" % (flops_qk/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_qk/(2**30)/GB_bandwidth*(10**3)))
    print()
    print("SV")
    print("Computation time (FP32): %.3fms" % (flops_sv/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_sv/(2**30)/GB_bandwidth*(10**3)))
    print()
    print("Add Norm")
    print("Computation time (FP32): %.3fms" % (flops_add/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_add/(2**30)/GB_bandwidth*(10**3)))
    print()
    print("FFN")
    print("Computation time (FP32): %.3fms" % (flops_ffn/(T_flops*10**9)))
    print("Memory time: %.3fms" % (memory_ffn/(2**30)/GB_bandwidth*(10**3)))
    print()

    flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
    memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

    ct = flops/(T_flops*10**9)
    print("Gen Computation time (FP32): %.3fms" % ct)
    memory /= 2**30
    mt = memory/GB_bandwidth*(10**3)
    print("Gen Memory time: %.3fms" % mt)

    # print("Total Computation time (FP32): %ss" % format(ct * 96, '10.3e'))
    # print("Total Memory time: %ss" % format(mt * 96, '10.3e'))

    print("Gen Memory qkv + qk + sv : %.2f GB" % ((memory_qkv + memory_qk + memory_sv)/(2**30)))
    print("Gen Memory ffn : %.2f GB" % ((memory_ffn)/(2**30)))

def calculate_file(batch, seq, out, file):
    with open(file, 'a', newline='') as f:
        wr = csv.writer(f)
        row = []
        flops = 0
        memory = 0

        # initialization phase
        # QKV
        flops_qkv = batch * seq * emb * (2*emb - 1) * 3
        memory_qkv = batch * (seq * emb) * 4 + (emb * emb) * 4 * 3
        row.append(flops_qkv/(T_flops*10**9))
        row.append(memory_qkv/(2**30)/GB_bandwidth*(10**3))

        # Q * K
        flops_qk = batch * seq * seq * (2*dim - 1) * head
        memory_qk = batch * ((seq * dim) * 4 * head + (dim * seq) * 4 * head)
        row.append(flops_qk/(T_flops*10**9))
        row.append(memory_qk/(2**30)/GB_bandwidth*(10**3))

        # S * V
        flops_sv =  batch * seq * dim * (2*seq - 1) * head
        memory_sv = batch * ((seq * seq) * 4 * head + (seq * dim) * 4 * head)
        row.append(flops_sv/(T_flops*10**9))
        row.append(memory_sv/(2**30)/GB_bandwidth*(10**3))

        # add norm
        flops_add = batch * (seq * emb * (2*emb - 1))
        memory_add = batch * (seq * emb) * 4 + (emb * emb) * 4
        row.append(flops_add/(T_flops*10**9))
        row.append(memory_add/(2**30)/GB_bandwidth*(10**3))

        # ffn
        flops_ffn = batch * (seq * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1))
        memory_ffn = batch * (seq * emb) * 4 + (emb * ffn) * 4 + batch * (seq * ffn) * 4 + (ffn * emb) * 4
        row.append(flops_ffn/(T_flops*10**9))
        row.append(memory_ffn/(2**30)/GB_bandwidth*(10**3))

        flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
        memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

        ct = flops/(T_flops*10**9)
        row.append(ct)
        memory /= 2**30
        mt = memory/GB_bandwidth*(10**3)
        row.append(mt)

        row.append((memory_qkv + memory_qk + memory_sv)/(2**30))
        row.append(((memory_ffn)/(2**30)))

        flops_qkv = 0
        flops_qk = 0
        flops_sv = 0
        flops_add = 0
        flops_ffn = 0
        memory_qkv = 0
        memory_qk = 0
        memory_sv = 0
        memory_add = 0
        memory_ffn = 0

        for i in range(1, out):
            seq_ = seq + i
            # QKV
            flops_qkv += batch * (1 * emb * (2*emb - 1) * 3)
            memory_qkv += batch * (1 * emb) * 4 + (emb * emb) * 4 * 3

            # Q * K
            flops_qk += batch * (1 * seq_ * (2*dim - 1) * head)
            memory_qk += batch * ((1 * dim) * 4 * head + (dim * seq_) * 4 * head )

            # S * V
            flops_sv += batch * (1 * dim * (2*seq_ - 1) * head)
            memory_sv += batch * ((1 * seq_) * 4 * head + (seq_ * dim) * 4 * head)

            # add norm
            flops_add += batch * (1 * emb * (2*emb - 1))
            memory_add += batch * (1 * emb) * 4 + (emb * emb) * 4

            # ffn
            flops_ffn += batch * (1 * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1))
            memory_ffn += batch * (1 * emb) * 4 + (emb * ffn) * 4 + batch * (1 * ffn) * 4 + (ffn * emb) * 4

        row.append(flops_qkv/(T_flops*10**9))
        row.append(memory_qkv/(2**30)/GB_bandwidth*(10**3))
        row.append(flops_qk/(T_flops*10**9))
        row.append(memory_qk/(2**30)/GB_bandwidth*(10**3))
        row.append(flops_sv/(T_flops*10**9))
        row.append(memory_sv/(2**30)/GB_bandwidth*(10**3))
        row.append(flops_add/(T_flops*10**9))
        row.append(memory_add/(2**30)/GB_bandwidth*(10**3))
        row.append(flops_ffn/(T_flops*10**9))
        row.append(memory_ffn/(2**30)/GB_bandwidth*(10**3))

        flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
        memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

        ct = flops/(T_flops*10**9)
        row.append(ct)
        memory /= 2**30
        mt = memory/GB_bandwidth*(10**3)
        row.append(mt)

        row.append((memory_qkv + memory_qk + memory_sv)/(2**30))
        row.append(((memory_ffn)/(2**30)))

        type = [batch, seq, out]
        type.extend(row)
        wr.writerow(type)


# run experiment
# init csv file
with open('result.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['batch', 'seq', 'out', 
                 'init_qkv_c', 'init_qkv_m', 'init_qk_c', 'init_qk_m', 'init_sv_c', 'init_sv_m', 'init_add_c', 'init_add_m', 'init_ffn_c', 'init_ffn_m', 'init_ct', 'init_mt', 'init_qk_mem', 'init_ffn_mem',
                 'gen_qkv_c', 'gen_qkv_m', 'gen_qk_c', 'gen_qk_m', 'gen_sv_c', 'gen_sv_m', 'gen_add_c', 'gen_add_m', 'gen_ffn_c', 'gen_ffn_m', 'gen_ct', 'gen_mt', 'gen_qk_mem', 'gen_ffn_mem'
                 ])


#################################
#           EXP CONFIG          #
#################################
batches = [16, 32, 64, 128, 256, 384, 512]
seqs = range(64, 2049, 64) # input length
outs = range(64, 2049, 64) # output length
# input + output > 2048 is automatically deprecated

for batch in batches:
    for seq in seqs:
        for out in outs:
            # out of range impossible config
            if out + seq > 2048:
                continue
            # calculate_print(batch, seq, out)
            calculate_file(batch, seq, out, 'result.csv')
            
# make graph
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('result.csv', index_col=False)
df1 = df.loc[df.gen_qk_m >= df.gen_qkv_c, ['batch', 'seq', 'out', 'gen_qkv_c', 'gen_qkv_m', 'gen_qk_c', 'gen_qk_m', 'gen_sv_c', 'gen_sv_m', 'gen_add_c', 'gen_add_m', 'gen_ffn_c', 'gen_ffn_m']]
print(df1)
df1.to_csv('mem_bound.csv', index=False)
