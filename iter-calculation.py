flops = 0
memory = 0

#################################
#            CONFIG             #
#       USING 1 A100 GPU        #
#################################

emb = 12288
head = 96
dim = 128
ffn = emb * 4

batch = 512
seq = 512 # input length
out = 32 # output length

# initialization phase
print("==========INIT PAHSE==========")
print()
# QKV
flops_qkv = batch * seq * emb * (2*emb - 1) * 3
memory_qkv = batch * ((seq * emb) * 4 + (emb * emb) * 4 * 3)
print("QKV")
print("Computation time (FP32): %ss" % format(flops_qkv/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_qkv/(2**30)/1935, '10.3e'))
print()

# Q * K
flops_qk = batch * seq * seq * (2*dim - 1) * head
memory_qk = batch * ((seq * dim) * 4 * head + (dim * seq) * 4 * head)
print("QK")
print("Computation time (FP32): %ss" % format(flops_qk/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_qk/(2**30)/1935, '10.3e'))
print()

# S * V
flops_sv =  batch * seq * dim * (2*seq - 1) * head
memory_sv = batch * ((seq * seq) * 4 * head + (seq * dim) * 4 * head)
print("SV")
print("Computation time (FP32): %ss" % format(flops_sv/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_sv/(2**30)/1935, '10.3e'))
print()

# add norm
flops_add = batch * (seq * emb * (2*emb - 1))
memory_add = batch * ((seq * emb) * 4 + (emb * emb) * 4)
print("Add Norm")
print("Computation time (FP32): %ss" % format(flops_add/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_add/(2**30)/1935, '10.3e'))
print()

# ffn
flops_ffn = batch * (seq * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1))
memory_ffn = batch * ((seq * emb) * 4 + (emb * ffn) * 4 + (seq * ffn) * 4 + (ffn * emb) * 4)

print("FFN")
print("Computation time (FP32): %ss" % format(flops_ffn/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_ffn/(2**30)/1935, '10.3e'))
print()

flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

ct = flops/(19.5*10**12)
print("Init Computation time (FP32): %ss" % format(ct, '10.3e'))
memory /= 2**30
mt = memory/1935
print("Init Memory time: %ss" % format(mt, '10.3e'))

# print("Total Computation time (FP32): %ss" % format(ct * 96, '10.3e'))
# print("Total Memory time: %ss" % format(mt * 96, '10.3e'))

print("Init Memory qkv + qk + sv : %f GB" % ((memory_qkv + memory_qk + memory_sv)/(2**30)))
print("Init Memory ffn : %f GB" % ((memory_ffn)/(2**30)))

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
    memory_qkv += batch * ((1 * emb) * 4 + (emb * emb) * 4 * 3)

    # Q * K
    flops_qk += batch * (1 * seq * (2*dim - 1) * head)
    memory_qk += batch * ((1 * dim) * 4 * head + (dim * seq) * 4 * head )

    # S * V
    flops_sv += batch * (1 * dim * (2*seq - 1) * head)
    memory_sv += batch * ((1 * seq) * 4 * head + (seq * dim) * 4 * head)

    # add norm
    flops_add += batch * (1 * emb * (2*emb - 1))
    memory_add += batch * ((1 * emb) * 4 + (emb * emb) * 4)

    # ffn
    flops_ffn += batch * (1 * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1))
    memory_ffn += batch * ((1 * emb) * 4 + (emb * ffn) * 4 + (1 * ffn) * 4 + (ffn * emb) * 4)

print("QKV")
print("Computation time (FP32): %ss" % format(flops_qkv/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_qkv/(2**30)/1935, '10.3e'))
print()
print("QK")
print("Computation time (FP32): %ss" % format(flops_qk/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_qk/(2**30)/1935, '10.3e'))
print()
print("SV")
print("Computation time (FP32): %ss" % format(flops_sv/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_sv/(2**30)/1935, '10.3e'))
print()
print("Add Norm")
print("Computation time (FP32): %ss" % format(flops_add/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_add/(2**30)/1935, '10.3e'))
print()
print("FFN")
print("Computation time (FP32): %ss" % format(flops_ffn/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_ffn/(2**30)/1935, '10.3e'))
print()

flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

ct = flops/(19.5*10**12)
print("Gen Computation time (FP32): %ss" % format(ct, '10.3e'))
memory /= 2**30
mt = memory/1935
print("Gen Memory time: %ss" % format(mt, '10.3e'))

# print("Total Computation time (FP32): %ss" % format(ct * 96, '10.3e'))
# print("Total Memory time: %ss" % format(mt * 96, '10.3e'))

print("Gen Memory qkv + qk + sv : %f GB" % ((memory_qkv + memory_qk + memory_sv)/(2**30)))
print("Gen Memory ffn : %f GB" % ((memory_ffn)/(2**30)))