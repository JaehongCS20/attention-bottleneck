flops = 0
memory = 0

#################################
#            CONFIG             #
#       USING 1 A100 GPU        #
#            batch 1            #
#################################

emb = 12288
head = 96
dim = 128
seq = 2048
ffn = emb * 4

# QKV
flops_qkv = 1 * emb * (2*emb - 1) * 3
memory_qkv = (1 * emb) * 4 + (emb * emb) * 4 * 3
print("QKV")
print("Computation time (FP32): %ss" % format(flops_qkv/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_qkv/(2**30)/1935, '10.3e'))
print()

# Q * K
flops_qk = 1 * seq * (2*dim - 1) * head
memory_qk = (1 * dim) * 4 * head + (dim * seq) * 4 * head 
print("QK")
print("Computation time (FP32): %ss" % format(flops_qk/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_qk/(2**30)/1935, '10.3e'))
print()

# S * V
flops_sv = 1 * dim * (2*seq - 1) * head
memory_sv = (1 * seq) * 4 * head + (seq * dim) * 4 * head
print("SV")
print("Computation time (FP32): %ss" % format(flops_sv/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_sv/(2**30)/1935, '10.3e'))
print()

# add norm
flops_add= 1 * emb * (2*emb - 1)
memory_add = (1 * emb) * 4 + (emb * emb) * 4
print("Add Norm")
print("Computation time (FP32): %ss" % format(flops_add/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_add/(2**30)/1935, '10.3e'))
print()

# ffn
flops_ffn = 1 * ffn * (2*emb - 1) + 1 * emb * (2*ffn - 1)
memory_ffn = (1 * emb) * 4 + (emb * ffn) * 4 + (1 * ffn) * 4 + (ffn * emb) * 4

print("FFN")
print("Computation time (FP32): %ss" % format(flops_ffn/(19.5*10**12), '10.3e'))
print("Memory time: %ss" % format(memory_ffn/(2**30)/1935, '10.3e'))
print()

flops += flops_qkv + flops_qk + flops_sv + flops_add + flops_ffn
memory += memory_qkv + memory_qk + memory_sv + memory_add + memory_ffn

ct = flops/(19.5*10**12)
print("Computation time (FP32): %ss" % format(ct, '10.3e'))
memory /= 2**30
mt = memory/1935
print("Memory time: %ss" % format(mt, '10.3e'))

print("Total Computation time (FP32): %ss" % format(ct * 96, '10.3e'))
print("Total Memory time: %ss" % format(mt * 96, '10.3e'))


print("Memory qkv + qk + sv : %f GB" % ((memory_qkv + memory_qk + memory_sv)/(2**30)))
print("Memory ffn : %f GB" % ((memory_ffn)/(2**30)))

print("Memory d_model x d_ffn x 2 size: %f GB" % (((ffn * emb) * 4 * 2)/(2**30)))