import torch

# N = 16
# SIZE = N*N

# h_A = torch.Tensor((SIZE))
# h_B = torch.Tensor((SIZE))

# for i in range(N):
#     for j in range(N):
#         h_A[i*N+j] = i
#         h_B[i*N+j] = j

# print("h_A", h_A)
# print("h_B", h_B)
# print("------------")

# print(h_A @ h_B)


# # Addition
# N = 10
# h_A = torch.Tensor((N))
# h_B = torch.Tensor((N))

# for i in range(N):
#     h_A[i] = i
#     h_B[i] = i * 2

# print(h_A + h_B)



# matmul
N = 8
SIZE = N * N
h_A = torch.Tensor((SIZE))
h_B = torch.Tensor((SIZE))

for i in range(SIZE):
    h_A[i] = i
    h_B[i] = i * 2

h_A = h_A.reshape((N, N))
h_B = h_B.reshape((N, N))

print(h_B.flatten())

print(h_A @ h_B)