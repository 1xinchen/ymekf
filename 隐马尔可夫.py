import numpy as np
T = 8
N = 3
# 状态转移矩阵A
A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
# 观测概率矩阵B
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# 初始状态概率向量pi
pi = [0.2, 0.3, 0.5]
# 初始观测序列O
O = ['红', '白', '红', '红', '白', '红', '白', '白']
# 转换O为0,1标志（数组o中，1代表白色，0代表红色）
o = np.zeros(T, int) # 初始化为int型0数组
for i in range(T):
    if O[i] == '白':
        o[i] = 1
    else:
        o[i] = 0

# 前向算法
# 计算初值alpha（alpha(i)=p(i)*b(i,o(i))）
alpha = np.zeros((T, N))
for i in range(N):
    h = o[0]
    alpha[0][i] = pi[i] * B[i][h]
# 递推，对t=1,2,3...(T-1)
# alpha_t+1(i)=[和：alpha(t,j)*A（j，i]*b(i,o(i))
for t in range(T-1):
    h = o[t+1]
    for i in range(N):
        a = 0
        for j in range(N):
            a += (alpha[t][j] * A[j][i]) # 求和
        alpha[t+1][i] = a * B[i][h]
# 终止
Pf = 0
for i in range(N):
    Pf += alpha[T-1][i]



# 后向算法
beta = np.ones((T, N)) # T时刻的状态，后向概率都是1
for t in range(T-1):
    t = T - t - 2
    h = o[t + 1]
    h = int(h)
    for i in range(N):
        beta[t][i] = 0
        for j in range(N):
            # β(t,i)=和：a(i,j)*b(j,0(i))*β(t+1,j)
            beta[t][i] += A[i][j] * B[j][h] * beta[t+1][j]
# 终止
Pb = 0
for i in range(N):
    h = o[0]
    h = int(h)
    # 求和
    Pb += pi[i] * B[i][h] * beta[0][i]

print(" 前向算法计算可得Pf:", Pf, '\n',"后向算法计算可得Pb:", Pb)
P = alpha[4-1][3-1] * beta[4-1][3-1]
print(" 前向后向概率计算可得 P(i4=q3|O,lmta)=", P / Pf)