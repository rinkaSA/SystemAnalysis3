import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg

a1 = int(input('Введіть а1 від 1 до 10 '))
a2 = int(input('Введіть а2 від 1 до 10 '))
a = np.array([[0, 1, 0],
              [0, 0, 1],
              [-1, -a1, -a2]])
b_i = int(input('Введіть b від 1 до 10 '))
b = np.array([[0],
              [0],
              [b_i]])

c = []
for i in range(3):
    value = np.zeros((1, 3))
    value[0][i] = 1
    c.append(value)

k0 = int(input('Кількість проміжків k0 від 20 до 40 '))
t0 = float(input("Період квантування t0 [ 0.001, 0.05] "))
q = int(input("Введіть точність q "))

def f():
    tmp_a = a.dot(t0)
    i = 0
    res = (1 / math.factorial(i)) * linalg.matrix_power(tmp_a, i)
    while i != q:
        i += 1
        res += (1 / math.factorial(i)) * linalg.matrix_power(tmp_a, i)

    return res


def suma_for_l(f, f_min, g, k0):
    c = 0
    p = np.eye(3)
    while c <= k0-1:
        p = f.dot(p)
        c += 1
    sum_of_g_j = 0
    c = 0
    R = f
    list_of_g_j = []
    while c < k0:
        G_j = R.dot(g)
        list_of_g_j.append(G_j)
        sum_of_g_j += (G_j.dot(G_j.transpose()))
        R = f_min.dot(R)
        c += 1
    return p, sum_of_g_j, list_of_g_j

def calc_u(g_j, l0, k0):
    u_k = []
    for k in range(0, k0):
        g_t = g_j[k].transpose()
        u_k.append(g_t.dot(l0))
    return u_k

def calculate_equation_1(f, x, g, u_k):
    res = f.dot(x) + g.dot(u_k)
    return res


def calculate_equation_2(x, c, y):
    y.append(c.dot(x))
    return y

x_aim_1 = int(input("В який стан потрібно перейти? (х1 компонента) "))
x_aim = np.array([[x_aim_1], [0], [0]])
x_s = int(input('З якого стану починаємо? (x1 компонента) '))
x_start = np.array([[x_s], [0], [0]])

x_aim[0] -= x_start[0]
e = np.eye(3)
f = f()

g = (f - e).dot(linalg.inv(a)).dot(b)

f_min = linalg.inv(f)
p, s, g_j = suma_for_l(f, f_min, g, k0)
l = p.dot(s)
l_ob = linalg.inv(l)
l0 = l_ob.dot(x_aim)
u_k = calc_u(g_j, l0, k0)
x_k = [x_start]
t = [t0 * i for i in range(k0)]
for k in range(len(t)):
    x_k.append(calculate_equation_1(f, x_k[k], g, u_k[k]))
del x_k[0]
y1 = []
for i in x_k:
    y1 = calculate_equation_2(i, c[0], y1)

lol1 = []
for i in y1:
    a = i.tolist()
    lol1.append(a[0])
list_y1 = []
for m in range(len(lol1)):
    list_y1.append(lol1[m][0])

lol3 = []
for n in u_k:
    s = n.tolist()
    lol3.append(s[0])
uk_listed = []
for p in range(len(lol3)):
    uk_listed.append(lol3[p][0])
plt.xlabel('t')
plt.ylabel('x1(t)')
plt.grid()

plt.xticks(np.arange(t[0], t[-1] + 1, 0.1))
plt.plot(t, list_y1, color='purple')
plt.show()
plt.xlabel('t')
plt.ylabel('u(t)')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 0.1))
plt.plot(t, uk_listed, color='pink')
plt.show()


print('k    |     x(1)        |       x(2)       |       x(3)        |        u_k       |')
for j in range(k0):
    print(j, x_k[j][0], x_k[j][1], x_k[j][2], uk_listed[j], sep='   |   ')
