import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg

def create_A(a1, a2):
    a = np.zeros((3, 3))
    a[0][1] = 1
    a[1][2] = 1
    a[2][0] = -1
    a[2][1] = -a1
    a[2][2] = -a2
    return a


def create_b(b):
    b_i = np.zeros((3, 1))
    b_i[0][0] = 0
    b_i[1][0] = 0
    b_i[2][0] = b
    return b_i


def create_c():
    c = []
    for i in range(3):
        value = np.zeros((1, 3))
        value[0][i] = 1
        c.append(value)
    return c


def calculate_f(a, t0, q):
    k = 0
    tmp_a = a.dot(t0)
    res = (1 / math.factorial(k)) * linalg.matrix_power(tmp_a, k)
    while k != q:
        k += 1
        res += (1 / math.factorial(k)) * linalg.matrix_power(tmp_a, k)
    return res



def calculate_g(f,a, b):
    e = np.eye(3)
    #tmp = e*t0
    #for j in range(2, q+1):
    #    tmp += (np.linalg.matrix_power(a, j-1).dot(e)*(t0**j))/np.math.factorial(j)

    return (f - e).dot(np.linalg.inv(a)).dot(b)


def calculate_f_ob(a, t0):
    f = np.eye(3)
    f += np.linalg.matrix_power(a.dot(t0),2)/2
    f = f - a.dot(t0) - np.linalg.matrix_power(a.dot(t0),3)/6
    return f


def calc_step_f_k(f, k0):
    p0 = np.eye(3)
    for j in range(1, k0):
        p0 = f.dot(p0)
    return p0


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


def interface():
    a1 = int(input('Введіть а1 від 1 до 10 '))
    a2 = int(input('Введіть а2 від 1 до 10 '))
    b = int(input('Введіть b від 1 до 10 '))
    a = create_A(a1, a2)
    b_a = create_b(b)
    k0 = int(input('Кількість проміжків k0 від 20 до 40 '))
    t0 = float(input("Період квантування t0 [ 0.001, 0.05] "))
    x_aim_1 = int(input("В який стан потрібно перейти? (х1 компонента) "))
    q = int(input("Введіть точність q "))
    x_aim = np.array([[x_aim_1], [0], [0]])

    x_start = np.array([[m], [0], [0]])

    f = calculate_f(a, t0, q)
    g = calculate_g(f,a,b_a)
    #f_min = calculate_f_ob(a, t0)
    f_min = linalg.inv(f)
    #f_step_k0 = calc_step_f_k(f, k0)
    #p = linalg.matrix_power(f, k0-1)
    p, s, g_j = suma_for_l(f, f_min, g, k0)
    l = p.dot(s)
    print('sum ',s)
    l_ob = linalg.inv(l)
    l0 = l_ob.dot(x_aim)
    print(l0)
    #l0 = np.array([[6.34965004e+13], [1.14293533e+12], [6.09501373e+09]])
    u_k = calc_u(g_j, l0, k0)
    c = create_c()
    t = [t0 * i for i in range(k0)]
    x_k = [x_start]

    for k in range(len(t)):
        x_k.append(calculate_equation_1(f, x_k[k], g, u_k[k]))
    del x_k[0]
    y1 = []
    y2 = []
    y3 = []
    for i in x_k:
        y1 = calculate_equation_2(i, c[0], y1)
        y2 = calculate_equation_2(i, c[1], y2)
        y3 = calculate_equation_2(i, c[2], y3)

    lol1 = []
    for i in y1:
        a = i.tolist()
        lol1.append(a[0])
    list_y1 = []
    for m in range(len(lol1)):
        list_y1.append(lol1[m][0])

    lol2 = []
    for w in y2:
        a = w.tolist()
        lol2.append(a[0])
    list_y2 = []
    for s in range(len(lol2)):
        list_y2.append(lol2[s][0])

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
    plt.xticks(np.arange(t[0],t[-1]+1,0.1))
    plt.plot(t, list_y1, color='purple')
    plt.show()
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.grid()
    plt.xticks(np.arange(t[0],t[-1]+1,0.2))
    plt.plot(t, uk_listed, color='purple')
    plt.show()

    for j in range(k0):
        print(j, x_k[j][0], x_k[j][1], x_k[j][2], uk_listed[j], sep='   |   ')

interface()