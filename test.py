import numpy as np

t = [0,0.05,0.1,0.15,1.,1.05]
for i in t:
    print(int(i))
    plt.xlabel('t')
    plt.ylabel('x2(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + t0, t0 * 5))
    plt.plot(t, list_y2, color='purple')
    plt.show()
    plt.xlabel('t')
    plt.ylabel('x3(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + t0, t0 * 5))
    plt.plot(t, y3_listed, color='purple')
    plt.show()




def suma_for_l(f, f_min, g, k0):
    p_j = f
    g_j = []
    for j in range(0, k0):
        p_j = f_min.dot(p_j)
        g_j.append(p_j.dot(g))
    summ = np.zeros((3, 3))
    for i in g_j:
        summ += i.dot(i.transpose())
    return summ, g_j