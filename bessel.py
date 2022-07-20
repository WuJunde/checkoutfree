import math

def bessel_con(x,s = 127, tol = 1e-18):
    r = 1.0
    # print(((x * math.e) / (2*s)))
    t1 = ((x * math.e) / (2*s))**s
    t2 = 1 + 1/(12*s) + 1/(288*(s**2)) - 139/(51840 * (s**3))
    t1 = t1* math.sqrt(s/(2* math.pi)) / t2
    m = 1 / s
    k = 1
    conv = False
    while conv is not True:
        r = r * ((0.25*(x**2))/(k*(s+k)))
        m += r
        if r / m < tol:
            conv = True
        k += 1
    return t1 * m
