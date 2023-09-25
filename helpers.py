from math import log, log10, exp
from numpy import RankWarning
from warnings import simplefilter
simplefilter('ignore', RankWarning)


def vi_calc(x, Pri, Nc, V_rev, T, r1, r2, s1, s2, s3, t1, t2, t3, A):
    fa = list([x[0] - V_rev - (r1+r2*T)*x[1]/A - (s1+s2*T+s3*T**2)*log10((t1+t2/T+t3/T**2)*x[1]/A+1)])
    fa.append((Nc*x[0])*x[1]-Pri)
    return fa


