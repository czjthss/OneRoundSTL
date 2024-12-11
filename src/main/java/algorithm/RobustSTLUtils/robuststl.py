import sys
import numpy as np
from RobustSTL.RobustSTL import RobustSTL


def robuststl(y, T, reg1=1., reg2=0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.):
    _, trend, seasonal, resid = RobustSTL(y, T, reg1=reg1, reg2=reg2, K=K, H=H, dn1=dn1, dn2=dn2, ds1=ds1, ds2=ds2)
    return trend, seasonal, resid

if __name__ == '__main__':
    T = 12
    if len(sys.argv) >= 2:
        T = int(sys.argv[1])  # First argument
    y = []
    f = open("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/input.txt","r")
    for line in f.readlines():
        if line.strip() != "":
            y.append(float(line.strip()))
    y = np.array(y)
    f.close()
    trend, seasonal, residual = robuststl(y, T)

    f = open("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/trend.txt","w")
    f.write("\n".join([str(item) for item in trend]))
    f.close()

    f = open("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/seasonal.txt","w")
    f.write("\n".join([str(item) for item in seasonal]))
    f.close()

    f = open("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/residual.txt","w")
    f.write("\n".join([str(item) for item in residual]))
    f.close()
