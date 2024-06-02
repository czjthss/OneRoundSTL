package algorithm;

import algorithm.utils.LDLT;

public class TSDBSTL_Query {
    private final double[][][] TSDB;
    private final int period;
    private final int MAX_PAGE_SIZE;
    private final double[] v;
    private final LDLT ldlt;
    // results
    private double[] trend;
    private double[] seasonal;
    private double[] residual;

    public TSDBSTL_Query(double[][][] TSDB, int period, double[] v, LDLT ldlt) {
        this.TSDB = TSDB;
        this.period = period;
        this.v = v;
        this.ldlt = ldlt;

        // constant
        this.MAX_PAGE_SIZE = TSDB[0][0].length;
    }

    public void decompose(int begin, int end, double zeta) {
        headRecalculation(begin, end, zeta);
        tailRecalculation(end);

        int size = end - begin;
        double[] y = new double[2 * size];
        // backward substitution
        for (int i = 2 * size - 1; i > -1; i--) {
            double tmp = 0.0;
            for (int k = Math.min(i + 4, 2 * size - 1); k > i; k--) {
                if (k > 2 * size - 5)
                    tmp += ldlt.queryEndL(k, i, size) * y[k];
                else
                    tmp += ldlt.queryL(k, i) * y[k];
            }
            if (i > 2 * size - 5)
                y[i] = queryZ(i + 2 * begin) / ldlt.queryEndD(i, i, size) - tmp;
            else
                y[i] = queryZ(i + 2 * begin) / ldlt.queryD(i, i) - tmp;
        }

        // divide into seasonal trend residual
        trend = new double[size];
        seasonal = new double[size];
        residual = new double[size];
        for (int i = 0; i < size; ++i) {
            trend[i] = y[2 * i];
            seasonal[i] = y[2 * i + 1];
            residual[i] = queryB(2 * i) - trend[i] - seasonal[i];
        }
    }

    private void headRecalculation(int begin, int end, double zeta) {
        double zPre = Double.MAX_VALUE;

        for (int i = begin * 2; i < end * 2; i++) {
            double tmp = 0.0;
            for (int k = Math.max(i - 4, begin * 2); k < i; k++) {
                tmp += ldlt.queryL(i, k) * queryZ(k);
            }
            double zNew = queryB(i) - tmp;
//            System.out.println(Math.abs(queryZ(i) - zNew) + Math.abs(queryZ(i - 1) - zPre));
            if (i > begin * 2 && Math.abs(queryZ(i) - zNew) + Math.abs(queryZ(i - 1) - zPre) < zeta) {
                break;
            }
            zPre = zNew; // This was missing in your Python code, but seems necessary for logic.
            setZ(i, zNew);
        }
    }

    private void tailRecalculation(int end) {
        for (int i = 2 * end - 4; i < 2 * end; i++) {
            double tmp = 0.0;
            for (int k = i - 4; k < i; k++) {
                tmp += ldlt.queryEndL(i, k, end) * queryZ(k);
            }
            setZ(i, queryB(i) - tmp);
        }
    }

    private double queryB(int idx) {
        double val = TSDB[idx / (2 * MAX_PAGE_SIZE)][0][idx % (2 * MAX_PAGE_SIZE) / 2];
        return idx % 2 == 0 ? val : val + v[(idx / 2) % period];
    }

    private double queryZ(int idx) {
        return TSDB[idx / (2 * MAX_PAGE_SIZE)][idx % 2 + 1][idx % (2 * MAX_PAGE_SIZE) / 2];
    }

    private void setZ(int idx, double val) {
        if (idx % 2 == 0) {
            TSDB[idx / (2 * MAX_PAGE_SIZE)][1][idx % (2 * MAX_PAGE_SIZE) / 2] = val;
        } else {
            TSDB[idx / (2 * MAX_PAGE_SIZE)][2][idx % (2 * MAX_PAGE_SIZE) / 2] = val;
        }
    }

    public double[] getTrend() {
        return trend;
    }

    public double[] getSeasonal() {
        return seasonal;
    }

    public double[] getResidual() {
        return residual;
    }
}
