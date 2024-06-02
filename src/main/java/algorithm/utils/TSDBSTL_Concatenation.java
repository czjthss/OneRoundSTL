package algorithm.utils;

public class TSDBSTL_Concatenation {
    private final double[][][] TSDB;
    private final int period;
    private final int MAX_PAGE_SIZE, MAX_PAGE_NUM;
    private final double[] v;
    private final LDLT ldlt;

    public TSDBSTL_Concatenation(double[][][] TSDB, int period, double[] v, LDLT ldlt) {
        this.TSDB = TSDB;
        this.period = period;
        this.v = v;
        this.ldlt = ldlt;

        // constant
        this.MAX_PAGE_NUM = TSDB.length;
        this.MAX_PAGE_SIZE = TSDB[0][0].length;
    }

    public void concat(double zeta) {
        concat(0, MAX_PAGE_NUM * MAX_PAGE_SIZE, zeta);
    }

    public void concat(int begin, int end, double zeta) {
        int pageBegin = begin / MAX_PAGE_SIZE, pageEnd = end / MAX_PAGE_SIZE;
        for (int pageIdx = pageBegin + 1; pageIdx <= pageEnd; pageIdx++) {
            if (TSDB[pageIdx][0] == null) break; // skip null
            convergeConcatenation(pageIdx, zeta);
        }
    }

    private void convergeConcatenation(int pageIdx, double zeta) {
        double zPre = Double.MAX_VALUE;
        int begin = 2 * pageIdx * MAX_PAGE_SIZE, end = 2 * (pageIdx + 1) * MAX_PAGE_SIZE;

        // converge concatenation
        for (int i = begin; i < end; ++i) {
            double tmp = 0.0;
            for (int k = i - 4; k < i; k++) {
                tmp += ldlt.queryL(i, k) * queryZ(k);
            }
            double zNew = queryB(i) - tmp;
            if (Math.abs(queryZ(i) - zNew) + Math.abs(queryZ(i - 1) - zPre) < zeta) {
                break;
            }
            zPre = queryZ(i); // Assign z[zi] to zPre for the next iteration
            setZ(i, zNew);// Update the value in the z array
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
}
