package algorithm;

import algorithm.utils.LDLT;
import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;

public class OneRoundSTL {
    private double[] v;
    private double[] coldStart(double[] ts, int period, int cs_period_num) {
        int cs_size = cs_period_num * period;
        double[] v = new double[period];
        double[] cs_array = new double[cs_size];

        // generate cold start array
        System.arraycopy(ts, 0, cs_array, 0, cs_size);

        // stl decomposition
        SeasonalTrendLoess.Builder stlBuilder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess stl = stlBuilder.setPeriodic().setPeriodLength(period).buildSmoother(cs_array);
        SeasonalTrendLoess.Decomposition stlDecomposition = stl.decompose();
        double[] seasonal_array = stlDecomposition.getSeasonal();

        // generate v
        double mean;
        for (int i = 0; i < period; ++i) {
            mean = 0.0;
            for (int j = 0; j < cs_period_num; ++j) {
                mean += seasonal_array[i + j * period];
            }
            mean /= cs_period_num;
            v[i] = mean;
        }
        return v;
    }

    private double[] generateB(double[] ts, double[] v, int period) {
        double[] b = new double[ts.length * 2];
        for (int i = 0; i < ts.length; i++) {
            b[i * 2] = ts[i];
            b[i * 2 + 1] = ts[i] + v[i % period];
        }
        return b;
    }

    private void headRecalculation(double[] ts, double[] z, double[] b, LDLT ldlt, double tau) {
        double zPre = Double.MAX_VALUE;

        for (int i = 0; i < ts.length * 2; i++) {
            double tmp = 0.0;
            for (int k = Math.max(i - 4, 0); k < i; k++) {
                tmp += ldlt.queryL(i, k) * z[k];
            }
            double zNew = b[i] - tmp;
            if (i > 0 && Math.abs(z[i] - zNew) + Math.abs(z[i - 1] - zPre) < tau) {
                break;
            }
            zPre = zNew; // This was missing in your Python code, but seems necessary for logic.
            z[i] = zNew;
        }
    }

    private void tailRecalculation(double[] ts, double[] z, double[] b, LDLT ldlt) {
        for (int i = 2 * ts.length - 4; i < 2 * ts.length; i++) {
            double tmp = 0.0;
            for (int k = i - 4; k < i; k++) {
                tmp += ldlt.queryEndL(i, k, ts.length) * z[k];
            }
            z[i] = b[i] - tmp;
        }
    }

    public double[][] decompose(double[] ts, int period, int initPeriodNum, double epsilon, double tau, double lambda) throws Exception {
        int MAX_TS_SIZE = 1000000;
        LDLT ldlt = new LDLT(MAX_TS_SIZE, epsilon, lambda, 1.0);

        v = coldStart(ts, period, initPeriodNum);
        double[] b = generateB(ts, v, period);

        // output
        double[][] result = new double[3][ts.length];

        // forward substitution
        double[] z = new double[ts.length * 2];
        for (int i = 0; i < b.length; i++) {
            double res = 0.0;
            for (int k = Math.max(i - 4, 0); k < i; k++) {
                res += ldlt.queryL(i, k) * z[k];
            }
            z[i] = b[i] - res;
        }

        // recalculation
        headRecalculation(ts, z, b, ldlt, tau);
        tailRecalculation(ts, z, b, ldlt);

        // backward substitution
        double[] y = new double[2 * ts.length];
        for (int i = 2 * ts.length - 1; i > -1; i--) {
            double tmp = 0.0;
            for (int k = Math.min(i + 4, 2 * ts.length - 1); k > i; k--) {
                if (k > 2 * ts.length - 5)
                    tmp += ldlt.queryEndL(k, i, ts.length) * y[k];
                else
                    tmp += ldlt.queryL(k, i) * y[k];
            }
            if (i > 2 * ts.length - 5)
                y[i] = z[i] / ldlt.queryEndD(i, i, ts.length) - tmp;
            else
                y[i] = z[i] / ldlt.queryD(i, i) - tmp;
        }

        for (int i = 0; i < ts.length; ++i) {
            result[0][i] = y[2 * i];
            result[1][i] = y[2 * i + 1];
            result[2][i] = b[2 * i] - result[0][i] - result[1][i];
        }

        return result;
    }
}
