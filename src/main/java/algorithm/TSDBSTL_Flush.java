package algorithm;

import algorithm.utils.LDLT;
import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;

import java.util.Arrays;

public class TSDBSTL_Flush {
    private double[] ts;
    private final int period;
    private final double[] v;
    private final LDLT ldlt;
    private boolean isColdStart = false;
    private double[] b;
    private double[] z;

    public TSDBSTL_Flush(int period, LDLT ldlt) {
        this.period = period;
        this.v = new double[period];
        this.ldlt = ldlt;
    }

    private void coldStart(int cs_period_num) {
        int cs_size = cs_period_num * period;
        double[] cs_array = new double[cs_size];

        // head imputation
        headImpute(cs_size);

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
    }

    public void preCalculate(double[] ts) {
        preCalculate(ts, 5);
    }

    public void preCalculate(double[] ts_array, int cs_period_num) {
        this.ts = ts_array;
        if (!isColdStart) {
            coldStart(cs_period_num);
            isColdStart = true;
        }
        // impute before converge
        headImpute(ldlt.getConvergeSize());

        // generate b
        generateB();

        // forward substitution
        z = new double[ts.length * 2];
        for (int i = 0; i < b.length; i++) {
            if (Double.isNaN(b[i])) {
                decompBasedImpute(i);
            }
            double res = 0.0;
            for (int k = Math.max(i - 4, 0); k < i; k++) {
                res += ldlt.queryL(i, k) * z[k];
            }
            z[i] = b[i] - res;
        }
    }

    private void generateB() {
        b = new double[ts.length * 2];
        for (int i = 0; i < ts.length; i++) {
            b[i * 2] = ts[i];
            b[i * 2 + 1] = ts[i] + v[i % period];
        }
    }

    private void headImpute(int bodySize) {
        // Check for leading NaNs (missing values at the start) and impute with the first available non-NaN value
        int leading; // Declare an index for tracking the start of non-NaN segment
        for (leading = 0; leading < ts.length; leading++) {
            if (!Double.isNaN(ts[leading])) {
                // Found the first non-NaN value, impute preceding NaNs with this value
                Arrays.fill(ts, 0, leading, ts[leading]);
                break;
            }
        }

        // impute body
        int left_i, right_i;
        for (int i = leading; i < bodySize; i++) {
            if (Double.isNaN(ts[i])) {
                left_i = i - 1;
                while (Double.isNaN(ts[i])) {
                    i++;
                }
                right_i = i;
                for (int j = left_i + 1; j < right_i; j++) {
                    ts[j] = (ts[right_i] - ts[left_i]) / (right_i - left_i) *
                            (j - left_i) + ts[left_i];
                }
            }
        }
    }

    private void decompBasedImpute(int iImpute) {
        double[] zTemp = Arrays.copyOf(z, z.length);
        double tmp, tauImpute, vImpute;
        for (int i = iImpute - 4; i < iImpute; i++) {
            tmp = 0.0;
            for (int k = Math.max(i - 4, 0); k < i; k++) {
                tmp += ldlt.queryEndL(i, k, iImpute / 2) * zTemp[k];
            }
            zTemp[i] = b[i] - tmp;
        }

        tauImpute = zTemp[iImpute - 2] / ldlt.queryEndD(iImpute - 2, iImpute - 2, iImpute / 2) -
                zTemp[iImpute - 1] / ldlt.queryEndD(iImpute - 1, iImpute - 1, iImpute / 2);
        vImpute = v[(iImpute / 2) % period];

        // impute
        ts[iImpute / 2] = tauImpute + vImpute;
        b[iImpute] = tauImpute + vImpute;
        b[iImpute + 1] = tauImpute + 2 * vImpute;
    }

    public double[] getIntermediateTrend() {
        double[] intermediateTrend = new double[ts.length];
        for (int i = 0; i < ts.length; ++i) {
            intermediateTrend[i] = z[2 * i];
        }
        return intermediateTrend;
    }

    public double[] getIntermediateSeasonal() {
        double[] intermediateSeasonal = new double[ts.length];
        for (int i = 0; i < ts.length; ++i) {
            intermediateSeasonal[i] = z[2 * i + 1];
        }
        return intermediateSeasonal;
    }

    public double[] getV() {
        return v;
    }
}
