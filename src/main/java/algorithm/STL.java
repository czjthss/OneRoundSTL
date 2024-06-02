package algorithm;

import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;

public class STL {
    public double[][] decompose(double[] ts, int period) throws Exception {
        // output
        double[][] result = new double[3][];

        SeasonalTrendLoess.Builder stlBuilder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess stl = stlBuilder.setPeriodic().setPeriodLength(period).buildSmoother(ts);
        SeasonalTrendLoess.Decomposition stlDecomposition = stl.decompose();

        result[0] = stlDecomposition.getTrend();
        result[1] = stlDecomposition.getSeasonal();
        result[2] = stlDecomposition.getResidual();

        return result;
    }
}


