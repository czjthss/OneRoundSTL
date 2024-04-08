package algorithm;

import java.util.*;

import algorithm.utils.FormatUtil;
import algorithm.utils.OneShotSTLUtil;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;

public class OneShotSTL {
    private final static int coldStartPeriodNumber = 5;
    private final Map<String, double[]> results = new HashMap<>();

    public OneShotSTL(long[] time, double[] ts, int period, int shiftWindow, double lambda1, double lambda3, int maxIter, String f, String h, String g, int ratio) {
        int trainTestSplit = coldStartPeriodNumber * period;

        // input
        List<Double> y = FormatUtil.doubleToList(ts);
        OneShotSTLUtil oneShotSTLModel = new OneShotSTLUtil(period, shiftWindow, 10000, lambda1, 1.0, lambda3, maxIter, f, h, g, ratio);

        // output
        int size = y.size();
        double[] trend = new double[size];
        double[] seasonal = new double[size];
        double[] residual = new double[size];

        // cold start
        List<Double> initY = y.subList(0, trainTestSplit);
        oneShotSTLModel.initialize(initY);

        // decompose
        Map<String, Double> res;
        for (int i = trainTestSplit; i < size; i++) {
            res = oneShotSTLModel.decompose(y.get(i), i - trainTestSplit);
            // record
            trend[i] = res.get("trend");
            seasonal[i] = res.get("seasonal");
            residual[i] = res.get("residual");
        }

        results.put("trend", trend);
        results.put("seasonal", seasonal);
        results.put("residual", residual);
    }

    public Map<String, double[]> get_decompose_results() {
        return results;
    }
}

