package algorithm;

import algorithm.utils.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WindowSTL {
    private final static int coldStartPeriodNumber = 5;

    public WindowSTL(long[] time, double[] ts, int period, int slidingWindow) {
        int trainTestSplit = coldStartPeriodNumber * period;

        // input
        List<Double> y = FormatUtil.doubleToList(ts);
        WindowSTLUtil windowSTLModel = new WindowSTLUtil(period, slidingWindow);

        // output
        int size = y.size();
        Map<String, double[]> results = new HashMap<>();
        double[] trend = new double[size];
        double[] seasonal = new double[size];
        double[] residual = new double[size];

        // cold start
        List<Double> initY = y.subList(0, trainTestSplit);
        windowSTLModel.initialize(initY);

        // decompose
        long begin = System.nanoTime();
        Map<String, Double> res;
        for (int i = trainTestSplit; i < y.size(); i++) {
            res = windowSTLModel.decompose(y.get(i));
            // record
            trend[i] = res.get("trend");
            seasonal[i] = res.get("seasonal");
            residual[i] = res.get("residual");
        }
        long end = System.nanoTime();
        double timeCost = (1.0 * (end - begin) / 1000000000);

        results.put("trend", trend);
        results.put("seasonal", seasonal);
        results.put("residual", residual);
    }
}
