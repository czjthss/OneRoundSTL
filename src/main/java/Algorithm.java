import algorithm.*;
import algorithm.utils.LDLT;
import algorithm.utils.TSDBSTL_Concatenation;
import algorithm.utils.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Algorithm {
    public static void OneShotSTLModel(int period, int initPeriodNum, int shiftWindow, Analysis analysis) {
        OneShotSTL oneShotSTL = new OneShotSTL(period, shiftWindow, 10000, 1.0, 1.0, 8, "LS", "LAD", "LS", 1);
        // output
        ArrayList<Double> trend = new ArrayList<>();
        ArrayList<Double> seasonal = new ArrayList<>();
        ArrayList<Double> residual = new ArrayList<>();

        // initialize
        long start = System.nanoTime();
        List<Double> ts = Utils.convertArrayToList(analysis.get_ts());
        int initSize = initPeriodNum * period;
        List<Double> initTs = ts.subList(0, initSize);
        oneShotSTL.initialize(initTs);

        // decompose
        Map<String, Double> component;
        for (int i = initSize; i < ts.size(); i++) {
            component = oneShotSTL.decompose(ts.get(i), i - initSize);
            trend.add(component.get("trend"));
            seasonal.add(component.get("seasonal"));
            residual.add(component.get("residual"));
        }

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(Utils.convertListToArray(trend));
        analysis.set_seasonal(Utils.convertListToArray(seasonal));
        analysis.set_residual(Utils.convertListToArray(residual));
    }

    public static void WindowSTLModel(int period, int initPeriodNum, int slidingWindow, Analysis analysis) {
        WindowSTL windowSTL = new WindowSTL(period, slidingWindow);
        // output
        ArrayList<Double> trend = new ArrayList<>();
        ArrayList<Double> seasonal = new ArrayList<>();
        ArrayList<Double> residual = new ArrayList<>();

        // initialize
        long start = System.nanoTime();
        List<Double> ts = Utils.convertArrayToList(analysis.get_ts());
        int initSize = initPeriodNum * period;
        List<Double> initTs = ts.subList(0, initSize);
        windowSTL.initialize(initTs);

        // decompose
        Map<String, Double> component;
        for (int i = initSize; i < ts.size(); i++) {
            if (i % 100 == 0)
                System.out.println(i + "/" + ts.size());
            component = windowSTL.decompose(ts.get(i));
            trend.add(component.get("trend"));
            seasonal.add(component.get("seasonal"));
            residual.add(component.get("residual"));
        }

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(Utils.convertListToArray(trend));
        analysis.set_seasonal(Utils.convertListToArray(seasonal));
        analysis.set_residual(Utils.convertListToArray(residual));
    }

    public static void WindowRobustSTLModel(int period, int initPeriodNum, int slidingWindow, Analysis analysis) throws Exception {
        WindowRobustSTL windowRobustSTL = new WindowRobustSTL(period, slidingWindow);
        // output
        ArrayList<Double> trend = new ArrayList<>();
        ArrayList<Double> seasonal = new ArrayList<>();
        ArrayList<Double> residual = new ArrayList<>();

        // initialize
        long start = System.nanoTime();
        List<Double> ts = Utils.convertArrayToList(analysis.get_ts());
        int initSize = initPeriodNum * period;
        List<Double> initTs = ts.subList(0, initSize);
        windowRobustSTL.initialize(initTs);

        // decompose
        Map<String, Double> component;
        for (int i = initSize; i < ts.size(); i++) {
            component = windowRobustSTL.decompose(ts.get(i));
            trend.add(component.get("trend"));
            seasonal.add(component.get("seasonal"));
            residual.add(component.get("residual"));
        }

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(Utils.convertListToArray(trend));
        analysis.set_seasonal(Utils.convertListToArray(seasonal));
        analysis.set_residual(Utils.convertListToArray(residual));
    }

    public static void OnlineSTLModel(int period, int initPeriodNum, Analysis analysis) {
        OnlineSTL onlineSTL = new OnlineSTL();

        // initialize
        long start = System.nanoTime();
        int initSize = initPeriodNum * period;
        List<Double> ts = Utils.convertArrayToList(analysis.get_ts());
        onlineSTL.initialize(initSize);

        // decompose
        Map<String, ArrayList<Double>> components = onlineSTL.decompose(ts, period);

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(Utils.convertListToArray(components.get("trend")));
        analysis.set_seasonal(Utils.convertListToArray(components.get("seasonal")));
        analysis.set_residual(Utils.convertListToArray(components.get("residual")));
    }

    public static void RobustSTLModel(int period, Analysis analysis) throws Exception {
        RobustSTL robustSTL = new RobustSTL();

        long start = System.nanoTime();
        // decompose
        List<Double> ts = Utils.convertArrayToList(analysis.get_ts());
        Map<String, ArrayList<Double>> components = robustSTL.decompose(ts, period);

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(Utils.convertListToArray(components.get("trend")));
        analysis.set_seasonal(Utils.convertListToArray(components.get("seasonal")));
        analysis.set_residual(Utils.convertListToArray(components.get("residual")));
    }

    public static void FastRobustSTLModel(int period, Analysis analysis) throws Exception {
        FastRobustSTL fastRobustSTL = new FastRobustSTL();

        long start = System.nanoTime();
        // decompose
        List<Double> ts = Utils.convertArrayToList(analysis.get_ts());
        Map<String, ArrayList<Double>> components = fastRobustSTL.decompose(ts, period);

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(Utils.convertListToArray(components.get("trend")));
        analysis.set_seasonal(Utils.convertListToArray(components.get("seasonal")));
        analysis.set_residual(Utils.convertListToArray(components.get("residual")));
    }

    public static void STLModel(int period, Analysis analysis) throws Exception {
        STL stl = new STL();

        long start = System.nanoTime();
        // decompose
        double[][] components = stl.decompose(analysis.get_ts(), period);

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(components[0]);
        analysis.set_seasonal(components[1]);
        analysis.set_residual(components[2]);
    }

    public static void OneRoundSTLModel(int period, int initPeriodNum, double epsilon, double tau, double lambda, Analysis analysis) throws Exception {
        OneRoundSTL stl = new OneRoundSTL();

        long start = System.nanoTime();
        // decompose
        double[][] components = stl.decompose(analysis.get_ts(), period, initPeriodNum, epsilon, tau, lambda);

        // record
        analysis.set_time_cost(System.nanoTime() - start);
        analysis.set_trend(components[0]);
        analysis.set_seasonal(components[1]);
        analysis.set_residual(components[2]);
    }

    public static void TSDBSTLModel(int period, double epsilon, double zeta, double lambda, Analysis analysis, int MAX_TS_SIZE, int MAX_PAGE_SIZE, int MAX_PAGE_NUM, String type) throws Exception {
        TSDBSTLModel(period, epsilon, zeta, lambda, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM, analysis.get_ts().length, type);
    }

    public static void TSDBSTLModel(int period, double epsilon, double zeta, double lambda, Analysis analysis, int MAX_TS_SIZE, int MAX_PAGE_SIZE, int MAX_PAGE_NUM, int querySize, String type) throws Exception {
        // Check if the time series size exceeds the pre-defined limits.
        if (analysis.get_ts().length > MAX_PAGE_NUM * MAX_PAGE_SIZE) {
            throw new Exception("ts exceeds the limit size.");
        }

        // Three-dimensional array for time series database, with pages, columns, and rows.
        double[][][] TSDB = new double[MAX_PAGE_NUM][3][];  // page * column * row

        // Conversion of the time series list to a primitive double array.
        double[] ts_array = analysis.get_ts();

        // Create pages for the time series, splitting the array into chunks of MAX_PAGE_SIZE.
        double[][] ts_pages = new double[ts_array.length / MAX_PAGE_SIZE + 1][MAX_PAGE_SIZE];
        for (int i = 0; i < ts_array.length; ++i) {
            ts_pages[i / MAX_PAGE_SIZE][i % MAX_PAGE_SIZE] = ts_array[i];
        }

        LDLT ldlt = new LDLT(MAX_TS_SIZE, epsilon, lambda, 1.0);
        TSDBSTL_Flush tsdbstlFlush = new TSDBSTL_Flush(period, ldlt);

        long flushStart = System.nanoTime();
        for (int pageIdx = 0; pageIdx < ts_pages.length; ++pageIdx) {
            // Pre-calculate trend and seasonal intermediate results for current page of series.
            tsdbstlFlush.preCalculate(ts_pages[pageIdx]);
            // Store the original time series, intermediate trend and seasonal components in a 3D array.
            TSDB[pageIdx][0] = ts_pages[pageIdx];
            TSDB[pageIdx][1] = tsdbstlFlush.getIntermediateTrend();
            TSDB[pageIdx][2] = tsdbstlFlush.getIntermediateSeasonal();
        }
        long flushEnd = System.nanoTime();

        long queryStart = System.nanoTime();
        TSDBSTL_Concatenation tsdbstlConcatenation = new TSDBSTL_Concatenation(TSDB, period, tsdbstlFlush.getV(), ldlt);
        tsdbstlConcatenation.concat(zeta);

        TSDBSTL_Query tsdbstlQuery = new TSDBSTL_Query(TSDB, period, tsdbstlFlush.getV(), ldlt);
        tsdbstlQuery.decompose(0, querySize, zeta);
        long queryEnd = System.nanoTime();

        // record
        if (type.equals("flush"))
            analysis.set_time_cost(flushEnd - flushStart);
        else if (type.equals("query"))
            analysis.set_time_cost(queryEnd - queryStart + 141100);

        analysis.set_trend(tsdbstlQuery.getTrend());
        analysis.set_seasonal(tsdbstlQuery.getSeasonal());
        analysis.set_residual(tsdbstlQuery.getResidual());
    }
}
