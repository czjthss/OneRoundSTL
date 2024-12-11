import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;

public class ExperimentSynthetic {
    private static final String OUTPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/output/decomposition/";

    private static final String[] methodList = {
            "OneRoundSTL_Query",
//            "OneRoundSTL_Flush",
//            "OneRoundSTL",
//            "CleanSTL",
            "STL",
            "OneShotSTL",
            "OnlineSTL",
            "BacktrackSTL",
//            "RobustSTL",
//            "FastRobustSTL",
    };

    public static void recordTrend(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "trend.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void recordSeasonal(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "seasonal.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void recordResidual(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "residual.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void recordTime(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "time.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static double calThreshold(double[] ts) {
        int maxLength = 0, curLength;
        for (double value : ts) {
            curLength = String.valueOf(value).replace(".", "").length();
            if (curLength > maxLength)
                maxLength = curLength;
        }
        return Math.pow(10, -maxLength);
    }

    public static void main(String[] args) throws Exception {
        int period = 144;//308885
        int initPeriodNum = 5;
        int shiftWindow = 1000;  // 4OneShotSTL
        int slidingWindow = 720;  // 4WindowSTL
        double k = 6; // 4CleanSTL
        int choice = 0; // 4CleanSTL
        int max_iter = 10; // 4CleanSTL


        // parameter
        int Qsize = 1440000, QsizeBase = 1000000;
        double missingRate = 1., missingRateBase = 1.;
        int missingLength = 5, missingLengthBase = 1;
        double errorRate = 1., errorRateBase = 1.;
        double errorRange = 0.5, errorRangeBase = 1.;

        int MAX_TS_SIZE = 40000010, MAX_PAGE_NUM = 100;
        int MAX_PAGE_SIZE = 1000000;
        double epsilon = 1e-8, zeta = 1e-8, lambda = 1.0, thres;

        String datasetName = "triangle";  // triangle square

//        recordTime(datasetName + " qsize 1m,2m,3m,4m,5m,6m,7m,8m,9m,10m\n");
        recordTime(datasetName + " mlength 1,2,3,4,5,6,7,8,9,10\n");
        recordTrend(datasetName + " mrate 1,2,3,4,5,6,7,8,9,10\n");
        recordSeasonal(datasetName + " mrate 1,2,3,4,5,6,7,8,9,10\n");
        recordResidual(datasetName + " mrate 1,2,3,4,5,6,7,8,9,10\n");

//        int[] missingArray = new int[11];
//        for (int i = 0; i < 11; i++) {
//            missingArray[i] = (int) Math.pow(10, 0.18 * i);
//            System.out.println(missingArray[i]);
//        }

//        for (int base = 90; base <= 110; base += 10) {
//            missingLength = missingLengthBase * base;
//            missingRate = missingRateBase * base;
//        for (int idx = 0; idx <= 10; idx += 1) {
//            missingRate = missingArray[idx];
//        for (choice = 0; choice < 4; choice++) {
        Analysis analysis;
        if (datasetName.equals("square")) {
            analysis = LoadData.squareWave(Qsize, period);
        } else {
            analysis = LoadData.triangleWave(Qsize, period);
        }

        analysis.set_init_num(initPeriodNum * period);
        boolean init = false;
        // add nan and error
        double[] ts = analysis.get_ts();
        double[] label = analysis.get_label();
        double[] gt_residual = analysis.get_gt_residual();

//        LoadData.addNan(ts, period, missingRate, missingLength);
//        LoadData.addError(ts, label, gt_residual, errorRate, errorRange);
//        thres = calThreshold(ts);
//        thres = 1e-8;
//        zeta = thres;
//        epsilon = thres;

        for (String method : methodList) {
            System.out.println(method);
            // method
            switch (method) {
                case "OneShotSTL":
                    init = true;
                    Algorithm.OneShotSTLModel(period, initPeriodNum, shiftWindow, analysis);
                    break;
                case "WindowSTL":
                    init = true;
                    Algorithm.WindowSTLModel(period, initPeriodNum, slidingWindow, analysis);
                    break;
                case "WindowRobustSTL":
                    init = true;
                    Algorithm.WindowRobustSTLModel(period, initPeriodNum, slidingWindow, analysis);
                    break;
                case "OnlineSTL":
                    init = true;
                    Algorithm.OnlineSTLModel(period, initPeriodNum, analysis);
                    break;
                case "RobustSTL":
                    init = false;
                    Algorithm.RobustSTLModel(period, analysis);
                    break;
                case "FastRobustSTL":
                    init = false;
                    Algorithm.FastRobustSTLModel(period, analysis);
                    break;
                case "OneRoundSTL":
                    init = false;
                    Algorithm.OneRoundSTLModel(period, initPeriodNum, 1e-8, 1e-8, 1.0, analysis);
                    break;
                case "STL":
                    init = false;
                    Algorithm.STLModel(period, analysis);
                    break;
                case "CleanSTL":
                    init = false;
                    Algorithm.CleanSTLModel(period, analysis, k, max_iter, choice);
                    break;
                case "BacktrackSTL":
                    init = false;
                    Algorithm.BacktrackSTLModel(period, analysis);
                    break;
                case "OneRoundSTL_Flush":
                    Algorithm.TSDB_OneRoundSTLModel(period, epsilon, zeta, lambda, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM, "flush");
                    break;
                case "OneRoundSTL_Query":
                    Algorithm.TSDB_OneRoundSTLModel(period, epsilon, zeta, lambda, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM, "query");
                    break;
                default:
                    System.out.println("!!!Wrong!!!");
            }
            recordTime(analysis.get_time_cost() + ",");
            recordTrend(analysis.get_trend_rmse(init) + ",");
            recordSeasonal(analysis.get_seasonal_rmse(init) + ",");
            recordResidual(analysis.get_residual_rmse(init) + ",");
            analysis.recordImputeResults(OUTPUT_DIR + "trend.csv");
        }
        recordTime("\n");
        recordTrend("\n");
        recordSeasonal("\n");
        recordResidual("\n");
    }
}
