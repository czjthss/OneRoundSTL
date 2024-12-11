import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class ExperimentAnomalyDetection {
    private static final String INPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/input/cleanstl/";
    private static final String OUTPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/output/cleanstl/";
    // dataset
    private static final String[] anomalyDetectionFileList = {
            "hangxin_c_from2019-08-11to2019-08-30_7428.csv",
            "yinlian_value_from2020-11-10to2020-12-07_8438.csv",
            "liantong_data_from2018-12-19to2019-01-31_8288.csv",
            "IOPS_KPI-43115f2a-baeb-3b01-96f7-4ea14188343c.test.csv",
            "YAHOO_YahooA3Benchmark-TS88_data.csv",
    };

    private static final String[] methodList = {
            "CleanSTL",
//            "TSDBSTL_Query",
//            "TSDBSTL_Flush",
//            "STL",
//            "STDR",
//            "OneShotSTL",
//            "OnlineSTL",
//            "OneRoundSTL",
//            "WindowSTL",
//            "RobustSTL",
//            "WindowRobustSTL",
//            "FastRobustSTL",
    };

    public static void recordAnomalyDetection(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "tsad.txt", true);
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

    public static void write2csv(double[] data) {
        // Specify the output CSV file path
        String outputFile = INPUT_DIR + "test.csv";

        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.append("timestamp,value\n");
            for (int i = 0; i < data.length; i++) {
                writer.append(String.valueOf(i));
                writer.append(","); // Add a comma between values
                writer.append(String.valueOf(data[i]));
                writer.append(",\n"); // Add a comma between values
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        for (String dataset : anomalyDetectionFileList) {
            String datasetName = dataset.split("_")[0];

            int period;
            if (datasetName.equals("IOPS"))
                period = 144;
            else if (datasetName.equals("YAHOO"))
                period = 12;
            else
                period = 288;

            int dataSize = 0x3f3f3f3f;
//            int dataSize = datasetName.equals("power") ? 144000 : 1440000;

            int initPeriodNum = 5;
            int shiftWindow = 1000;  // 4OneShotSTL
            int slidingWindow = 720;  // 4WindowSTL

            double epsilon = 1e-8, zeta = 1e-8, lambda = 1.0, thres;
            int MAX_TS_SIZE = 40000010, MAX_PAGE_NUM = 100;
            int MAX_PAGE_SIZE = datasetName.equals("power") ? 14400 : 144000;
            double missingRate = 0.0;
            double errorRate = 10.;
            double errorRange = 5.;
            boolean init = false;
            double k = 3;
            System.out.println(dataset);

            // record
            recordAnomalyDetection(datasetName + " anomaly_detection 0,1,2,3,4\n");

            for (int choice = 0; choice < 4; choice++) {
                MAX_PAGE_NUM = MAX_TS_SIZE / MAX_PAGE_SIZE;

                Analysis analysis = LoadData.loadAnomalyDetectionTimeSeriesData(INPUT_DIR + dataset, dataSize);
                double[] ts = analysis.get_ts();
                double[] label = analysis.get_label();
                double[] gt_residual = analysis.get_gt_residual();

                LoadData.addError(ts, label, gt_residual, errorRate, errorRange);
//                write2csv(ts);

                thres = calThreshold(ts);
                zeta = thres;
                epsilon = thres;

                for (String method : methodList) {
                    System.out.println(dataSize + " " + method);
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
                            Algorithm.CleanSTLModel(period, analysis, k, 1000, choice);
                            break;
                        case "TSDBSTL_Flush":
                            Algorithm.TSDB_OneRoundSTLModel(period, epsilon, zeta, lambda, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM, "flush");
                            break;
                        case "TSDBSTL_Query":
                            Algorithm.TSDB_OneRoundSTLModel(period, epsilon, zeta, lambda, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM, "query");
                            break;
                        default:
                            System.out.println("!!!Wrong!!!");
                    }
//                System.out.println("time_cost: " + 10e-9 * analysis.get_time_cost());
                    recordAnomalyDetection(analysis.get_anomaly_detection_roc_auc(init, period) + ",");
                }
                recordAnomalyDetection("\n");
            }
        }
    }
}
