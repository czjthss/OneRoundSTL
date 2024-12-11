import java.io.*;

public class ExperimentForecast {
    private static final String INPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/input/cleanstl/";
    private static final String OUTPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/output/cleanstl/";
    // dataset
    private static final String[] forecastFileList = {
            "power_5241600.csv",
            "voltage_22825440.csv",
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

    public static void recordForecasting(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "tsf.txt", true);
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

    public static void write2csv(double[] data) {
        // Specify the output CSV file path
        String outputFile = OUTPUT_DIR + "test.csv";

        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.append("timestamp,value,\n");
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
        for (String dataset : forecastFileList) {
            String datasetName = dataset.split("_")[0];

            int period = datasetName.equals("power") ? 144 : 1440;
            int dataSize = datasetName.equals("power") ? 144000 : 1440000;

            int initPeriodNum = 5;
            int shiftWindow = 1000;  // 4OneShotSTL
            int slidingWindow = 720;  // 4WindowSTL

            double epsilon = 1e-8, zeta = 1e-8, lambda = 1.0, thres;
            int MAX_TS_SIZE = 40000010, MAX_PAGE_NUM = 100;
            int MAX_PAGE_SIZE = datasetName.equals("power") ? 1440 : 14400;
            double missingRate = 0.0;
            double errorRate = 10.;
            double errorRange = 5.;
            boolean init = false;
            int choice = 0;
            int k = 6;

            // record
            recordForecasting(datasetName + " forecast 0,1,2,3,4\n");
            recordTime(datasetName + " forecast 0,1,2,3,4\n");

//            for (errorRate = 0.0; errorRate < 5.0; errorRate += 0.5) {
            for (int max_iter = 1; max_iter <= 10; max_iter += 1) {
//                for (int choice = 0; choice < 4; choice++) {

                MAX_PAGE_NUM = MAX_TS_SIZE / MAX_PAGE_SIZE;

                Analysis analysis = LoadData.loadTimeSeriesData(INPUT_DIR + dataset, dataSize);
                double[] ts = analysis.get_ts();
                double[] label = analysis.get_label();
                double[] gt_residual = analysis.get_gt_residual();

                LoadData.addError(ts, label, gt_residual, errorRate, errorRange);

                thres = calThreshold(ts);
                zeta = thres;
                epsilon = thres;

                for (String method : methodList) {
//                        System.out.println(dataSize + " " + method);
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
                        case "STDR":
                            init = false;
                            Algorithm.STDRModel(period, analysis);
                            break;
                        case "CleanSTL":
                            init = false;
                            Algorithm.CleanSTLModel(period, analysis, k, max_iter, choice);
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
                    recordForecasting(analysis.get_forecast_rmse(init, period) + ",");
                    recordTime(analysis.get_time_cost() + ",");
                }
//                }
                recordForecasting("\n");
                recordTime("\n");
            }
        }
    }
}
