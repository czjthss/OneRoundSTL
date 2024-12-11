import java.io.*;

public class ExperimentReal {
    private static final String INPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/input/decomposition/";
    private static final String OUTPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/output/decomposition/";
    // dataset
    private static final String[] datasetFileList = {
            "power_5241600.csv",
//            "voltage_22825440.csv",
//            "test.csv"
    };

    private static final String[] methodList = {
            "TSDBSTL_Query",
//            "TSDBSTL_Flush",
//            "STL",
//            "STDR",
//            "OneShotSTL",
//            "OnlineSTL",
//            "OneRoundSTL",
//            "WindowSTL",
//            "BacktrackSTL",
//            "RobustSTL",
//            "WindowRobustSTL",
//            "FastRobustSTL",
    };

    public static void recordFlush(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "flush.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void recordQuery(String string) throws Exception {
        FileWriter fileWritter = new FileWriter(OUTPUT_DIR + "query.txt", true);
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
        String recordType = "query";
        for (String dataset : datasetFileList) {
            String datasetName = dataset.split("_")[0];

            int period = datasetName.equals("power") ? 144 : 1440;
            int dataSize = datasetName.equals("power") ? 1000000 : 10000000;

            int initPeriodNum = 5;
            int shiftWindow = 1000;  // 4OneShotSTL
            int slidingWindow = 720;  // 4WindowSTL

            double epsilon = 1e-8, zeta = 1e-8, lambda = 1.0, thres;
            int MAX_TS_SIZE = 40000010, MAX_PAGE_NUM = 100;
            int MAX_PAGE_SIZE = datasetName.equals("power") ? 14400 : 144000;
            double missingRate = 0.0;
            double errorRate = 1.;
            double errorRange = 1.;

            // record
            if (recordType.equals("flush")) {
                recordFlush(datasetName + " mrate 0,1,2,3,4,5,6,7,8,9\n");
//                recordFlush(datasetName + " scala 0.5m,1m,1.5m,2m,2.5m,3m,3.5m,4m,4.5m,5m\n");
//                recordFlush(datasetName + " scala 2m,4m,6m,8m,10m,12m,14m,16m,18m,20m\n");
//                recordFlush(datasetName + " psize 15k,30k,45k,60k,75k,90k,105k,120k,135k,150k\n");
            } else {
//                recordQuery(datasetName + " mrate 0,1,2,3,4,5,6,7,8,9\n");
//                recordQuery(datasetName + " psize 15k,30k,45k,60k,75k,90k,105k,120k,135k,150k\n");
                recordQuery(datasetName + " qsize 0.5m,1m,1.5m,2m,2.5m,3m,3.5m,4m,4.5m,5m\n");
//                recordQuery(datasetName + " qsize 2m,4m,6m,8m,10m,12m,14m,16m,18m,20m\n");
            }

            for (dataSize = 500000; dataSize <= 5000000; dataSize += 500000) {
//            for (missingRate = 10; missingRate <= 10; missingRate += 1) {
//            int MAX_PAGE_SIZE_BASE = MAX_PAGE_SIZE;
//            for (MAX_PAGE_SIZE = MAX_PAGE_SIZE_BASE; MAX_PAGE_SIZE <= MAX_PAGE_SIZE_BASE * 10; MAX_PAGE_SIZE += MAX_PAGE_SIZE_BASE) {
//            for (missingRate = 0.00; missingRate < 0.095; missingRate += 0.01) {
                MAX_PAGE_NUM = MAX_TS_SIZE / MAX_PAGE_SIZE;

                Analysis analysis = LoadData.loadTimeSeriesData(INPUT_DIR + dataset, dataSize);
                double[] ts = analysis.get_ts();

//                LoadData.addNan(ts, missingRate, 5);
//                LoadData.addError(ts, errorRate, errorRange);

//                write2csv(ts);


                thres = calThreshold(ts);
                zeta = thres;
                epsilon = thres;

                for (String method : methodList) {
                    System.out.println(dataSize + " " + method);
                    // method
                    switch (method) {
                        case "OneShotSTL":
                            Algorithm.OneShotSTLModel(period, initPeriodNum, shiftWindow, analysis);
                            break;
                        case "WindowSTL":
                            Algorithm.WindowSTLModel(period, initPeriodNum, slidingWindow, analysis);
                            break;
                        case "WindowRobustSTL":
                            Algorithm.WindowRobustSTLModel(period, initPeriodNum, slidingWindow, analysis);
                            break;
                        case "OnlineSTL":
                            Algorithm.OnlineSTLModel(period, initPeriodNum, analysis);
                            break;
                        case "RobustSTL":
                            Algorithm.RobustSTLModel(period, analysis);
                            break;
                        case "OneRoundSTL":
                            Algorithm.OneRoundSTLModel(period, initPeriodNum, 1e-8, 1e-8, 1.0, analysis);
                            break;
                        case "FastRobustSTL":
                            Algorithm.FastRobustSTLModel(period, analysis);
                            break;
                        case "STL":
                            Algorithm.STLModel(period, analysis);
                            break;
                        case "STDR":
                            Algorithm.STDRModel(period, analysis);
                            break;
                        case "BacktrackSTL":
                            Algorithm.BacktrackSTLModel(period, analysis);
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
                    if (recordType.equals("flush"))
                        recordFlush(analysis.get_time_cost() + ",");
                    else
                        recordQuery(analysis.get_time_cost() + ",");
                }
                if (recordType.equals("flush"))
                    recordFlush("\n");
                else
                    recordQuery("\n");
            }
        }
    }
}
