import be.cylab.java.roc.Roc;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

public class Analysis {
    private final double[] ts, gt_ts;
    private final double[] gt_trend, gt_seasonal, gt_residual;
    private final double[] label;
    private double[] trend, seasonal, residual;
    private double time_cost;
    private int init_num = 0;

    public Analysis(double[] ts) {
        this.ts = ts;
        this.gt_ts = ts.clone();
        this.label = new double[ts.length];
        this.gt_trend = new double[ts.length];
        this.gt_seasonal = new double[ts.length];
        this.gt_residual = new double[ts.length];
    }

    public Analysis(double[] ts, double[] label) {
        this.ts = ts;
        this.gt_ts = ts.clone();
        this.label = label;
        this.gt_trend = new double[ts.length];
        this.gt_seasonal = new double[ts.length];
        this.gt_residual = new double[ts.length];
    }


    public Analysis(double[] ts, double[] trend, double[] seasonal, double[] residual) {
        this.ts = ts;
        this.gt_ts = ts.clone();
        this.label = new double[ts.length];
        this.gt_trend = trend;
        this.gt_seasonal = seasonal;
        this.gt_residual = residual;
    }

    public void set_trend(double[] trend) {
        this.trend = trend;
    }

    public void set_seasonal(double[] seasonal) {
        this.seasonal = seasonal;
    }

    public void set_residual(double[] residual) {
        this.residual = residual;
    }

    public void set_time_cost(double time_cost) {
        this.time_cost = time_cost;
    }

    public void set_init_num(int init_num) {
        this.init_num = init_num;
    }

    public double[] get_ts() {
        return this.ts;
    }

    public double[] get_label() {
        return this.label;
    }

    public double[] get_gt_residual() {
        return this.gt_residual;
    }

    public String get_time_cost() {
        return String.format("%.3f", 1e-9 * this.time_cost);
    }


    private double getRMSE(double[] array, double[] gt_array, boolean init) {
        double sum = 0.0, gap, mean;
        int arraySize = 0;
        for (int i = init ? init_num : 0; i < array.length; i++) {
            if (!Double.isNaN(ts[i])) {
                gap = init ? array[i - init_num] - gt_array[i] : array[i] - gt_array[i];
                sum += gap * gap;
                arraySize++;
            }
        }
        mean = sum / arraySize;
        return Math.sqrt(mean);
    }

    public String get_trend_rmse(boolean init) {
        write2csv(gt_seasonal);
        return String.format("%.3f", getRMSE(trend, gt_trend, init));
    }

    public String get_seasonal_rmse(boolean init) {
        return String.format("%.3f", getRMSE(seasonal, gt_seasonal, init));
    }

    public String get_residual_rmse(boolean init) {
        return String.format("%.3f", getRMSE(residual, gt_residual, init));
    }

    public static void write2csv(double[] data) {
        // Specify the output CSV file path
        String outputFile = "/Users/chenzijie/Documents/GitHub/data/output/cleanstl/test.csv";

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

    public String get_forecast_rmse(boolean init, int period) {
        double[] forecast_ts = new double[seasonal.length];
        for (int i = 1; i < seasonal.length; i++) {
            if (i - period < 0) {
                forecast_ts[i] = seasonal[i] + trend[i - 1];
            } else {
                forecast_ts[i] = seasonal[i - period] + trend[i - 1];
            }
        }

//        write2csv(gt_ts);
//        write2csv(forecast_ts);
//        for (int i = 1000; i < 1100; i++) {
//            System.out.println(forecast_ts[i] + " " + ts[i]);
//        }
//        System.out.println("##############");

        return String.format("%.6f", getRMSE(forecast_ts, gt_ts, init));
    }

    private double sub(double a, double b) {
        return a > b ? a - b : b - a;
    }

    public String get_anomaly_detection_roc_auc(boolean init, int period) {
        // 1. truth label
        boolean[] truth = new boolean[label.length];
        for (int i = 0; i < label.length; ++i) {
            truth[i] = label[i] == 1.0;
        }
        // 2. estimate
        double[] score = new double[label.length];
        // get the sum of array
        double sum = 0.0;
        for (double i : residual) {
            sum += i;
        }
        // get the mean of array
        int length = residual.length;
        double mu = sum / length;
        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : residual) {
            standardDeviation += Math.pow(num - mu, 2);
        }
        double sigma = Math.sqrt(standardDeviation / length);
        // 3. score
        for (int i = 0; i < residual.length; ++i) {
            if (sub(residual[i], mu) > 3 * sigma) {
                score[i] = 1.0;
            } else {
                score[i] = sub(residual[i], mu) / (3 * sigma);
            }
        }

//        write2csv(residual);
//        for (int i = 0; i < 100; ++i) {
//            System.out.println(score[i] + " " + truth[i]);
//        }
//        System.out.println("##########");

        Roc roc = new Roc(score, truth);

        return String.format("%.3f", roc.computeAUC());
    }


    public void recordImputeResults(String path) {
        // Specify the output CSV file path
        try (FileWriter writer = new FileWriter(path)) {
            writer.append("timestamp,value,\n");
            for (int i = 0; i < trend.length; i++) {
                writer.append(String.valueOf(i));
                writer.append(","); // Add a comma between values
                writer.append(String.valueOf(trend[i]));
                writer.append(",\n"); // Add a comma between values
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        double a = 123;
        String arr = String.valueOf(a);
        arr = arr.replace(".", "");
        System.out.println(arr.length());
    }
}
