import algorithm.utils.Utils;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class LoadData {
    public static void normalize(double[] ts, double[] trend, double[] seasonal, double[] residual) {
        // normalize
        double min = Arrays.stream(ts).min().getAsDouble();
        double max = Arrays.stream(ts).max().getAsDouble();
        for (int i = 0; i < ts.length; ++i) {
            ts[i] = (ts[i] - min) / (max - min);
            trend[i] = (trend[i] - min) / (max - min);
            seasonal[i] = seasonal[i] / (max - min);
            residual[i] = residual[i] / (max - min);
        }
    }

    public static Analysis triangleWave(int size, int period) {
        double[] ts = new double[size];
        double[] trend = new double[size];
        double[] seasonal = new double[size];
        double[] residual = new double[size];
        double trend_now, seasonal_now, residual_now;
        int errorNum = 10;
        Random rand = new Random();
        rand.setSeed(3047);
        int randomNum;

        for (int errorIdx = 0; errorIdx < errorNum; ++errorIdx) {
            randomNum = rand.nextInt(size);
            residual[randomNum] -= 2;
        }

        for (int time = 0; time < size; time++) {
            trend_now = (double) time * 0.003;
//            trend_now = 0;
            seasonal_now = Math.sin(Math.PI * time / ((double) period / 2));
            // data
            trend[time] = trend_now;
            seasonal[time] = seasonal_now;
            ts[time] = trend[time] + seasonal[time] + residual[time];
        }
        // normalize
//        normalize(ts, trend, seasonal, residual);

        return new Analysis(ts, trend, seasonal, residual);
    }

    public static Analysis squareWave(int size, int period) {
        double[] ts = new double[size];
        double[] trend = new double[size];
        double[] seasonal = new double[size];
        double[] residual = new double[size];
        double trend_now, seasonal_now, residual_now;

        for (int time = 0; time < size; time++) {
            trend_now = -(double) time * 0.003;
//            trend_now = 0;
            seasonal_now = (time % period) < (period / 2) ? 1.0 : -1.0;
            if (time % period == (period / 3 * 2)) {
                seasonal_now += 2.0;
            }
            residual_now = 0;
            // data
            trend[time] = trend_now;
            seasonal[time] = seasonal_now;
            residual[time] = residual_now;
            ts[time] = trend_now + seasonal_now + residual_now;
        }
        // normalize
//        normalize(ts, trend, seasonal, residual);

//        for (int i = 0; i < 144; i++) {
//            System.out.println(ts[i]);
//        }
//        System.exit(0);

        return new Analysis(ts, trend, seasonal, residual);
    }

    public static Analysis loadTimeSeriesData(String filename, int dataLen) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filename));
        ArrayList<Double> tsList = new ArrayList<Double>();

        sc.nextLine();  // skip table header
        for (int k = dataLen; k > 0 && sc.hasNextLine(); --k) {  // the size of td_clean is dataLen
            String[] line_str = sc.nextLine().split(",");
            // ts
            double v = Double.parseDouble(line_str[1]);
            tsList.add(v);
            // standardize_prepare
        }
        double[] ts = Utils.convertListToArray(tsList);

        // normalize
        double min = Arrays.stream(ts).min().getAsDouble();
        double max = Arrays.stream(ts).max().getAsDouble();
        for (int i = 0; i < ts.length; ++i) {
            ts[i] = (ts[i] - min) / (max - min);
        }

        return new Analysis(ts);
    }

    public static Analysis loadAnomalyDetectionTimeSeriesData(String filename, int dataLen) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filename));
        ArrayList<Double> tsList = new ArrayList<Double>();
        ArrayList<Double> labelList = new ArrayList<Double>();

        sc.nextLine();  // skip table header
        for (int k = dataLen; k > 0 && sc.hasNextLine(); --k) {  // the size of td_clean is dataLen
            String[] line_str = sc.nextLine().split(",");
            // ts
            double v = Double.parseDouble(line_str[1]);
            tsList.add(v);

            double l = Double.parseDouble(line_str[2]);
            labelList.add(l);
            // standardize_prepare
        }
        double[] ts = Utils.convertListToArray(tsList);
        double[] label = Utils.convertListToArray(labelList);

        // normalize
        double min = Arrays.stream(ts).min().getAsDouble();
        double max = Arrays.stream(ts).max().getAsDouble();
        for (int i = 0; i < ts.length; ++i) {
            ts[i] = (ts[i] - min) / (max - min);
        }

        return new Analysis(ts, label);
    }

    public static void addNan(double[] ts, int period, double missingRate, int missingLength) {
        Random random = new Random();
        random.setSeed(3047);
        int missingLengthNow = 0;

        for (int i = period; i < ts.length - period; i++) {
            if (random.nextDouble() * 100. < missingRate) {
                missingLengthNow = random.nextInt(missingLength) + 1;
                for (; i < ts.length && missingLengthNow > 0; missingLengthNow--, i++)
                    ts[i] = Double.NaN;
            }
        }
    }

    private static double calRange(double[] ts) {  // data range
        double v_min = Double.MAX_VALUE, v_max = Double.MIN_VALUE;
        for (double value : ts) {
            if (value < v_min) v_min = value;
            if (value > v_max) v_max = value;
        }
        return v_max - v_min;
    }

    public static void addError(double[] ts, double[] label, double[] gt_residual, double errorRate, double errorRange) {
        Random random = new Random();
        random.setSeed(3047);

        double tsRange = calRange(ts);
        double error, newValue;
        for (int i = 0; i < ts.length; i++) {
            if (random.nextDouble() * 1000. < errorRate) {
                // mark in label
                label[i] = 1.0;
                // error range
                error = random.nextGaussian() * tsRange * errorRange;
                newValue = ts[i] + error;
                gt_residual[i] += error;
                BigDecimal b = new BigDecimal(newValue);
                ts[i] = b.setScale(8, RoundingMode.HALF_UP).doubleValue();
            }
        }
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
}

