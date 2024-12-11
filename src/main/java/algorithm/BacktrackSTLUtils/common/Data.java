package algorithm.BacktrackSTLUtils.common;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.StatUtils;

/**
 * @author Haoyu Wang
 * @date 2023/10/11
 */
public class Data {

    private static List<DataPoint> syntheticBase(int maxShift, double outlier, boolean trendJump) {
        int period = 200;
        int total = 3000;
        Random random = new Random(0xc0ffee);

        double[] trend = new double[total];
        if (trendJump) {
            for (int i = 833; i < 1059; i++) {
                trend[i] += 3;
            }
            for (int i = 1059; i < 1558; i++) {
                trend[i] += 2;
            }
            for (int i = 1558; i < 2177; i++) {
                trend[i] += 1;
            }
        }

        double[] season = new double[total];
        double[] mode = new double[period];
        for (int i = 0; i < period; i++) {
            mode[i] = Math.max(0, Math.sin((i - 60) * 2 * Math.PI / period));
        }
        mode[100] += 1;
        double mean = StatUtils.mean(mode);
        for (int i = 0; i < period; i++) {
            mode[i] -= mean;
        }
        int[] shift = new int[total / period];
        shift[6] = maxShift;
        shift[13] = -maxShift;
        for (int i = 0; i < total / period; i++) {
            for (int j = 0; j < period; j++) {
                season[i * period + j] = mode[(j + shift[i] + period) % period];
            }
        }

        double[] residual = new double[total];
        for (int i = 0; i < total; i++) {
            residual[i] = 0.03 * random.nextGaussian();
        }
        residual[2023] += outlier;

        List<DataPoint> list = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            DataPoint point = new DataPoint(trend[i] + season[i] + residual[i], trend[i], season[i]);
            list.add(point);
        }
        return list;
    }

    public static void generateSynthetic() throws Exception {
        CsvUtils.toCsv(syntheticBase(5, 10, true), "dataset/synthetic.csv");
    }

    public static void generateScalability() throws Exception {
        List<DataPoint> list = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            list.addAll(syntheticBase(0, 10, false));
        }
        CsvUtils.toCsv(list, "dataset/scalability.csv");
    }

    public static void generateData() throws Exception {
        generateSynthetic();
        generateScalability();
    }

}
