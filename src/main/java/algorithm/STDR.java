package algorithm;

import java.util.Arrays;

public class STDR {
    public double[][] decompose(double[] ts, int period) {
        double[] y = ts;
        int T = period;
        int N = y.length;
        int K = N / T;

        double[][] yy = new double[K][T];
        for (int i = 0; i < K; i++) {
            System.arraycopy(y, i * T, yy[i], 0, T);
        }

        // Trend
        double[] ym = new double[K];
        for (int i = 0; i < K; i++) {
            ym[i] = mean(yy[i]);
        }
        double[] trend = new double[N];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < T; j++) {
                trend[i * T + j] = ym[i];
            }
        }

        // Dispersion
        double[] yd = new double[K];
        for (int i = 0; i < K; i++) {
            yd[i] = dispersion(yy[i], ym[i]);
        }
        double[] dispersion = new double[N];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < T; j++) {
                dispersion[i * T + j] = yd[i];
            }
        }

        // Seasonal for STD
        double[] seasonal = new double[N];
        for (int i = 0; i < T * K; i++) {
            seasonal[i] = (y[i] - trend[i]) / dispersion[i];
        }

        // Seasonal and reminder for STDR
        double[][] q = new double[K][T];
        for (int i = 0; i < K; i++) {
            System.arraycopy(seasonal, i * T, q[i], 0, T);
        }
        double[] sp = new double[T];
        for (int i = 0; i < T; i++) {
            double[] col = new double[K];
            for (int j = 0; j < K; j++) {
                col[j] = q[j][i];
            }
            sp[i] = mean(col);
        }
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < T; j++) {
                seasonal[i * T + j] = sp[j];
            }
        }

        // extend
        if (N - T * K >= 0) System.arraycopy(sp, 0, seasonal, K * T, N - T * K);
        for (int i = T * K; i < N; i++) {
            trend[i] = trend[T * K - 1];
            dispersion[i] = dispersion[T * K - 1];
        }

        double[] residual = new double[N];
        for (int i = 0; i < N; i++) {
            residual[i] = y[i] - (seasonal[i] * dispersion[i] + trend[i]);
        }

        return new double[][]{trend, multiply(dispersion, seasonal), residual};
    }

    private static double mean(double[] arr) {
        double sum = 0.0;
        for (double v : arr) {
            sum += v;
        }
        return sum / arr.length;
    }

    private static double dispersion(double[] arr, double mean) {
        double sumSq = 0.0;
        for (double v : arr) {
            sumSq += Math.pow(v - mean, 2);
        }
        return Math.sqrt(sumSq);
    }

    private static double[] multiply(double[] arr1, double[] arr2) {
        double[] result = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++) {
            result[i] = arr1[i] * arr2[i];
        }
        return result;
    }

//    public static void main(String[] args) {
//        // Example usage
//        double[] y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//        int T = 4;
//        double[][] result = decompose(y, T);
//        System.out.println("Trend: " + Arrays.toString(result[0]));
//        System.out.println("Seasonal: " + Arrays.toString(result[1]));
//        System.out.println("Residual: " + Arrays.toString(result[2]));
//    }
}