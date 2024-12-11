package algorithm;

//import java.util.ArrayList;

import algorithm.utils.CleanSTLUtil;

public class CleanSTL {
    private double mu, sigma;
    private int choice;

    public double[][] decompose(double[] ts, int period, double k, int max_iter, int choice) throws Exception {
        int size = ts.length;
        this.choice = choice;

        int chunkSize = period * 100;
        int numberOfChunks = (int) Math.ceil((double) size / chunkSize);
        int resultIndex = 0;

        // output
        double[][] result = new double[3][size];

        for (int chunk = 0; chunk < numberOfChunks; chunk++) {
            int start = chunk * chunkSize;
            int end = Math.min(size, start + chunkSize);
            double[] tsc = new double[end - start];
            System.arraycopy(ts, start, tsc, 0, end - start);

            // Decompose the current chunk
            double[][] resultc = decomposeSingle(tsc, period, k, max_iter);

            // Store the results back into the respective arrays
            System.arraycopy(resultc[0], 0, result[0], resultIndex, resultc[0].length);
            System.arraycopy(resultc[1], 0, result[1], resultIndex, resultc[1].length);
            System.arraycopy(resultc[2], 0, result[2], resultIndex, resultc[2].length);

            resultIndex += chunkSize;
        }

        return result;
    }

    private double[][] decomposeSingle(double[] ts, int period, double k, int max_iter) throws Exception {
        double[][] resultc = new double[3][];
        double[] errorRecord = new double[ts.length];
        double repairValue;
        if (choice == 3) {
            max_iter = 0;
        }
        int h = 0;
//        int errorNum = 0;
        for (; h < max_iter; ++h) {
            CleanSTLUtil de;
            if (choice == 1) {
                de = new CleanSTLUtil(ts, period, "mean");
            } else {
                de = new CleanSTLUtil(ts, period, "median");
            }
            resultc[0] = de.getTrend();
            resultc[1] = de.getSeasonal();
            resultc[2] = de.getResidual();

            estimate(resultc[2]);

            boolean flag = true;
            for (int i = 0; i < resultc[0].length; ++i) {
                if (sub(resultc[2][i], mu) > k * sigma) {
                    flag = false;
                    // how to generate
                    if (choice == 2) {
                        repairValue = generate(i, ts);
                    } else {
                        repairValue = generate(i, period, resultc[0], resultc[1]);
                    }
//                    errorNum += 1;
                    errorRecord[i] += ts[i] - repairValue;
                    ts[i] = repairValue;
                }
            }
            if (flag) break;
        }

        CleanSTLUtil de = new CleanSTLUtil(ts, period, "mean");
        resultc[0] = de.getTrend();
        resultc[1] = de.getSeasonal();
        resultc[2] = de.getResidual();

        // record error
        for (int i = 0; i < resultc[2].length; ++i) {
            resultc[2][i] += errorRecord[i];
        }
        return resultc;
    }


    private void estimate(double[] residual) {
        // get the sum of array
        double sum = 0.0;
        for (double i : residual) {
            sum += i;
        }

        // get the mean of array
        int length = residual.length;

        mu = sum / length;

        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : residual) {
            standardDeviation += Math.pow(num - mu, 2);
        }

        sigma = Math.sqrt(standardDeviation / length);
    }

    private double generate(int pos, double[] ts) {
        if (pos == 0) {
            return ts[pos + 1];
        } else {
            return ts[pos - 1];
        }
//        else if (pos == ts.length - 1) {
//            return ts[pos - 1];
//        }
//        else {
//            return (ts[pos + 1] + ts[pos - 1]) / 2;
//        }
    }

    private double generate(int pos, int period, double[] trend, double[] seasonal) {
        // in each cycle
//        int i = pos % period;
//        ArrayList<Double> arr = new ArrayList<>();
//
//        for (int j = 0; j < size / period; ++j)
//            if (j * period + i != pos)  // remove anomaly
//                arr.add(residual[j * period + i]);
//        if (i < size % period && i + (size / period) * period != pos)
//            arr.add(residual[i + (size / period) * period]);
//
//        double[] cal_median = new double[arr.size()];
//        for (int j = 0; j < arr.size(); j++)
//            cal_median[j] = arr.get(j);

        if (pos == 0) {
            return seasonal[pos] + trend[pos];
        } else {
            return seasonal[pos] + trend[pos - 1];
        }

//        return get_median(cal_median) + seasonal[pos % period] + trend[pos];
    }

    private double sub(double a, double b) {
        return a > b ? a - b : b - a;
    }

    private double get_median(double[] A) {
        return quickSelect(A, 0, A.length - 1, A.length / 2);
    }

    private double quickSelect(double[] A, int l, int r, int k) {
        if (l >= r) return A[k];

        double x = A[l];
        int i = l - 1, j = r + 1;
        while (i < j) {
            do i++; while (A[i] < x);
            do j--; while (A[j] > x);
            if (i < j) {
                double temp = A[i];
                A[i] = A[j];
                A[j] = temp;
            }
        }

        if (k <= j) return quickSelect(A, l, j, k);
        else return quickSelect(A, j + 1, r, k);
    }
}
