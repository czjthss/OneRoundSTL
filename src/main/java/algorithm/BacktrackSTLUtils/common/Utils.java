package algorithm.BacktrackSTLUtils.common;

import java.util.List;

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Haoyu Wang
 * @date 2023/10/11
 */
public class Utils {
    /**
     * Calculate the final result of bilateral filter
     *
     * @param diffT      the time difference between the neighbor and the reference data point
     * @param v          the value of the neighbor
     * @param refV       the value of the reference data point
     * @param deltaTime  the standard deviation of time dimension
     * @param deltaValue the standard deviation of value dimension
     * @return the result of bilateral filter
     */
    public static double bilateralFilter(List<Integer> diffT, List<Double> v, double refV, double deltaTime,
        double deltaValue) {
        double totalWeight = 0, temp = 0;
        for (int i = 0; i < v.size(); i++) {
            double weight = bilateralFilterWeight(diffT.get(i), 0, v.get(i), refV, deltaTime, deltaValue);
            totalWeight += weight;
            temp += weight * v.get(i);
        }
        if (totalWeight == 0) {
            // At that time, refV is an outlier.
            // Thus, all weights are smaller than the lower bound of the floating point number, leading to 0.
            // The result of the filter should be the nearest value in the neighborhood.
            totalWeight = 1;
            temp = Double.MAX_VALUE;
            for (Double aDouble : v) {
                if (FastMath.abs(temp - refV) > FastMath.abs(aDouble - refV)) {
                    temp = aDouble;
                }
            }
        }
        return temp / totalWeight;
    }

    /**
     * Calculate the weight of given point in bilateral filter
     *
     * @param t          the time (index) of the data point
     * @param refT       the time (index) of the reference data point
     * @param v          the value of the data point
     * @param refV       the value of the reference data point
     * @param deltaTime  the standard deviation of time dimension
     * @param deltaValue the standard deviation of value dimension
     * @return weight of given point
     */
    private static double bilateralFilterWeight(int t, int refT, double v, double refV, double deltaTime,
        double deltaValue) {
        // Avoid division by 0
        deltaTime = Math.max(deltaTime, 1e-10);
        deltaValue = Math.max(deltaValue, 1e-10);
        // Use multiplication instead of FastMath.pow() since the latter is very slow
        double expT = -(t - refT) * (t - refT) / (deltaTime * deltaTime * 2);
        double expV = -(v - refV) * (v - refV) / (deltaValue * deltaValue * 2);
        return FastMath.exp(expT + expV);
    }

    /**
     * Return the elements of the array x minus the elements of the array y
     *
     * @param x array x
     * @param y array y
     * @return the result array
     */
    public static double[] arrayMinus(double[] x, double[] y) {
        double[] ans = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ans[i] = x[i] - y[i];
        }
        return ans;
    }

    /**
     * Check if the given data point is a local maximum
     *
     * @param x      sequence
     * @param target the index of the given data point
     * @param from   the start index of the subsequence (inclusive)
     * @param to     the end index of the subsequence (exclusive)
     * @return true if the point is the local maximum, otherwise false
     */
    public static boolean isLocalMax(double[] x, int target, int from, int to) {
        for (int i = from; i < to; i++) {
            if (i == target) {
                continue;
            }
            if (x[i] >= x[target]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculate the mean absolute error
     *
     * @param result the result array
     * @param truth  the truth array
     * @return the mean absolute error
     */
    public static double meanAbsoluteError(List<Double> result, List<Double> truth) {
        double sum = 0;
        for (int i = 0; i < result.size(); i++) {
            sum += Math.abs(result.get(i) - truth.get(i));
        }
        return sum / result.size();
    }

    /**
     * Calculate the estimated standard deviation of values
     *
     * @param values    the values
     * @param period    the period length
     * @param sideWidth the side width of the window
     * @return the estimated standard deviation
     */
    public static double estimatedStandardDeviation(double[] values, int period, int sideWidth) {
        double[] diff = new double[values.length - period];
        for (int i = 0; i < diff.length; i++) {
            diff[i] = Double.MAX_VALUE;
            for (int j = i + period - sideWidth; j <= Math.min(values.length - 1, i + period + sideWidth); j++) {
                diff[i] = Math.min(diff[i], Math.abs(values[i] - values[j]));
            }
        }
        return Math.sqrt(StatUtils.variance(diff));
    }

}
