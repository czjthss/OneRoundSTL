package algorithm.BacktrackSTLUtils;

import java.util.ArrayList;

import algorithm.BacktrackSTLUtils.common.Utils;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Haoyu Wang
 * @date 2022/11/9
 */
public class PeriodDecomposition {

    private double[] value;
    private double[] trend;
    private double[] season;
    private double[] residual;

    private final int sideWidth;
    private final double delta;
    private final int periodLength;
    private final int periodNum;

    private int lastJump = 0;

    public PeriodDecomposition(int sideWidth, double delta, int periodLength, int periodNum) {
        this.sideWidth = sideWidth;
        this.delta = delta;
        this.periodLength = periodLength;
        this.periodNum = periodNum;
    }

    /**
     * Decompose the given periodic values into trend, season and residual.
     *
     * @param value the periodic values
     */
    public void decompose(double[] value) {
        this.value = value;
        double[] denoiseSample = denoise(value);
        this.trend = extractTrend(denoiseSample);
        this.season = extractSeason(Utils.arrayMinus(denoiseSample, trend));
        this.residual = extractResidual(value, trend, season);
    }

    private double[] denoise(double[] sample) {
        double[] denoiseSample = new double[sample.length];
        for (int i = 0; i < sample.length; i++) {
            denoiseSample[i] = denoise(sample, i);
        }
        return denoiseSample;
    }

    private double denoise(double[] sample, int i) {
        ArrayList<Integer> diffT = new ArrayList<>();
        ArrayList<Double> v = new ArrayList<>();
        for (int j = Math.max(0, i - sideWidth); j <= Math.min(sample.length - 1, i + sideWidth); j++) {
            diffT.add(i - j);
            v.add(sample[j]);
        }
        return Utils.bilateralFilter(diffT, v, sample[i], sideWidth, delta);
    }

    private double[] extractTrend(double[] sample) {
        // Calculate the preliminary trend with moving average
        double[] preliminaryTrend = new double[sample.length];
        movingAverage(sample, preliminaryTrend, 0, sample.length);
        // Seasonal difference of the preliminary trend
        double[] seasonalDiffTrend = seasonalAbsDiff(preliminaryTrend);
        // Look for the turning points
        ArrayList<Integer> turnPoints = new ArrayList<>();
        turnPoints.add(0);
        for (int i = 0; i + periodLength <= seasonalDiffTrend.length; i++) {
            if (Utils.isLocalMax(seasonalDiffTrend, i + periodLength / 2, i, i + periodLength) &&
                    checkTurnPoint(sample, i + periodLength, periodLength)) {
                turnPoints.add(i + periodLength);
                this.lastJump = i + periodLength;
            }
        }
        turnPoints.add(sample.length);
        // Recalculate the trend with the turning points
        double[] trend = new double[sample.length];
        for (int i = 0; i < turnPoints.size() - 1; i++) {
            movingAverage(sample, trend, turnPoints.get(i), turnPoints.get(i + 1));
        }
        return trend;
    }

    private void movingAverage(double[] sample, double[] mv, int from, int to) {
        int len = Math.min(to - from, periodLength);
        // Initialize the window
        int left = from;
        double sum = 0;
        for (int i = from; i < from + len; i++) {
            sum += sample[i];
        }
        // Calculate the moving average incrementally
        for (int i = from; i < to; i++) {
            if (left < i - periodLength && left + len < to) {
                sum = sum - sample[left] + sample[left + len];
                left++;
            }
            mv[i] = sum / len;
        }
    }

    private double[] seasonalAbsDiff(double[] x) {
        double[] ans = new double[x.length - periodLength];
        for (int i = 0; i < ans.length; i++) {
            ans[i] = FastMath.abs(x[i + periodLength] - x[i]);
        }
        return ans;
    }

    private boolean checkTurnPoint(double[] sample, int point, int window) {
        if (point < window || point + window > sample.length) {
            return false;
        }
        double baseLeft = StatUtils.mean(sample, point - window, window);
        double sigmaLeft = FastMath.sqrt(StatUtils.variance(sample, baseLeft, point - window, window));
        double baseRight = StatUtils.mean(sample, point, window);
        double sigmaRight = FastMath.sqrt(StatUtils.variance(sample, baseRight, point, window));
        return FastMath.abs(baseLeft - baseRight) > 3 * FastMath.min(sigmaLeft, sigmaRight);
    }

    private double[] extractSeason(double[] sample) {
        double[] season = new double[sample.length];
        for (int i = 0; i < sample.length; i++) {
            season[i] = extractSeason(sample, i);
        }
        return season;
    }

    private double extractSeason(double[] sample, int i) {
        if (i < periodLength) {
            return sample[i];
        }
        int periodNum = Math.min(i / periodLength, this.periodNum);
        ArrayList<Double> v = new ArrayList<>();
        ArrayList<Integer> diffT = new ArrayList<>();
        for (int j = 1; j <= periodNum; j++) {
            int c = i - j * periodLength;
            for (int k = Math.max(0, c - sideWidth); k <= Math.min(sample.length - 1, c + sideWidth); k++) {
                v.add(sample[k]);
                diffT.add(k - c);
            }
        }
        return Utils.bilateralFilter(diffT, v, sample[i], sideWidth, delta);
    }

    private double[] extractResidual(double[] sample, double[] trend, double[] season) {
        double[] residual = new double[sample.length];
        for (int i = 0; i < sample.length; i++) {
            residual[i] = sample[i] - season[i] - trend[i];
        }
        return residual;
    }

    public double[] getValue() {
        return value;
    }

    public double[] getTrend() {
        return trend;
    }

    public double[] getSeason() {
        return season;
    }

    public double[] getResidual() {
        return residual;
    }

    public int getSideWidth() {
        return sideWidth;
    }

    public double getDelta() {
        return delta;
    }

    public int getPeriodLength() {
        return periodLength;
    }

    public int getPeriodNum() {
        return periodNum;
    }

    public int getLastJump() {
        return lastJump;
    }
}
