package algorithm.BacktrackSTLUtils;

import java.util.ArrayList;

import algorithm.BacktrackSTLUtils.common.CircularQueue;
import algorithm.BacktrackSTLUtils.common.DataPoint;
import algorithm.BacktrackSTLUtils.common.Utils;
import org.apache.commons.math3.stat.StatUtils;

/**
 * @author Haoyu Wang
 * @date 2023/11/06
 */
public class BackTrackSTL {

    private final CircularQueue<DataPoint> data;

    private final MovingAverage trendEstimator;

    private final KSigma kSigmaDetector;

    private int periodNum;
    private int periodLength;
    private double delta = Double.NaN;
    private int sideWidth;
    private int jumpLag;
    private boolean outlierDetect;

    private int mutationCnt = 0;

    public BackTrackSTL(int periodNum, int periodLength, int sideWidth, int jumpLag, double kSigma,
                        boolean outlierDetect) {
        this.periodNum = periodNum;
        this.periodLength = periodLength;
        this.sideWidth = sideWidth;
        this.jumpLag = jumpLag;
        this.outlierDetect = outlierDetect;

        int windowSize = periodLength * (periodNum + 1);
        this.data = new CircularQueue<>(windowSize);
        this.trendEstimator = new MovingAverage(windowSize);
        this.kSigmaDetector = new KSigma(windowSize, delta, kSigma);
    }

    /**
     * Decompose a value into trend, season and residual
     *
     * @param value the original value
     * @return the decomposed data point
     */
    public DataPoint decompose(double value) {
        this.data.add(new DataPoint(value));
        update(this.data.size() - 1);
        return data.getLast();
    }

    /**
     * Initialize the model with the given values by an offline decomposition
     *
     * @param values the original values for initialization
     * @return the decomposed data points for initialization
     */
    public CircularQueue<DataPoint> initialize(double[] values) {
        if (Double.isNaN(delta)) {
            this.delta = Utils.estimatedStandardDeviation(values, periodLength, sideWidth);
            this.kSigmaDetector.setDelta(this.delta);
        }

        // Offline decomposition
        PeriodDecomposition decomposition = new PeriodDecomposition(sideWidth, delta, periodLength, periodNum);
        decomposition.decompose(values);
        double[] trend = decomposition.getTrend();
        double[] season = decomposition.getSeason();
        double[] residual = decomposition.getResidual();

        // Save the results
        for (int i = 0; i < values.length; i++) {
            DataPoint point = new DataPoint(values[i], trend[i], season[i]);
            this.data.add(point);
            this.trendEstimator.forceAdd(values[i]);
            this.kSigmaDetector.add(residual[i]);
        }
        return data;
    }

    private void jumpDetect(int id) {
        DataPoint point = data.get(id);
        double res = point.getValue() - point.getSeason() - point.getTrend();
        if (kSigmaDetector.kSigmaDetect(res)) {
            mutationCnt++;
        } else {
            mutationCnt = 0;
        }
        if (mutationCnt == jumpLag) {
            // backtrack if a jump is detected
            backtrack(data.size() - mutationCnt);
        }
    }

    private void backtrack(int jumpId) {
        double[] temp = new double[data.size() - jumpId];
        for (int i = jumpId; i < data.size(); i++) {
            temp[i - jumpId] = data.get(i).getValue() - data.get(i - periodLength).getSeason();
        }
        double trend = StatUtils.mean(temp);
        double gap = trend - data.get(jumpId - 1).getTrend();

        for (int i = jumpId; i < data.size(); i++) {
            trendEstimator.removeLast();
            kSigmaDetector.removeLast();
        }
        trendEstimator.compensate(gap);

        //Decompose each values
        for (int i = jumpId; i < data.size(); i++) {
            //Decompose
            data.get(i).setTrend(trend);
            double season = extractSeason(i);
            data.get(i).setSeason(season);
            data.get(i).setResidual(data.get(i).getValue() - trend - season);
            //Add values to trend estimator
            this.trendEstimator.forceAdd(data.get(i).getValue());
            this.kSigmaDetector.add(data.get(i).getResidual());
        }
    }

    private void update(int id) {
        DataPoint point = data.get(id);
        double trend = extractTrend(id);
        point.setTrend(trend);
        double season = extractSeason(id);
        point.setSeason(season);
        jumpDetect(id);
        point.setResidual(point.getValue() - point.getSeason() - point.getTrend());
        kSigmaDetector.add(point.getResidual());
    }

    private double extractTrend(int id) {
        if (outlierDetect) {
            double ref = data.get(id).getValue() - data.get(id - 1).getTrend();
            double value = getNearestValueInNeighborhood(id, ref);
            if (kSigmaDetector.kSigmaDetect(value - ref)) {
                trendEstimator.add(data.get(id).getValue());
            } else {
                trendEstimator.add(value + data.get(id - 1).getTrend());
            }
        } else {
            trendEstimator.add(data.get(id).getValue());
        }
        return trendEstimator.estimateTrend();
    }

    private double getNearestValueInNeighborhood(int i, double v) {
        int periodNum = Math.min(i / periodLength, this.periodNum);
        int side = sideWidth;
        double diff = Double.MAX_VALUE, value = 0;
        for (int j = 1; j <= periodNum; j++) {
            int c = i - j * periodLength;
            for (int k = Math.max(0, c - side); k <= c + side; k++) {
                if (Math.abs(data.get(k).getSeason() - v) < diff) {
                    diff = Math.abs(data.get(k).getSeason() - v);
                    value = data.get(k).getSeason();
                }
            }
        }
        return value;
    }

    private double extractSeason(int i) {
        int periodNum = Math.min(i / periodLength, this.periodNum);
        int side = sideWidth;
        ArrayList<Double> v = new ArrayList<>();
        ArrayList<Integer> diffT = new ArrayList<>();
        for (int j = 1; j <= periodNum; j++) {
            int c = i - j * periodLength;
            for (int k = Math.max(0, c - side); k <= c + side; k++) {
                v.add(data.get(k).getValue() - data.get(k).getTrend());
                diffT.add(k - c);
            }
        }
        return Utils.bilateralFilter(diffT, v, data.get(i).getValue() - data.get(i).getTrend(), side, delta);
    }

    public int getInitLength() {
        return this.periodLength * (this.periodNum + 1);
    }

}
