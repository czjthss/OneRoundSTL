package algorithm;

import java.util.Arrays;

import algorithm.BacktrackSTLUtils.common.CircularQueue;
import algorithm.BacktrackSTLUtils.common.DataPoint;
import algorithm.BacktrackSTLUtils.BackTrackSTL;


public class BacktrackSTL {

    public double[][] decompose(double[] ts, int period) throws Exception {
        return decompose(ts, period, 2, 5, 4, 6, true);
    }

    public double[][] decompose(double[] ts, int period, int periodNum, int sideWidth, int jumpLag, double kSigma, boolean outlierDetect) throws Exception {
        // output
        DataPoint dataPointsTemp;
        double[][] result = new double[3][ts.length];
        // model
        BackTrackSTL model = new BackTrackSTL(periodNum, period, sideWidth, jumpLag, kSigma, outlierDetect);
        int initLength = model.getInitLength();

        double[] initValues = Arrays.copyOf(ts, initLength);

        CircularQueue<DataPoint> initResult = model.initialize(initValues);
        for (int i = 0; i < initLength; i++) {
            dataPointsTemp = initResult.get(i).clone();
            result[0][i] = dataPointsTemp.getTrend();
            result[1][i] = dataPointsTemp.getSeason();
            result[2][i] = dataPointsTemp.getResidual();
        }

        for (int i = initLength; i < ts.length; i++) {
            dataPointsTemp = model.decompose(ts[i]);
            result[0][i] = dataPointsTemp.getTrend();
            result[1][i] = dataPointsTemp.getSeason();
            result[2][i] = dataPointsTemp.getResidual();
        }

        return result;
    }
}
