package algorithm.utils;

public class CleanSTLUtil {
    private final double[] td;
    private final int period;

    private final double[] seasonal;
    private final double[] trend;
    private final double[] residual;

    public CleanSTLUtil(double[] td, int period) throws Exception {
        this.td = td;
        this.period = period;

        this.seasonal = new double[td.length];
        this.trend = new double[td.length];
        this.residual = new double[td.length];

        this.decompose();
    }

    public CleanSTLUtil(double[] td, int period, String type) throws Exception {
        this.td = td;
        this.period = period;

        this.seasonal = new double[td.length];
        this.trend = new double[td.length];
        this.residual = new double[td.length];

        if (type.equals("median")) {
            this.decompose();
        } else {
            this.decomposeMean();
        }
    }

    private void decomposeMean() throws Exception {
        if (period > td.length)
            throw new Exception("Error: Period exceed the size of time series!");

        // constant
        int interval = period / 2;
        int size = td.length;
        double sumCur = 0;
        // structure
        double[] de_trend = new double[size];

        // step 1: trend
        if (period % 2 == 1) {
            throw new Exception("Period must be even.");
        } else {
            // initial
            for (int i = 0; i < period - 1; ++i) sumCur += td[i];
            // moving median
            for (int i = period - 1; i < size; ++i) {
                sumCur += td[i];
                trend[i - interval + 1] = sumCur / period;
                sumCur -= td[i - period + 1];
            }
        }

        // trend extension
        constant_ext();

        // step 2: de-trend
        for (int i = 0; i < size; ++i)
            de_trend[i] = td[i] - trend[i];

        // step 3: seasonal
        for (int i = 0; i < period; ++i) {
            sumCur = 0;
            // in each cycle
            for (int j = 0; j < size / period; ++j)
                sumCur += de_trend[j * period + i];

            if (i < size % period) {
                sumCur += de_trend[i + (size / period) * period];
                seasonal[i] = sumCur / ((double) size / period + 1);
            } else {
                seasonal[i] = sumCur / ((double) size / period);
            }
        }

        // de-median
        sumCur = 0;
        for (int i = 0; i < period; i++) {
            sumCur += seasonal[i];
        }
        // TODO: mean?
        double mean_s = sumCur / period;
        for (int i = 0; i < period; ++i)
            seasonal[i] = seasonal[i] - mean_s;
        for (int i = period; i < seasonal.length; ++i) {
            seasonal[i] = seasonal[i % period];
        }


        // step 3: residual
        for (int i = 0; i < size; ++i)
            residual[i] = de_trend[i] - seasonal[i];
    }

    private void decompose() throws Exception {
        if (period > td.length)
            throw new Exception("Error: Period exceed the size of time series!");

        // constant
        int interval = period / 2;
        int size = td.length;
        // structure
        double[] de_trend = new double[size];
        MovingMedian movingMedian = new MovingMedian(period);

        // step 1: trend
        if (period % 2 == 1) {
            throw new Exception("Period must be even.");
        } else {
            // initial
            for (int i = 0; i < period - 1; ++i) movingMedian.update(td[i]);
            // moving median
            for (int i = period - 1; i < size; ++i) {
                movingMedian.update(td[i]);
                trend[i - interval + 1] = movingMedian.getMedian();
            }
        }

        // trend extension
        constant_ext();

        // step 2: de-trend
        for (int i = 0; i < size; ++i)
            de_trend[i] = td[i] - trend[i];

        // step 3: seasonal
        double[] cal_median = new double[Math.max(period, size / period + 1)];
        for (int i = 0; i < period; ++i) {
            // in each cycle
            for (int j = 0; j < size / period; ++j)
                cal_median[j] = de_trend[j * period + i];

            if (i < size % period) {
                cal_median[size / period] = de_trend[i + (size / period) * period];
                seasonal[i] = LinearMedian.getMedian(cal_median, size / period + 1);
            } else {
                seasonal[i] = LinearMedian.getMedian(cal_median, size / period);
            }
        }

        // de-median
        for (int i = 0; i < period; i++) {
            cal_median[i] = seasonal[i];
        }
        // TODO: mean?
        double median_s = LinearMedian.getMedian(cal_median, period);
        for (int i = 0; i < period; ++i)
            seasonal[i] = seasonal[i] - median_s;
        for (int i = period; i < seasonal.length; ++i) {
            seasonal[i] = seasonal[i % period];
        }


        // step 3: residual
        for (int i = 0; i < size; ++i)
            residual[i] = de_trend[i] - seasonal[i];
    }

    private void constant_ext() {
        int interval = period / 2;
        for (int i = interval; i > 0; --i)
            trend[i - 1] = trend[i];
        for (int i = trend.length - interval - 1; i < trend.length - 1; ++i)
            trend[i + 1] = trend[i];
    }

    private void ar_ext() {
        int interval = period / 2;
        int end = trend.length - interval - 1;

        double a = 0.0, b = 0.0, d = trend.length - 2 * interval - 1, tmp;
        for (int i = interval; i < end; ++i) {
            b -= trend[i];
            a += trend[i] * trend[i];
        }
        tmp = a * d - b * b;
        a /= tmp;
        b /= tmp;
        d /= tmp;

        double sigma = 0.0, a1 = 0.0;
        for (int i = interval; i < end; ++i) {
            sigma += (a + b * trend[i]) * trend[i + 1];
            a1 += (b + d * trend[i]) * trend[i + 1];
        }

        // extend
        for (int i = interval; i > 0; --i)
            trend[i - 1] = (trend[i] - sigma) / a1;
        for (int i = trend.length - interval - 1; i < trend.length - 1; ++i)
            trend[i + 1] = a1 * trend[i] + sigma;
    }

    public double[] getSeasonal() {
        return seasonal;
    }

    public double[] getTrend() {
        return trend;
    }

    public double[] getResidual() {
        return residual;
    }
}

