package algorithm;

import java.util.*;


public class OnlineSTL {
    private final double gamma;
    private int initSize;

    public OnlineSTL() {
        this.gamma = 0.3;
    }

    public void initialize(int initSize) {
        this.initSize = initSize;
    }

    public Map<String, ArrayList<Double>> decompose(List<Double> ts, int period) {
        ArrayList<Double> trend = new ArrayList<>();
        ArrayList<Double> seasonal = new ArrayList<>();
        ArrayList<Double> residual = new ArrayList<>();
        Map<String, ArrayList<Double>> result = new HashMap<>();

        double num, rate, rateSum;
        for (int i = initSize; i < ts.size(); i++) {
            num = 0.0;
            rateSum = 0.0;
            for (int j = Math.max(0, i - period); j < Math.min(i + period, ts.size()); j++) {
                rate = Math.pow((1 - Math.pow(Math.abs((double) i - j) / period, 3)), 3);
                num += rate * ts.get(j);
                rateSum += rate;
            }
            trend.add(num / rateSum);
        }

        for (int i = initSize; i < ts.size(); i++) {
            if (i - initSize < period) {
                seasonal.add(ts.get(i) - trend.get(i - initSize));
            } else {
                seasonal.add(gamma * (ts.get(i) - trend.get(i - initSize)) + (1 - gamma) * seasonal.get(i - period - initSize));
            }
        }

        for (int i = initSize; i < ts.size(); i++) {
            residual.add(ts.get(i) - seasonal.get(i - initSize) - trend.get(i - initSize));
        }

        result.put("trend", trend);
        result.put("seasonal", seasonal);
        result.put("residual", residual);

        return result;
    }
}
