package algorithm.BacktrackSTLUtils;

import algorithm.BacktrackSTLUtils.common.CircularQueue;

/**
 * @author Haoyu Wang
 * @date 2023/10/18
 */
public class KSigma {

    private double delta;
    private double kSigma;

    private CircularQueue<Double> data;
    private double sum = 0;
    private double squareSum = 0;
    private int count = 0;

    public KSigma(int windowSize, double delta, double kSigma) {
        this.data = new CircularQueue<>(windowSize);
        this.delta = delta;
        this.kSigma = kSigma;
    }

    public void add(double value) {
        if (data.isAtFullCapacity()) {
            sum -= data.get(0);
            squareSum -= data.get(0) * data.get(0);
            count--;
        }
        data.add(value);
        sum += value;
        squareSum += value * value;
        count++;
    }

    public void removeLast() {
        double v = data.removeLast();
        sum -= v;
        squareSum -= v * v;
        count--;
    }

    public boolean kSigmaDetect(double x) {
        double base = sum / count;
        double sigma = Math.sqrt(squareSum / count - base * base);
        sigma = Math.min(sigma, delta);
        return Math.abs(x - 0) > sigma * kSigma;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getkSigma() {
        return kSigma;
    }

    public void setkSigma(double kSigma) {
        this.kSigma = kSigma;
    }
}
