package algorithm.BacktrackSTLUtils;

import java.util.LinkedList;

import algorithm.BacktrackSTLUtils.common.CircularQueue;

/**
 * @author Haoyu Wang
 * @date 2023/10/08
 */
public class MovingAverage {

    private CircularQueue<Double> data;
    private double sum = 0;
    private int count = 0;
    private LinkedList<CompensationTerm> compensation = new LinkedList<>();

    public MovingAverage(int windowSize) {
        data = new CircularQueue<>(windowSize);
    }

    public void add(double value) {
        if (data.isAtFullCapacity()) {
            sum -= data.get(0);
            count--;
            for (CompensationTerm term : compensation) {
                term.decease();
            }
            if (!compensation.isEmpty() && compensation.getFirst().getCount() == 0) {
                compensation.removeFirst();
            }
        }
        data.add(value);
        sum += value;
        count++;
    }

    public void forceAdd(double value) {
        add(value);
    }

    public double estimateTrend() {
        double totalSum = sum;
        for (CompensationTerm term : compensation) {
            totalSum += term.getTotal();
        }
        return totalSum / count;
    }

    public void clear() {
        data.clear();
        sum = 0;
        count = 0;
    }

    public void removeLast() {
        Double v = this.data.removeLast();
        sum -= v;
        count--;
    }

    public void compensate(double gap) {
        CompensationTerm term = new CompensationTerm(gap, count);
        compensation.add(term);
    }

    protected static class CompensationTerm {
        private double gap;
        private int count;

        public CompensationTerm(double gap, int count) {
            this.gap = gap;
            this.count = count;
        }

        public double getTotal() {
            return gap * count;
        }

        public void decease() {
            count--;
        }

        public double getGap() {
            return gap;
        }

        public int getCount() {
            return count;
        }

    }
}
