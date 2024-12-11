package algorithm.BacktrackSTLUtils.common;

/**
 * @author Haoyu Wang
 * @date 2022/8/4
 */
public class DataPoint implements Cloneable {

    private double value;

    private double trend;

    private double season;

    private double residual;

    public DataPoint(double value) {
        this.value = value;
    }

    public DataPoint(double value, double trend, double season) {
        this.value = value;
        this.trend = trend;
        this.season = season;
        this.residual = value - trend - season;
    }

    @Override
    public DataPoint clone() {
        try {
            return (DataPoint) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getTrend() {
        return trend;
    }

    public void setTrend(double trend) {
        this.trend = trend;
    }

    public double getSeason() {
        return season;
    }

    public void setSeason(double season) {
        this.season = season;
    }

    public double getResidual() {
        return residual;
    }

    public void setResidual(double residual) {
        this.residual = residual;
    }
}
