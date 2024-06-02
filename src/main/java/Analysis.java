public class Analysis {
    private final double[] ts;
    private final double[] gt_trend, gt_seasonal, gt_residual;
    private double[] trend, seasonal, residual;
    private double time_cost;
    private int init_num = 0;

    public Analysis(double[] ts) {
        this.ts = ts;
        this.gt_trend = new double[ts.length];
        this.gt_seasonal = new double[ts.length];
        this.gt_residual = new double[ts.length];
    }


    public Analysis(double[] ts, double[] trend, double[] seasonal, double[] residual) {
        this.ts = ts;
        this.gt_trend = trend;
        this.gt_seasonal = seasonal;
        this.gt_residual = residual;
    }

    public void set_trend(double[] trend) {
        this.trend = trend;
    }

    public void set_seasonal(double[] seasonal) {
        this.seasonal = seasonal;
    }

    public void set_residual(double[] residual) {
        this.residual = residual;
    }

    public void set_time_cost(double time_cost) {
        this.time_cost = time_cost;
    }

    public void set_init_num(int init_num) {
        this.init_num = init_num;
    }

    public double[] get_ts() {
        return this.ts;
    }

    public String get_time_cost() {
        return String.format("%.3f", 1e-9 * this.time_cost);
    }


    private double getRMSE(double[] array, double[] gt_array, boolean init) {
        double sum = 0.0, gap, mean;
        int arraySize = 0;
        for (int i = init ? init_num : 0; i < array.length; i++) {
            if (!Double.isNaN(gt_array[i])) {
                gap = init ? array[i - init_num] - gt_array[i] : array[i] - gt_array[i];
                sum += gap * gap;
                arraySize++;
            }
        }
        mean = sum / arraySize;
        return Math.sqrt(mean);
    }

    public String get_trend_rmse(boolean init) {
        return String.format("%.3f", getRMSE(trend, gt_trend, init));
    }

    public String get_seasonal_rmse(boolean init) {
        return String.format("%.3f", getRMSE(seasonal, gt_seasonal, init));
    }

    public String get_residual_rmse(boolean init) {
        return String.format("%.3f", getRMSE(residual, gt_residual, init));
    }

    public static void main(String[] args) {
        double a = 123;
        String arr = String.valueOf(a);
        arr = arr.replace(".", "");
        System.out.println(arr.length());
    }
}
