import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;
import org.junit.Test;

import java.util.Arrays;

public class TestOptimalSolution {
    @Test
    public void testArraysCopy() {
        double[] ts = new double[10000];
        long start = System.nanoTime();
        for (int i = 0; i < 100; ++i) {
            Arrays.fill(ts, 1000, 8000, 1.0);
        }
        System.out.println(System.nanoTime() - start);

        start = System.nanoTime();
        for (int i = 0; i < 100; ++i) {
            for (int j = 1000; j < 8000; ++j) {
                ts[j] = 1.0;
            }
        }
        System.out.println(System.nanoTime() - start);
    }

    @Test
    public void testSTL() {
        int period = 3;
        double[] ts = {1., 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
        // stl decomposition
        SeasonalTrendLoess.Builder stlBuilder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess stl = stlBuilder.setPeriodic().setPeriodLength(period).buildSmoother(ts);
        SeasonalTrendLoess.Decomposition stlDecomposition = stl.decompose();
        double[] seasonal_array = stlDecomposition.getSeasonal();
        double[] trend_array = stlDecomposition.getTrend();

        for (int i = 0; i < seasonal_array.length; ++i) {
            System.out.print(seasonal_array[i] + " ");
        }
    }
}
