package algorithm.utils;

import java.util.Arrays;
import java.util.List;

public class FormatUtil {
    // convert double[] to List<Double>
    public static List<Double> doubleToList(double[] arr_double) {
        int size = arr_double.length;
        Double[] arr_Double = new Double[size];
        for (int i = 0; i < size; i++) {
            arr_Double[i] = arr_double[i];
        }
        return Arrays.asList(arr_Double);
    }
}
