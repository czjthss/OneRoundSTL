package algorithm.utils;

import java.util.List;
import java.util.ArrayList;


public class Utils {
    public static double[] convertListToArray(List<Double> list) {
        double[] array = new double[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i); // Unboxing the Double to double
        }
        return array;
    }

    public static boolean[] convertBooleanListToArray(List<Boolean> list) {
        boolean[] array = new boolean[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i); // Unboxing the Double to double
        }
        return array;
    }

    public static List<Double> convertArrayToList(double[] array) {
        List<Double> list = new ArrayList<>();
        for (double value : array) {
            list.add(value); // Autoboxing the double to Double
        }
        return list;
    }


    public static void main(String[] args) {
        List<Double> doubleList = List.of(1.0, 2.0, 3.0, 4.0);
        double[] doubleArray = convertListToArray(doubleList);

        // Just for demonstration, print out the array
        for (double value : doubleArray) {
            System.out.println(value);
        }

        double[] doubleArray2 = {1.0, 2.0, 3.0, 4.0};
        List<Double> doubleList2 = convertArrayToList(doubleArray2);

        // Just for demonstration, print out the list
        for (Double value : doubleList2) {
            System.out.println(value);
        }
    }
}
