import algorithm.*;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.Map;
import java.util.Scanner;

public class Experiment {

    public static double[] LoadData(String name, int size) throws Exception {
        Scanner sc = new Scanner(new FileReader(name));
        int idx = 0;
        double[] rtn = new double[size];
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            if (idx < size)
                rtn[idx++] = Double.parseDouble(line);
        }
        return rtn;
    }

    public static void OneShotSTLQuery(long[] time, double[] ts, int period, int shiftWindow, double lambda1, double lambda3, int maxIter, String f, String h, String g, int ratio, String name) throws Exception {
        System.out.println("OneShotSTL");

        long begin = System.nanoTime();
        OneShotSTL oneShotSTL = new OneShotSTL(time, ts, period, shiftWindow, lambda1, lambda3, maxIter, f, h, g, ratio);
        long end = System.nanoTime();

        Map<String, double[]> results = oneShotSTL.get_decompose_results();

        double timeCost = (1.0 * (end - begin) / 1000000000);
        System.out.println(timeCost);

        double[] seasonal = results.get("seasonal");
        double[] trend = results.get("trend");
        double[] residual = results.get("residual");

//        System.out.println("seasonal");
        FileWriter writer = new FileWriter("D:\\project\\python\\demo38_lsmdecomposition\\data\\seasonal" + name + ".txt");
        String file_string = "";
        for (int i = 5 * period; i < seasonal.length; ++i) {
            file_string += seasonal[i] + "\n";
        }
        writer.write(file_string);
        writer.flush();
        writer.close();

//        System.out.println("trend");
        writer = new FileWriter("D:\\project\\python\\demo38_lsmdecomposition\\data\\trend" + name + ".txt");
        file_string = "";
        for (int i = 5 * period; i < trend.length; ++i) {
            file_string += trend[i] + "\n";
        }
        writer.write(file_string);
        writer.flush();
        writer.close();

        System.out.println("residual");
        writer = new FileWriter("D:\\project\\python\\demo38_lsmdecomposition\\data\\residual" + name + ".txt");
        file_string = "";
        for (int i = 5 * period; i < residual.length; ++i) {
            file_string += residual[i] + "\n";
        }
        writer.write(file_string);
        writer.flush();
        writer.close();
    }

    public static void WindowSTLQuery(long[] time, double[] ts, int period, int slidingWindow) {
        System.out.println("WindowSTL");

        WindowSTL windowSTLAlg = new WindowSTL(time, ts, period, slidingWindow);
    }

    public static void main(String[] args) throws Exception {
        long[] time = new long[]{1};
        int period = 12;

        double[] ts = LoadData("D:\\project\\python\\demo38_lsmdecomposition\\results\\syn1.txt", 300);
//        OneShotSTLQuery(time, ts, period, 100, 0., 15., 20, "LS", "LS", "LS", 1, "1");
        ts = LoadData("D:\\project\\python\\demo38_lsmdecomposition\\results\\syn2.txt", 300);
        OneShotSTLQuery(time, ts, period, 100, 0., 4., 200, "LS", "LS", "LAD", 1, "2");


//        for (double i : ts)
//            System.out.print(i + ",");

//        WindowSTLQuery(time, ts, period, 10);
    }
}
