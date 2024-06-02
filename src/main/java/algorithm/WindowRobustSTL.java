package algorithm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WindowRobustSTL {
    private int period;
    private int slidingWindow;
    private double[] slidingWindowY;

    public WindowRobustSTL(int period, int slidingWindow) {
        this.period = period;
        this.slidingWindow = slidingWindow;
    }

    public void initialize(List<Double> y) {
        if (y.size() > slidingWindow) {
            y = y.subList(y.size() - slidingWindow, y.size());
        }
        slidingWindowY = convertDouble(y.toArray());
    }

    public Map<String, Double> decompose(double yNew) throws Exception {
        double[] slidingWindowYNew = new double[slidingWindow];
        System.arraycopy(slidingWindowY, 1, slidingWindowYNew, 0, slidingWindow - 1);
        slidingWindowYNew[slidingWindow - 1] = yNew;
        slidingWindowY = slidingWindowYNew;
        Map<String, Double> result = new HashMap<>();

        // write
        StringBuilder string = new StringBuilder();
        for (double value : slidingWindowY) {
            string.append(value);
            string.append("\n");
        }
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/input.txt", false));
        bufferedWriter.write(string.toString());
        bufferedWriter.close();
        // python shell
        String[] cmd = {
                "/Users/chenzijie/anaconda3/envs/dq/bin/python",
                "/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/robuststl.py",
        };
        Process process = Runtime.getRuntime().exec(cmd);
        process.waitFor();
        String line;

        // read
        ArrayList<Double> trend = new ArrayList<>();
        ArrayList<Double> seasonal = new ArrayList<>();
        ArrayList<Double> residual = new ArrayList<>();
        BufferedReader bufferedReader = new BufferedReader(new FileReader("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/trend.txt"));
        while ((line = bufferedReader.readLine()) != null) {
            trend.add(Double.parseDouble(line));
        }
        bufferedReader.close();
        bufferedReader = new BufferedReader(new FileReader("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/seasonal.txt"));
        while ((line = bufferedReader.readLine()) != null) {
            seasonal.add(Double.parseDouble(line));
        }
        bufferedReader.close();
        bufferedReader = new BufferedReader(new FileReader("/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTL/residual.txt"));
        while ((line = bufferedReader.readLine()) != null) {
            residual.add(Double.parseDouble(line));
        }
        bufferedReader.close();

        result.put("trend", trend.get(slidingWindow - 1));
        result.put("seasonal", seasonal.get(slidingWindow - 1));
        result.put("residual", residual.get(slidingWindow - 1));
        return result;
    }

    private double[] convertDouble(Object[] x) {
        double[] x_new = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            x_new[i] = (double) x[i];
        }
        return x_new;
    }
}
