package algorithm;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RobustSTL {
    public final String FILE_DIR = "/Users/chenzijie/Documents/GitHub/std-benchmark/src/main/java/algorithm/RobustSTLUtils/";

    public Map<String, ArrayList<Double>> decompose(List<Double> ts, int period) throws Exception {
        ArrayList<Double> trend = new ArrayList<>();
        ArrayList<Double> seasonal = new ArrayList<>();
        ArrayList<Double> residual = new ArrayList<>();
        Map<String, ArrayList<Double>> result = new HashMap<>();

        // write
        StringBuilder string = new StringBuilder();
        for (double value : ts) {
            string.append(value);
            string.append("\n");
        }
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(FILE_DIR + "input.txt", false));
        bufferedWriter.write(string.toString());
        bufferedWriter.close();

        // python shell
        String[] cmd = {
                "/Users/chenzijie/anaconda3/envs/dq/bin/python",
                FILE_DIR + "robuststl.py",
                String.valueOf(period),
        };
        Process process = Runtime.getRuntime().exec(cmd);
        process.waitFor();

        if (process.exitValue() != 0){
            System.out.println("RobustSTL Complete UnSuccessfully.");
        }

        String line;

//        BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
//        BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
//        BufferedReader input = new BufferedReader(new InputStreamReader(process.getInputStream()));
//        while ((line = input.readLine()) != null) {
//            System.out.println(line);
//        }
//        input.close();

        // read
        BufferedReader bufferedReader = new BufferedReader(new FileReader(FILE_DIR + "trend.txt"));
        while ((line = bufferedReader.readLine()) != null) {
            trend.add(Double.parseDouble(line));
        }
        bufferedReader.close();

        bufferedReader = new BufferedReader(new FileReader(FILE_DIR + "seasonal.txt"));
        while ((line = bufferedReader.readLine()) != null) {
            seasonal.add(Double.parseDouble(line));
        }
        bufferedReader.close();

        bufferedReader = new BufferedReader(new FileReader(FILE_DIR + "residual.txt"));
        while ((line = bufferedReader.readLine()) != null) {
            residual.add(Double.parseDouble(line));
        }
        bufferedReader.close();

        // record
        result.put("trend", trend);
        result.put("seasonal", seasonal);
        result.put("residual", residual);

        return result;
    }
}