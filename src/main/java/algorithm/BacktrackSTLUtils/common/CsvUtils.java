package algorithm.BacktrackSTLUtils.common;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * @author Haoyu Wang
 * @date 2023/10/11
 */
public class CsvUtils {

    public static List<DataPoint> fromCsv(String path) throws Exception {
        List<DataPoint> list = new ArrayList<>();
        Scanner sc = new Scanner(new File(path));
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            String[] split = line.split(",");
            DataPoint dataPoint = new DataPoint(Double.parseDouble(split[1]), Double.parseDouble(split[2]),
                Double.parseDouble(split[3]));
            list.add(dataPoint);
        }
        return list;
    }

    public static void toCsv(List<DataPoint> result, String path) throws Exception {
        if (path == null) {
            return;
        }
        PrintWriter writer = new PrintWriter(path);
        for (int i = 0; i < result.size(); i++) {
            DataPoint dataPoint = result.get(i);
            writer.print(i);
            writer.print(",");
            writer.print(dataPoint.getValue());
            writer.print(",");
            writer.print(dataPoint.getTrend());
            writer.print(",");
            writer.print(dataPoint.getSeason());
            writer.print(",");
            writer.print(dataPoint.getResidual());
            writer.println();
        }
        writer.close();
    }
}
