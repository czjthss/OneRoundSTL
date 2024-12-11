import algorithm.utils.LDLT;

public class ExperimentEvaluation {
    private static final String INPUT_DIR = "/Users/chenzijie/Documents/GitHub/data/input/decomposition/";
    private static final String[] datasetFileList = {
            "power_5241600.csv",
            "voltage_22825440.csv",
    };

    private static void ldltAblation() {
        int dataSizeBase = 1000000, dataSize;
        double epsilon = 1e-8;
        double[][] timeCost = new double[3][10];
        long[][] spaceCost = new long[3][10];

        for (int dataIdx = 0; dataIdx < 5; dataIdx++) {
            dataSize = (dataIdx + 1) * dataSizeBase;

            ExpLDLT ldlt = new ExpLDLT(dataSize, epsilon, 1.0, 1.0);
            System.out.println(dataSize + "#######");
            for (int i = 0; i < 2; ++i) {
                ldlt.decompose(i);
                System.out.println(String.format("%.4f", 1e-9 * ldlt.getTimeCost()) + "," + ldlt.getSpaceCost());
                timeCost[i][dataIdx] = ldlt.getTimeCost();
                spaceCost[i][dataIdx] = ldlt.getSpaceCost();
            }
        }

        System.out.println();
        System.out.print("\n[");
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 10; ++j) {
                System.out.print(String.format("%.4f", 1e-9 * timeCost[i][j]) + ",");
            }
            System.out.println();
        }
        System.out.print("]");

        System.out.print("\n[");
        for (int i = 0; i < 3; ++i) {
            System.out.print("[");
            for (int j = 0; j < 10; ++j) {
                System.out.print(spaceCost[i][j] + ",");
            }
            System.out.println("]");
        }
        System.out.print("]");
    }

    private static void parameter() throws Exception {
        int period = 120;
        int dataSize = 2000;
        int initPeriodNum = 5;
        int errorNum = 1;
        double missingRateBase = 0.01;
        double missingRate = 0.00;
        String[][] rmse = new String[10][3];
        String[] time = new String[10];

        double epsilon = 1e-8, zeta = 1e-8, lambda = 1.0;
        double[] valueArray = new double[]{1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
        int MAX_TS_SIZE = 40000, MAX_PAGE_NUM = 100;
        int MAX_PAGE_SIZE = 720;

        for (int idx = 0; idx < 10; idx++) {
            missingRate = missingRateBase * idx;
//            epsilon = valueArray[idx];
//            zeta = valueArray[idx];

            Analysis analysis = LoadData.triangleWave(dataSize, period);
            analysis.set_init_num(initPeriodNum * period);
            double[] ts = analysis.get_ts();
            LoadData.addNan(ts, period, missingRate, 1);

            Algorithm.TSDB_OneRoundSTLModel(period, epsilon, zeta, lambda, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM, "flush");

            LDLT ldlt = new LDLT(ts.length, epsilon, 1.0, 1.0);

            rmse[idx][0] = analysis.get_trend_rmse(false);
            rmse[idx][1] = analysis.get_seasonal_rmse(false);
            rmse[idx][2] = analysis.get_residual_rmse(false);
            time[idx] = analysis.get_time_cost();
//            time[idx] = (ldlt.getConvergeSize() + 4) * 5 * 4;
        }

        System.out.println("[");
        for (int i = 0; i < 3; ++i) {
            System.out.print("[");
            for (int j = 0; j < 10; ++j) {
                System.out.print(rmse[j][i] + ",");
            }
            System.out.println("],");
        }
        System.out.println("]");

//        System.out.print("[");
//        for (int j = 0; j < 10; ++j) {
//            System.out.print(time[j] + ",");
//        }
//        System.out.println("]");

        System.out.print("[");
        for (int j = 0; j < 10; ++j) {
            System.out.print(time[j] + ",");
        }
        System.out.println("]");
    }

    public static void spaceCost() throws Exception {
        for (String dataset : datasetFileList) {
            String datasetName = dataset.split("_")[0];

            int dataSizeBase = datasetName.equals("power") ? 100000 : 1000000;

            double epsilon, thres;

            for (int dataSize = dataSizeBase; dataSize <= dataSizeBase * 10; dataSize += dataSizeBase) {
                Analysis analysis = LoadData.loadTimeSeriesData(INPUT_DIR + dataset, dataSize);
                double[] ts = analysis.get_ts();

                // epsilon
                thres = LoadData.calThreshold(ts);
                epsilon = thres;

                ExpLDLT ldlt = new ExpLDLT(dataSize, epsilon, 1.0, 1.0);
                ldlt.decompose(0);

                // sum
                int totalSpaceCost = (ldlt.getSpaceCost() + ts.length * 2 * 4);
                totalSpaceCost = datasetName.equals("power") ? totalSpaceCost / 1024 : totalSpaceCost / 1024 / 1024;
                System.out.print(totalSpaceCost + ",");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) throws Exception {
//        ldltAblation();
        spaceCost();
//        System.out.println(160000600 );
//        parameter();
    }
}


class ExpLDLT {
    private final int N;
    private final double epsilon;
    private double[][] a_start_lines;
    private double[][] a_mid_lines;
    private double[][] a_end_lines;
    private int spaceCost;
    private long timeCost;

    public ExpLDLT(int N, double epsilon, double lambda_t, double lambda_s) {
        this.N = N;
        this.epsilon = epsilon;

        this.constructA(lambda_t, lambda_s);
    }

    public void decompose(int type) {
        if (type == 0) {
            this.decomposeRS();
        } else if (type == 1) {
            this.decomposeS();
        } else {
            this.decomposeO();
        }
    }

    private void constructA(double lambda_t, double lambda_s) {
        double[][] start_lines1 = {{1, 1, 0, 0, 0}, {1, 1, 0, 0, 0}, {0, 0, 1, 1, 0}, {0, 0, 1, 1, 0}};
        double[][] start_lines2 = {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 1, 0}};
        multiplyMatrixByScalar(start_lines2, lambda_s);
        double[][] start_lines3 = {{1, 0, -2, 0, 1}, {0, 0, 0, 0, 0}, {-2, 0, 5, 0, -4}, {0, 0, 0, 0, 0}};
        multiplyMatrixByScalar(start_lines3, lambda_t);
        a_start_lines = addMatrices(addMatrices(start_lines1, start_lines2), start_lines3);

        // a_mid_lines and a_end_lines are computed here in a similar fashion
        double[][] mid_lines1 = {{0, 0, 0, 0, 1}, {0, 0, 0, 1, 1}};
        double[][] mid_lines2 = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}};
        multiplyMatrixByScalar(mid_lines2, lambda_s);
        double[][] mid_lines3 = {{1, 0, -4, 0, 6}, {0, 0, 0, 0, 0}};
        multiplyMatrixByScalar(mid_lines3, lambda_t);
        a_mid_lines = addMatrices(addMatrices(mid_lines1, mid_lines2), mid_lines3);

        double[][] end_lines1 = {{0, 0, 0, 0, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 1, 1}};
        double[][] end_lines2 = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}};
        multiplyMatrixByScalar(end_lines2, lambda_s);
        double[][] end_lines3 = {{1, 0, -4, 0, 5}, {0, 0, 0, 0, 0}, {1, 0, -2, 0, 1}, {0, 0, 0, 0, 0}};
        multiplyMatrixByScalar(end_lines3, lambda_t);
        a_end_lines = addMatrices(addMatrices(end_lines1, end_lines2), end_lines3);
    }

    private void multiplyMatrixByScalar(double[][] matrix, double scalar) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] *= scalar;
            }
        }
    }

    private double[][] addMatrices(double[][] m1, double[][] m2) {
        double[][] result = new double[m1.length][m1[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[0].length; j++) {
                result[i][j] = m1[i][j] + m2[i][j];
            }
        }
        return result;
    }

    private double queryA(int i, int j) {
        // Swapping if i < j to ensure i >= j
        if (i < j) {
            int temp = i;
            i = j;
            j = temp;
        }

        // Directly return 0 if the difference is larger than 4, as the matrix is banded
        if (Math.abs(i - j) > 4) {
            return 0.0;
        }

        // Depending on the indices, return the corresponding value from one of the lines arrays
        if (i < 4) {
            return a_start_lines[i][j];
        } else if (i < 2 * N - 4) {
            return a_mid_lines[i % 2][j - i + 4];
        } else {
            return a_end_lines[i - 2 * N + 4][j - i + 4];
        }
    }

    private void decomposeRS() {
        int converge_size = 2 * N;
        double[][] L = new double[2 * N][4];
        double[] D = new double[2 * N];
        double[][] C = new double[2][5];
        boolean converge = false;
        long startTime = System.nanoTime();

        for (int i = 0; i < 2 * N; i++) {
            if (converge && i < 2 * N - 4) {
//                D[i] = C[i % 2][4];
//                for (int j = i + 1; j < i + 5; j++) {
//                    L[j][4 - j + i] = C[i % 2][j - i - 1];
//                }
            } else {
                double sumDi = 0.0;
                for (int k = Math.max(0, i - 4); k < i; k++) {
                    sumDi += D[k] * L[i][k + 4 - i] * L[i][k + 4 - i];
                }
                D[i] = queryA(i, i) - sumDi;

                for (int j = i + 1; j < Math.min(2 * N, i + 5); j++) {
                    double sumLij = 0.0;
                    for (int k = Math.max(0, j - 4); k < i; k++) {
                        sumLij += L[j][k + 4 - j] * D[k] * L[i][k + 4 - i];
                    }
                    L[j][4 - j + i] = (queryA(j, i) - sumLij) / D[i];
                }

                if (i >= 4) {
                    double convergenceSum = 0.0;
                    for (int h : new int[]{i - 1, i}) {
                        convergenceSum += Math.abs(D[h] - D[h - 2]);
                        for (int k = 0; k < 4; k++) {
                            convergenceSum += Math.abs(L[h][k] - L[h - 2][k]);
                        }
                    }
                    if (convergenceSum < epsilon) {
                        converge = true;
                        converge_size = i;
                        for (int j = i + 1; j < i + 5; j++) {
                            C[i % 2][j - i - 1] = L[j][4 - j + i];
                            C[(i - 1) % 2][j - i - 1] = L[j - 1][4 - j + i];
                        }
                        C[i % 2][4] = D[i];
                        C[(i - 1) % 2][4] = D[i - 1];
                    }
                }
            }
        }

        this.timeCost = System.nanoTime() - startTime;
        this.spaceCost = (converge_size + 4) * 5 * 4;
    }

    private void decomposeS() {
        double[][] L = new double[2 * N][4];
        double[] D = new double[2 * N];
        long startTime = System.nanoTime();

        for (int i = 0; i < 2 * N; i++) {
            double sumDi = 0.0;
            for (int k = Math.max(0, i - 4); k < i; k++) {
                sumDi += D[k] * L[i][k + 4 - i] * L[i][k + 4 - i];
            }
            D[i] = queryA(i, i) - sumDi;

            for (int j = i + 1; j < Math.min(2 * N, i + 5); j++) {
                double sumLij = 0.0;
                for (int k = Math.max(0, j - 4); k < i; k++) {
                    sumLij += L[j][k + 4 - j] * D[k] * L[i][k + 4 - i];
                }
                L[j][4 - j + i] = (queryA(j, i) - sumLij) / D[i];
            }
        }

        this.timeCost = System.nanoTime() - startTime;
        this.spaceCost = 2 * N * 5 * 4;
    }

    private void decomposeO() {
        double[][] L = new double[2 * N][2 * N];
        double[][] D = new double[2 * N][2 * N];
        long startTime = System.nanoTime();

        for (int k = 0; k < 2 * N; k++) {
            L[k][k] = 1;
            double sumDi = 0.0;
            for (int i = 0; i < k; i++) {
                sumDi += D[i][i] * L[k][i] * L[k][i];
            }
            D[k][k] = queryA(k, k) - sumDi;
            for (int j = k + 1; j < 2 * N; j++) {
                L[k][j] = 0;
                double sumLij = 0.0;
                for (int i = 0; i < k; i++) {
                    sumLij += L[j][i] * D[i][i] * L[k][i];
                }
                L[j][k] = (queryA(j, k) - sumLij) / D[k][k];
            }
        }

        this.timeCost = System.nanoTime() - startTime;
        this.spaceCost = 2 * 2 * N * 2 * N * 4;
    }

    public int getSpaceCost() {
        return spaceCost;
    }

    public long getTimeCost() {
        return timeCost;
    }
}