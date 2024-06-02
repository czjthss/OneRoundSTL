package algorithm.utils;

public class LDLT {
    private final int N;
    private final double epsilon;
    private double[][] a_start_lines;
    private double[][] a_mid_lines;
    private double[][] a_end_lines;
    private final double[][] L;
    private final double[] D;
    private int converge_size;

    public LDLT(int N, double epsilon, double lambda_t, double lambda_s) {
        this.N = N;
        this.L = new double[2 * N][4];
        this.D = new double[2 * N];
        this.epsilon = epsilon;

        this.constructA(lambda_t, lambda_s);
        this.decomposeA();
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

    private void decomposeA() {
        double[][] C = new double[2][5];
        boolean converge = false;

        for (int i = 0; i < 2 * N; i++) {
            if (converge && i < 2 * N - 4) {
                D[i] = C[i % 2][4];
                for (int j = i + 1; j < i + 5; j++) {
                    L[j][4 - j + i] = C[i % 2][j - i - 1];
                }
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

                if (i >= 4 && i < 2 * N - 4) {
                    double convergenceSum = 0.0;
                    for (int h : new int[]{i - 1, i}) {
                        convergenceSum += Math.abs(D[h] - D[h - 2]);
                        for (int k = 0; k < 4; k++) {
                            convergenceSum += Math.abs(L[h][k] - L[h - 2][k]);
                        }
                    }
//                    System.out.println(convergenceSum);
                    if (convergenceSum <= epsilon) {
                        converge = true;
                        converge_size = (i + 1) / 2;
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
    }

    public double queryL(int i, int j) {
        if (i == j) {
            return 1.0;
        } else if (j > i || i - j > 4) {
            return 0.0;
        } else {
            return L[i][j - i + 4];
        }
    }

    public double queryD(int i, int j) {
        if (i != j) {
            return 0.0;
        } else {
            return D[i];
        }
    }

    public double queryEndL(int i, int j, int N_new) {
        if (i == j) {
            return 1.0;
        } else if (j > i || i - j > 4) {
            return 0.0;
        } else {
            return L[i - 2 * N_new + 2 * N][j - i + 4];
        }
    }

    public double queryEndD(int i, int j, int N_new) {
        if (i != j) {
            return 0.0;
        } else {
            return D[i - 2 * N_new + 2 * N];
        }
    }

    public int getConvergeSize() {
        return converge_size;
    }
}
