import algorithm.OneRoundSTL;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.Test;

import java.lang.reflect.Field;

import static org.junit.Assert.assertEquals;

public class TestOneRoundSTL {
    private void constructA(DMatrixRMaj A, int N, double lambda_t, double lambda_s) {
        // Set matrix B1
        DMatrixRMaj B1 = new DMatrixRMaj(N, 2 * N);
        for (int row = 0; row < N; row++) {
            B1.set(row, 2 * row, 1);
            B1.set(row, 2 * row + 1, 1);
        }

        // Set matrix B2
        DMatrixRMaj B2 = new DMatrixRMaj(N, 2 * N);
        for (int row = 0; row < N; row++) {
            B2.set(row, 2 * row + 1, 1);
        }

        // Set matrix B4
        DMatrixRMaj B4 = new DMatrixRMaj(N - 2, 2 * N);
        for (int row = 0; row < N - 2; row++) {
            B4.set(row, 2 * row, 1);
            B4.set(row, 2 * row + 2, -2);
            B4.set(row, 2 * row + 4, 1);
        }

        // Compute matrix A1 = B1.transpose * B1
        DMatrixRMaj A1 = new DMatrixRMaj(2 * N, 2 * N);
        CommonOps_DDRM.multTransA(B1, B1, A1);

        // Compute A2 = lambda_s * B2.transpose * B2
        DMatrixRMaj A2 = new DMatrixRMaj(2 * N, 2 * N);
        CommonOps_DDRM.multTransA(B2, B2, A2);
        CommonOps_DDRM.scale(lambda_s, A2);

        // Compute A4 = lambda_t * B4.transpose * B4
        DMatrixRMaj A4 = new DMatrixRMaj(2 * N, 2 * N);
        CommonOps_DDRM.multTransA(B4, B4, A4);
        CommonOps_DDRM.scale(lambda_t, A4);

        // Compute A = A1 + A2 + A4
        CommonOps_DDRM.addEquals(A, A1);
        CommonOps_DDRM.addEquals(A, A2);
        CommonOps_DDRM.addEquals(A, A4);
    }

    @Test
    public void testTSDBSTL() throws Exception {
        int period = 12, Qsize = 800, initPeriodNum = 5;
        double epsilon = 1e-8, tau = 1e-8, lambda = 1.0;
        Analysis analysis = LoadData.squareWave(Qsize, period);
        double[] ts = analysis.get_ts();

        OneRoundSTL oneRoundSTL = new OneRoundSTL();
        double[][] components = oneRoundSTL.decompose(analysis.get_ts(), period, initPeriodNum, epsilon, tau, lambda);

        // record
        double[] trend = components[0];
        double[] seasonal = components[1];

        double[] y_proposed = new double[ts.length * 2];
        for (int i = 0; i < ts.length; ++i) {
            y_proposed[2 * i] = trend[i];
            y_proposed[2 * i + 1] = seasonal[i];
        }

        // ##########################################
        // construct A
        DMatrixRMaj A = new DMatrixRMaj(2 * ts.length, 2 * ts.length);
        constructA(A, ts.length, lambda, 1.0);

        // Access the private variable using reflection
        Field privateField = OneRoundSTL.class.getDeclaredField("v");
        privateField.setAccessible(true);
        // Replace "SomeType" with the actual type of your private variable
        double[] testV = (double[]) privateField.get(oneRoundSTL);

        // generate new b
        double[] b = new double[ts.length * 2];
        for (int i = 0; i < ts.length; i++) {
            b[i * 2] = ts[i];
            b[i * 2 + 1] = ts[i] + testV[i % period];
        }
        DMatrixRMaj B = new DMatrixRMaj(b);

        // Create a solver that can decompose matrix 'A'
        DMatrixRMaj y_solver = new DMatrixRMaj(A.numCols, B.numCols);
        CommonOps_DDRM.solve(A, B, y_solver);

        for (int i = 0; i < y_proposed.length - 4; ++i) {
//            System.out.println(i + " " + y_solver.get(i, 0) + " " + y_proposed[i]);
            assertEquals(y_solver.get(i, 0), y_proposed[i], epsilon);
        }
    }
}
