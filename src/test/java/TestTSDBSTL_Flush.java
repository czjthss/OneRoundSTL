import algorithm.OneRoundSTL_Flush;
import algorithm.utils.LDLT;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.decomposition.chol.CholeskyDecompositionLDL_DDRM;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

import java.lang.reflect.Field;

public class TestTSDBSTL_Flush {
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
    public void testColdStart() throws Exception {
        int Qsize = 8000, period = 12;
        Analysis analysis = LoadData.triangleWave(Qsize, period);
        // Use streams to convert to a primitive array
        double[] ts = analysis.get_ts();

        OneRoundSTL_Flush lsmStl = new OneRoundSTL_Flush(period, new LDLT(Qsize, 1e-8, 1.0, 1.0));
        lsmStl.preCalculate(ts);

        // Access the private variable using reflection
        Field privateField = OneRoundSTL_Flush.class.getDeclaredField("v");
        privateField.setAccessible(true);

        // Replace "SomeType" with the actual type of your private variable
        double[] testV = (double[]) privateField.get(lsmStl);

        for (int i = 0; i < testV.length; ++i) {
            System.out.print(testV[i] + " ");
        }
    }

    @Test
    public void testDecompBasedImpute() {
        double epsilon = 1e-18, lambda = 1.0;
        int MAX_PAGE_SIZE = 800, Qsize = 800, period = 12;
        Analysis analysis = LoadData.triangleWave(Qsize, period);

        // Use streams to convert to a primitive array
        double[] ts = analysis.get_ts();

        int idx = 350;
        System.out.println(ts[idx - 1] + " " + ts[idx + 1]);
        System.out.println(ts[idx]);
        ts[idx] = Double.NaN;

        // TSDBSTL_Flush
        LDLT ldlt = new LDLT(MAX_PAGE_SIZE, epsilon, lambda, 1.0);
        OneRoundSTL_Flush tsdbstlFlush = new OneRoundSTL_Flush(period, ldlt);

        // preCalculate
        tsdbstlFlush.preCalculate(ts);
        double[] intermediateTrend = tsdbstlFlush.getIntermediateTrend();
        double[] intermediateSeasonal = tsdbstlFlush.getIntermediateSeasonal();

        System.out.println(ts[idx]);
    }

    @Test
    public void testTSDBSTLFlush() throws Exception {
        double epsilon = 1e-18, delta_equal = 1e-8, lambda = 1.0;
        int MAX_TS_SIZE = 10000;
        int MAX_PAGE_SIZE = 190, Qsize = 800, period = 12;
        Analysis analysis = LoadData.triangleWave(Qsize, period);

        // Use streams to convert to a primitive array
        double[] ts = analysis.get_ts();

        // TSDBSTL_Flush
        LDLT ldlt = new LDLT(MAX_TS_SIZE, epsilon, lambda, 1.0);
        OneRoundSTL_Flush tsdbstlFlush = new OneRoundSTL_Flush(period, ldlt);

        // preCalculate
        tsdbstlFlush.preCalculate(ts);
        double[] intermediateTrend = tsdbstlFlush.getIntermediateTrend();
        double[] intermediateSeasonal = tsdbstlFlush.getIntermediateSeasonal();
        double[] z_proposed = new double[2 * ts.length];
        for (int i = 0; i < ts.length; ++i) {
            z_proposed[2 * i] = intermediateTrend[i];
            z_proposed[2 * i + 1] = intermediateSeasonal[i];
        }

        // get v: Access the private variable using reflection
        Field privateField = OneRoundSTL_Flush.class.getDeclaredField("v");
        privateField.setAccessible(true);

        // Replace "SomeType" with the actual type of your private variable
        double[] v = (double[]) privateField.get(tsdbstlFlush);

        // ##########################################
        // construct A
        DMatrixRMaj A = new DMatrixRMaj(2 * ts.length, 2 * ts.length);
        constructA(A, ts.length, lambda, 1.0);

        // Perform LDLT decomposition
        CholeskyDecompositionLDL_DDRM ldlt_ejml = new CholeskyDecompositionLDL_DDRM();
        if (!ldlt_ejml.decompose(A)) {
            throw new RuntimeException("Decomposition failed!");
        }

        // Extract the L and D matrices
        DMatrixRMaj L = new DMatrixRMaj(A.numRows, A.numCols);
        DMatrixRMaj D = new DMatrixRMaj(A.numRows, A.numCols);
        ldlt_ejml.getL(L);
        ldlt_ejml.getD(D);

        // generate new b
        double[] b = new double[ts.length * 2];
        for (int i = 0; i < ts.length; i++) {
            b[i * 2] = ts[i];
            b[i * 2 + 1] = ts[i] + v[i % period];
        }
        DMatrixRMaj B = new DMatrixRMaj(b);

        // Create a solver that can decompose matrix 'A'
        DMatrixRMaj z_solver = new DMatrixRMaj(A.numCols, B.numCols);
        CommonOps_DDRM.solve(L, B, z_solver);

        for (int i = 0; i < b.length - 4; ++i) {
            assertEquals(z_solver.get(i, 0), z_proposed[i], delta_equal);
        }
    }
}
