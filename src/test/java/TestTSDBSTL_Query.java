import algorithm.OneRoundSTL_Flush;
import algorithm.OneRoundSTL_Query;
import algorithm.utils.LDLT;
import algorithm.utils.TSDBSTL_Concatenation;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestTSDBSTL_Query {
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
    public void testTSDBSTLQuery() throws Exception {
        int MAX_TS_SIZE = 10000;
        int MAX_PAGE_SIZE = 240;
        int MAX_PAGE_NUM = 10;
        int period = 12, Qsize = 800;
        double epsilon = 1e-8, zeta = 1e-8, lambda = 1.0;
        Analysis analysis = LoadData.squareWave(Qsize, period);
        double[] ts = analysis.get_ts();

        // Check if the time series size exceeds the pre-defined limits.
        if (analysis.get_ts().length > MAX_PAGE_NUM * MAX_PAGE_SIZE) {
            throw new Exception("ts exceeds the limit size.");
        }

        // Three-dimensional array for time series database, with pages, columns, and rows.
        double[][][] TSDB = new double[MAX_PAGE_NUM][3][];  // page * column * row

        // Conversion of the time series list to a primitive double array.
        double[] ts_array = analysis.get_ts();

        // Create pages for the time series, splitting the array into chunks of MAX_PAGE_SIZE.
        double[][] ts_pages = new double[ts_array.length / MAX_PAGE_SIZE + 1][MAX_PAGE_SIZE];
        for (int i = 0; i < ts_array.length; ++i) {
            ts_pages[i / MAX_PAGE_SIZE][i % MAX_PAGE_SIZE] = ts_array[i];
        }

        LDLT ldlt = new LDLT(MAX_TS_SIZE, epsilon, lambda, 1.0);
        OneRoundSTL_Flush tsdbstlFlush = new OneRoundSTL_Flush(period, ldlt);

        long start = System.nanoTime();
        for (int pageIdx = 0; pageIdx < ts_pages.length; ++pageIdx) {
            // Pre-calculate trend and seasonal intermediate results for current page of series.
            tsdbstlFlush.preCalculate(ts_pages[pageIdx]);
            // Store the original time series, intermediate trend and seasonal components in a 3D array.
            TSDB[pageIdx][0] = ts_pages[pageIdx];
            TSDB[pageIdx][1] = tsdbstlFlush.getIntermediateTrend();
            TSDB[pageIdx][2] = tsdbstlFlush.getIntermediateSeasonal();
        }
        TSDBSTL_Concatenation tsdbstlConcatenation = new TSDBSTL_Concatenation(TSDB, period, tsdbstlFlush.getV(), ldlt);
        tsdbstlConcatenation.concat(zeta);

        OneRoundSTL_Query tsdbstlQuery = new OneRoundSTL_Query(TSDB, period, tsdbstlFlush.getV(), ldlt);
        tsdbstlQuery.decompose(0, ts.length, 1e-8);
        double[] trend = tsdbstlQuery.getTrend();
        double[] seasonal = tsdbstlQuery.getSeasonal();

        double[] y_proposed = new double[ts.length * 2];
        for (int i = 0; i < ts.length; ++i) {
            y_proposed[2 * i] = trend[i];
            y_proposed[2 * i + 1] = seasonal[i];
        }

        // ##########################################
        // construct A
        DMatrixRMaj A = new DMatrixRMaj(2 * ts.length, 2 * ts.length);
        constructA(A, ts.length, lambda, 1.0);

        // generate new b
        double[] b = new double[ts.length * 2];
        for (int i = 0; i < ts.length; i++) {
            b[i * 2] = ts[i];
            b[i * 2 + 1] = ts[i] + tsdbstlFlush.getV()[i % period];
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
