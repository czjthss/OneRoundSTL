import algorithm.utils.LDLT;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.decomposition.chol.CholeskyDecompositionLDL_DDRM;
import org.ejml.dense.row.CommonOps_DDRM;


public class TestLDLT {
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
    public void testQueryLD() {
        int N = 100;
        double epsilon = 1e-8;
        double lambda_t = 1.5, lambda_s = 2.0;

        // construct A
        DMatrixRMaj A = new DMatrixRMaj(2 * N, 2 * N);
        constructA(A, N, lambda_t, lambda_s);

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

        LDLT ldlt_proposed = new LDLT(N, epsilon, lambda_t, lambda_s);
        for (int i = 0; i < 2 * N; ++i) {
            for (int j = 0; j < 2 * N; ++j) {
                assertEquals(L.get(i, j), ldlt_proposed.queryL(i, j), epsilon);
                assertEquals(D.get(i, j), ldlt_proposed.queryD(i, j), epsilon);
            }
        }
    }

    @Test
    public void testQueryEndLD() {
        int N = 100, N_new = 50;
        double epsilon = 1e-8;
        double lambda_t = 1.5, lambda_s = 2.0;

        // construct A
        DMatrixRMaj A = new DMatrixRMaj(2 * N, 2 * N);
        constructA(A, N, lambda_t, lambda_s);

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

        // comparison
        LDLT ldlt_proposed = new LDLT(N, epsilon, lambda_t, lambda_s);
        for (int i = 2 * N_new - 4; i < 2 * N_new; ++i) {
            for (int j = i - 4; j < i; ++j) {
                assertEquals(L.get(i + 2 * (N - N_new), j + 2 * (N - N_new)), ldlt_proposed.queryEndL(i, j, N_new), epsilon);
                assertEquals(D.get(i + 2 * (N - N_new), j + 2 * (N - N_new)), ldlt_proposed.queryEndD(i, j, N_new), epsilon);
            }
        }
    }
}
