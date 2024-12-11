import java.util.List;

public class TestTSDBSTL {
    public static void main(String[] args) throws Exception {
        int period = 12, Qsize = 1500;
        Analysis analysis = LoadData.squareWave(Qsize, period);
        int MAX_TS_SIZE = 1000000;
        int MAX_PAGE_SIZE = 600;
        int MAX_PAGE_NUM = 10;
//        Algorithm.TSDBSTLModel(period, 1e-8, 1.0, analysis, MAX_TS_SIZE, MAX_PAGE_SIZE, MAX_PAGE_NUM);
    }
}
