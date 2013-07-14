#include <ba/BundleAdjuster.h>
#include <ba/SparseBlockMatrixOps.h>
#include <ba/InterpolationBuffer.h>

using namespace ba;

/////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
    // test imu data
    ImuMeasurementT<double> meas(Eigen::Vector3d::Random(),Eigen::Vector3d::Random(),0);
    std::cout << "Measurement w:" << meas.W.transpose() << " a " << meas.A.transpose() << std::endl;
    ImuMeasurementT<double> meas2 = meas*2;
    std::cout << "Measurement*2 w:" << meas2.W.transpose() << " a " << meas2.A.transpose() << std::endl;
    ImuMeasurementT<double> measAdd = meas + meas2;
    std::cout << "Measurement sum w:" << measAdd.W.transpose() << " a " << measAdd.A.transpose() << std::endl;


    // test the interpolation buffer
    InterpolationBuffer<double,double> imuBuffer;


    unsigned int uRows = 70, uCols = 50;
    {
        // load up a sparse eigen matrix
        Eigen::MatrixXd testMat(uRows*6, uCols*3);
        Eigen::MatrixXd testMat2(uCols*3,uRows*6);
        // fille the matrices with random stuff
        testMat = Eigen::MatrixXd::Random(testMat.rows(),testMat.cols());
        testMat2 = Eigen::MatrixXd::Random(testMat2.rows(),testMat2.cols());
        Eigen::SparseBlockMatrix<Eigen::Matrix<double,6,3> > testBlockMat(uRows,uCols);
        Eigen::SparseBlockMatrix<Eigen::Matrix<double,3,6> > testBlockMat2(uCols,uRows);
        Eigen::SparseBlockMatrix<Eigen::Matrix<double,6,6> > testBlockMatRes(uRows,uRows);

        // now load up the two sparse matrices
        Eigen::LoadSparseFromDense(testMat, testBlockMat);
        Eigen::LoadSparseFromDense(testMat2, testBlockMat2);

        Eigen::MatrixXd sparseDenseTest(testMat.rows(),testMat.cols());
        Eigen::LoadDenseFromSparse(testBlockMat,sparseDenseTest);

        Eigen::MatrixXd sparseDenseTest2(testMat2.rows(),testMat2.cols());
        Eigen::LoadDenseFromSparse(testBlockMat2,sparseDenseTest2);

        // now convert back to dense
        std::cout << "Error for LoadSparseFromDense && LoadDenseFromSparse: " << (sparseDenseTest - testMat).norm() + (sparseDenseTest2 - testMat2).norm() << std::endl;

        Eigen::MatrixXd denseRes = testMat * testMat2;
        Eigen::SparseBlockProduct(testBlockMat,testBlockMat2,testBlockMatRes);

        Eigen::MatrixXd sparseDenseRes(denseRes.rows(),denseRes.cols());
        Eigen::LoadDenseFromSparse(testBlockMatRes,sparseDenseRes);

        // now convert back to dense
        std::cout << "Error for SparseBlockProduct (dense matrx): " << (denseRes - sparseDenseRes).norm() << std::endl;

        denseRes = testMat * testMat2.col(0);
        sparseDenseRes = Eigen::MatrixXd (denseRes.rows(),1);
        Eigen::SparseBlockVectorProductDenseResult(testBlockMat,testMat2.col(0),sparseDenseRes);

        std::cout << "Error for SparseBlockVectorProductDenseResult (dense matrx): " << (denseRes - sparseDenseRes).norm() << std::endl;

        testMat.setZero();
        testMat2.setZero();
        testBlockMat.setZero();
        testBlockMat2.setZero();

        // create some random matrices
        for(unsigned int ii = 0 ; ii < (uRows*uCols)/20 ; ++ii){
            const unsigned int uRow = rand() % (uRows-1);
            const unsigned int uCol = rand() % (uCols-1);
            testMat.block(uRow*6,uCol*3,6,3) = Eigen::MatrixXd::Random(6,3);
            testBlockMat.coeffRef(uRow,uCol) = testMat.block(uRow*6,uCol*3,6,3);
        }

        for(unsigned int ii = 0 ; ii < (uRows*uCols)/20 ; ++ii){
            const unsigned int uRow = rand() % (uCols-1);
            const unsigned int uCol = rand() % (uRows-1);
            testMat2.block(uRow*3,uCol*6,3,6) = Eigen::MatrixXd::Random(3,6);
            testBlockMat2.coeffRef(uRow,uCol) = testMat2.block(uRow*3,uCol*6,3,6);
        }


        sparseDenseTest = Eigen::MatrixXd(testMat.rows(),testMat.cols());
        Eigen::LoadDenseFromSparse(testBlockMat,sparseDenseTest);

        sparseDenseTest2 = Eigen::MatrixXd(testMat2.rows(),testMat2.cols());
        Eigen::LoadDenseFromSparse(testBlockMat2,sparseDenseTest2);

        denseRes = testMat * testMat2;
        Eigen::SparseBlockProduct(testBlockMat,testBlockMat2,testBlockMatRes);

        sparseDenseRes = Eigen::MatrixXd(denseRes.rows(),denseRes.cols());
        Eigen::LoadDenseFromSparse(testBlockMatRes,sparseDenseRes);

        // now convert back to dense
        std::cout << "Error for SparseBlockProduct (sparse matrx): " << (denseRes - sparseDenseRes).norm() << std::endl;

        denseRes = testMat * testMat2.col(0);
        sparseDenseRes = Eigen::MatrixXd(denseRes.rows(),1);
        Eigen::SparseBlockVectorProductDenseResult(testBlockMat,testMat2.col(0),sparseDenseRes);

        std::cout << "Error for SparseBlockVectorProductDenseResult (sparse matrx): " << (denseRes - sparseDenseRes).norm() << std::endl;
    }

    {
        Eigen::MatrixXd testMat(uRows*6, uCols*3);
        Eigen::SparseBlockMatrix<Eigen::Matrix<double,6,3> > testBlockMat(uRows,uCols);
        Eigen::SparseBlockMatrix<Eigen::Matrix<double,6,3> > testBlockMatRes(uRows,uCols);
        // fille the matrices with random stuff
        testMat = Eigen::MatrixXd::Random(testMat.rows(),testMat.cols());

        Eigen::LoadSparseFromDense(testMat, testBlockMat);
        Eigen::MatrixXd denseAddRes = testMat + testMat;

        Eigen::SparseBlockAdd(testBlockMat,testBlockMat,testBlockMatRes);
        Eigen::MatrixXd sparseDenseRes(testMat.rows(),testMat.cols());
        Eigen::LoadDenseFromSparse(testBlockMatRes,sparseDenseRes);

        std::cout << "Error for SparseBlockAdd: " << (denseAddRes - sparseDenseRes).norm() << std::endl;

        sparseDenseRes.setZero();
        SparseBlockAddDenseResult(testBlockMat,testBlockMat,sparseDenseRes);

        std::cout << "Error for SparseBlockAddDenseResult: " << (denseAddRes - sparseDenseRes).norm() << std::endl;
    }
}
