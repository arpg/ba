#ifndef SPARSEBLOCKMATRIXPRODUCT_H
#define SPARSEBLOCKMATRIXPRODUCT_H

namespace Eigen {

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockVectorProductDenseResult(const Lhs& lhs, const Rhs& rhs, ResultType& res)
{
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
    typedef typename ResultType::Scalar ResultScalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;
    typedef typename Rhs::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    const Index lhsCols = lhs.outerSize();
    eigen_assert(lhsCols*LhsScalar::ColsAtCompileTime == rhs.rows());

    res.setZero();
//    for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
    for (Index ii=0; ii< lhsCols; ++ii)
    {
        for (typename Lhs::InnerIterator lhsIt(lhs, ii); lhsIt; ++lhsIt)
        {
            res.template block<LhsScalar::RowsAtCompileTime,1>(LhsScalar::RowsAtCompileTime*lhsIt.index(),0).noalias() +=
                    lhsIt.value() * rhs.template block<LhsScalar::ColsAtCompileTime,1>(LhsScalar::ColsAtCompileTime*ii,0);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockTransposeVectorProductDenseResultAtb(const Lhs& lhs, const Rhs& rhs, ResultType& res)
{
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
    typedef typename ResultType::Scalar ResultScalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;
    typedef typename Rhs::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    const Index lhsCols = lhs.outerSize();
    eigen_assert(lhs.outerSize()*LhsScalar::ColsAtCompileTime == rhs.rows());

    res.setZero();

    for (Index ii=0; ii<lhsCols; ++ii)
    {
        for (typename Lhs::InnerIterator lhsIt(lhs, ii); lhsIt; ++lhsIt){
            res.template block<LhsScalar::ColsAtCompileTime,1>(LhsScalar::ColsAtCompileTime*ii,0).noalias() +=
                    lhsIt.value().transpose() * rhs.template block<LhsScalar::RowsAtCompileTime,1>(LhsScalar::RowsAtCompileTime*lhsIt.index(),0);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockTransposeProduct(const Lhs& lhs,const Rhs& rhs, ResultType& res)
{
    const typename ResultType::Scalar zero = ResultType::Scalar::Zero();
    typedef typename ResultType::Scalar ResultScalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;
    typedef typename Lhs::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    Index rows = lhs.outerSize(); // this is because lhs is actually transposed in the multiplication
    Index cols = rhs.outerSize();

    // allocate a temporary buffer
//     Eigen::internal::BlockAmbiVector<ResultScalar,Index> tempVector(rows,zero);

    Index estimated_nnz_prod =  lhs.nonZeros() + rhs.nonZeros();

    res.resize(rows, cols);

    res.reserve(estimated_nnz_prod);
    for (Index j=0; j<cols; ++j)
    {
        res.startVec(j);
        for (Index i=0; i<rows; ++i)
        {
            typename Rhs::InnerIterator jIt(rhs, j);
            typename Lhs::InnerIterator iIt(lhs, i);

            // do the dot product of the ith and jth columns
            bool bEmpty = true;
            ResultScalar value;
            value.setZero();
            while(jIt && iIt){
                if(jIt.index() == iIt.index()){
                    bEmpty = false;
                    value += iIt.value().transpose() * jIt.value();
                    ++jIt;
                    ++iIt;
                }else if(jIt.index() < iIt.index()) {
                    ++jIt;
                }else{
                    ++iIt;
                }
            }

            // insert the result of the dot product
            if(bEmpty == false){
                res.insertBackByOuterInner(j,i) = value;
            }

        }
    }
    res.finalize();
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockProduct(const Lhs& lhs, const Rhs& rhs, ResultType& res)
{
    const typename ResultType::Scalar zero = ResultType::Scalar::Zero();
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
//    const int nBlockRows = ResultType::Scalar::RowsAtCompileTime;
//    const int nBlockCols = ResultType::Scalar::ColsAtCompileTime;
    typedef typename ResultType::Scalar ResultScalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;
    typedef typename Rhs::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    Index rows = lhs.innerSize();
    Index cols = rhs.outerSize();
    //int size = lhs.outerSize();
    eigen_assert(lhs.outerSize() == rhs.innerSize());

    // allocate a temporary buffer
    Eigen::internal::BlockAmbiVector<ResultScalar,Index> tempVector(rows,zero);

    Index estimated_nnz_prod = lhs.nonZeros() + rhs.nonZeros();

    res.resize(rows, cols);

    res.reserve(estimated_nnz_prod);
    const double dRowCols = double(lhs.rows()*rhs.cols());
    const double ratioColRes = dRowCols == 0 ? 0 : double(estimated_nnz_prod)/dRowCols;
    for (Index j=0; j<cols; ++j)
    {
        // FIXME:
        //double ratioColRes = (double(rhs.innerVector(j).nonZeros()) + double(lhs.nonZeros())/double(lhs.cols()))/double(lhs.rows());
        // let's do a more accurate determination of the nnz ratio for the current column j of res
        tempVector.init(ratioColRes);
        tempVector.setZero();
        // this is going down the jth column of the rhs
        for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
        {
            // FIXME should be written like this: tmp += rhsIt.value() * lhs.col(rhsIt.index())
            tempVector.restart();
            for (typename Lhs::InnerIterator lhsIt(lhs, rhsIt.index()); lhsIt; ++lhsIt)
            {
                tempVector.coeffRef(lhsIt.index()).noalias() += lhsIt.value() * rhsIt.value();
            }
        }
        res.startVec(j);
        for (typename Eigen::internal::BlockAmbiVector<ResultScalar,Index>::Iterator it(tempVector,zero); it; ++it){
            res.insertBackByOuterInner(j,it.index()) = it.value();
        }
    }
    res.finalize();
}



//////////////////////////////////////////////////////////////////////////////////////////
template<typename Matrix>
static void SparseBlockAdd(const Matrix& lhs, const Matrix& rhs, Matrix& res, const int nRhsCoef = 1 )
{
    // cannot add in-place.
    assert(&lhs!=&res && &rhs != &res);

    const typename Matrix::Scalar zero = Matrix::Scalar::Zero();
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
    typedef typename Matrix::Scalar Scalar;
    typedef typename Matrix::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    Index rows = lhs.innerSize();
    Index cols = lhs.outerSize();
    //int size = lhs.outerSize();
    eigen_assert(lhs.innerSize() == rhs.innerSize() && lhs.outerSize() == rhs.outerSize());

    // allocate a temporary buffer
    Eigen::internal::BlockAmbiVector<Scalar,Index> tempVector(rows,zero);

    Index estimated_nnz_prod = lhs.nonZeros() + rhs.nonZeros();

    // mimics a resizeByInnerOuter:
    res.resize(rows, cols);

    res.reserve(estimated_nnz_prod);
    double ratioColRes = double(estimated_nnz_prod)/double(lhs.rows()*rhs.cols());
    for (Index j=0; j<cols; ++j)
    {
        tempVector.init(ratioColRes);
        tempVector.setZero();
        for (typename Matrix::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
        {
            tempVector.coeffRef(rhsIt.index()).noalias() += rhsIt.value()*nRhsCoef;
        }

        tempVector.restart();

        for (typename Matrix::InnerIterator lhsIt(lhs, j); lhsIt; ++lhsIt)
        {
            tempVector.coeffRef(lhsIt.index()).noalias() += lhsIt.value();
        }

        res.startVec(j);
        for (typename Eigen::internal::BlockAmbiVector<Scalar,Index>::Iterator it(tempVector,zero); it; ++it)
            res.insertBackByOuterInner(j,it.index()) = it.value();
    }
    res.finalize();
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Matrix, typename ResultType>
static void SparseBlockAddDenseResult(const Matrix& lhs, const Matrix& rhs, ResultType const & resMat, const int nRhsCoef = 1 )
{
    // this is a little hack as per (http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html) to
    // enable passing in a block expression as res. We pass in a const reference, then cast the const-ness
    // away to enable writing to it
    ResultType& res = const_cast< ResultType& >(resMat);

    const typename Matrix::Scalar zero = Matrix::Scalar::Zero();
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
    typedef typename Matrix::Scalar BlockType;
    typedef typename Matrix::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    Index rows = lhs.innerSize();
    Index cols = lhs.outerSize();
    //int size = lhs.outerSize();
    eigen_assert(lhs.innerSize() == rhs.innerSize() && lhs.outerSize() == rhs.outerSize());

    // reset the output
    res.setZero();
    for (Index j=0; j<cols; ++j)
    {
        for (typename Matrix::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
        {
            res.template block<BlockType::RowsAtCompileTime,BlockType::ColsAtCompileTime>
                    ( rhsIt.index()*BlockType::RowsAtCompileTime, j*BlockType::ColsAtCompileTime ).noalias() += rhsIt.value()*nRhsCoef;
        }

        for (typename Matrix::InnerIterator lhsIt(lhs, j); lhsIt; ++lhsIt)
        {
            res.template block<BlockType::RowsAtCompileTime,BlockType::ColsAtCompileTime>
                    ( lhsIt.index()*BlockType::RowsAtCompileTime, j*BlockType::ColsAtCompileTime ) += lhsIt.value();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Lhs>
static inline void SparseBlockSubtract(const Lhs& lhs, const Lhs& rhs, Lhs& res)
{
    SparseBlockAdd(lhs,rhs,res,-1);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename Matrix, typename ResultType>
static void SparseBlockSubtractDenseResult(const Matrix& lhs, const Matrix& rhs, ResultType const & res)
{    
    SparseBlockAddDenseResult(lhs,rhs,res,-1);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// UNOPTIMIZED -- USED FOR TESTING ONLY
template<typename SparseMatrix, typename DenseMatrix>
static void LoadSparseFromDense(const DenseMatrix& dense, SparseMatrix& sparse)
{
    typedef typename SparseMatrix::Scalar BlockType;
    const int _BlockRows = BlockType::RowsAtCompileTime;
    const int _BlockCols = BlockType::ColsAtCompileTime;

    for( int ii = 0 ; ii < sparse.rows() ; ii++ ){
        for( int jj = 0 ; jj < sparse.cols() ; jj++ ) {
            sparse.coeffRef(ii,jj) = dense.block(ii*_BlockRows, jj*_BlockCols,_BlockRows,_BlockCols);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename DenseMatrix, typename SparseMatrix >
static void LoadDenseFromSparse(const SparseMatrix& sparse,DenseMatrix const & denseMat)
{
    DenseMatrix& dense = const_cast< DenseMatrix& >(denseMat);

    typedef typename SparseMatrix::Scalar BlockType;
    const int _BlockRows = BlockType::RowsAtCompileTime;
    const int _BlockCols = BlockType::ColsAtCompileTime;

    assert(dense.rows() == _BlockRows*sparse.rows() && dense.cols() == _BlockCols*sparse.cols());

    // dense.resize(_BlockRows*sparse.rows(), _BlockCols*sparse.cols());
    dense.setZero();
    for (int jj=0; jj<sparse.cols(); ++jj)
    {
        for (typename SparseMatrix::InnerIterator sparseIt(sparse, jj); sparseIt; ++sparseIt)
        {
            dense.template block<BlockType::RowsAtCompileTime,BlockType::ColsAtCompileTime>(sparseIt.index()*_BlockRows, jj*_BlockCols) =
                    sparseIt.value();
        }
    }
//    for( int ii = 0 ; ii < sparse.rows() ; ii++ ){
//        for( int jj = 0 ; jj < sparse.cols() ; jj++ ) {
//            dense.template block<BlockType::RowsAtCompileTime,BlockType::ColsAtCompileTime>(ii*_BlockRows, jj*_BlockCols) = sparse.coeff(ii,jj);
//        }
//    }
}

} // end namespace Eigen


#endif // SPARSEBLOCKMATRIXPRODUCT_H
