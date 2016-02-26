#ifndef SPARSEBLOCKMATRIXPRODUCT_H
#define SPARSEBLOCKMATRIXPRODUCT_H

namespace Eigen {

template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockVectorProductDenseResult(
    const Lhs& lhs, const Rhs& rhs, ResultType& res, int rhs_stride = -1,
    int res_stride = -1)
{
  typedef typename Lhs::Scalar LhsScalar;
  typedef typename Rhs::Index Index;

  if (res_stride == -1) {
    res_stride = LhsScalar::RowsAtCompileTime;
  }

  if (rhs_stride == -1) {
    rhs_stride = LhsScalar::ColsAtCompileTime;
  }

  // make sure to call innerSize/outerSize since we fake the storage order.
  const Index lhsCols = lhs.outerSize();
  eigen_assert(lhsCols == rhs.rows() / rhs_stride);

  res.setZero();
  for (Index ii = 0; ii < lhsCols; ++ii)
  {
    for (typename Lhs::InnerIterator lhsIt(lhs, ii); lhsIt; ++lhsIt)
    {
      res.template block<LhsScalar::RowsAtCompileTime, 1>(
            res_stride * lhsIt.index(), 0).noalias() +=
          lhsIt.value() *
          rhs.template block<LhsScalar::ColsAtCompileTime, 1>(
            rhs_stride * ii, 0);
    }
  }
}


template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockTransposeVectorProductDenseResultAtb(
    const Lhs& lhs, const Rhs& rhs, ResultType& res, int res_stride = -1)
{
  // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
  typedef typename Lhs::Scalar LhsScalar;
  typedef typename Rhs::Index Index;

  if (res_stride == -1) {
    res_stride = LhsScalar::RowsAtCompileTime;
  }
  // make sure to call innerSize/outerSize since we fake the storage order.
  const Index lhs_cols = lhs.outerSize();
  eigen_assert(lhs.outerSize() * LhsScalar::ColsAtCompileTime == rhs.rows());

  res.setZero();

  for (Index ii = 0; ii < lhs_cols; ++ii)
  {
    for (typename Lhs::InnerIterator lhsIt(lhs, ii); lhsIt; ++lhsIt){
      res.template block<LhsScalar::ColsAtCompileTime, 1>(
            res_stride * ii, 0).noalias() += lhsIt.value().transpose() *
          rhs.template block<LhsScalar::RowsAtCompileTime, 1>(
            LhsScalar::RowsAtCompileTime*lhsIt.index(), 0);
    }
  }
}


template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockTransposeProduct(const Lhs& lhs,const Rhs& rhs,
                                        ResultType& res)
{
  typedef typename ResultType::Scalar ResultScalar;
  typedef typename Lhs::Index Index;

  // make sure to call innerSize/outerSize since we fake the storage order.
  // this is because lhs is actually transposed in the multiplication
  Index rows = lhs.outerSize();
  Index cols = rhs.outerSize();

  Index estimated_nnz_prod =  lhs.nonZeros() + rhs.nonZeros();

  res.resize(rows, cols);

  res.reserve(estimated_nnz_prod);
  for (Index j=0; j<cols; ++j)
  {
    res.startVec(j);
    for (Index i=0; i<rows; ++i)
    {
      typename Lhs::InnerIterator lhs_it(lhs, i);

      bool is_empty = true;
      ResultScalar value;
      value.setZero();
      for (typename Rhs::InnerIterator rhs_it(rhs, j); rhs_it; ++rhs_it)
      {
        const int rhs_index = rhs_it.index();
        while (lhs_it.index() < rhs_index && lhs_it) {
          ++lhs_it;
        }

        if (!lhs_it) {
          break;
        }

        if (lhs_it.index() == rhs_it.index() ){
          value += lhs_it.value().transpose() * rhs_it.value();
          is_empty = false;
        }
      }

      // insert the result of the dot product
      if(is_empty == false){
        res.insertBackByOuterInner(j,i) = value;
      }

    }
  }
  res.finalize();
}

template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockProductDenseResult(const Lhs& lhs, const Rhs& rhs,
                                          ResultType& res)
{
  // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
  typedef typename Lhs::Scalar LhsScalar;
  typedef typename Rhs::Scalar RhsScalar;
  typedef typename Rhs::Index Index;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = rhs.outerSize();
  //int size = lhs.outerSize();
  eigen_assert(lhs.outerSize() == rhs.innerSize());

  res.setZero();
  for (Index jj = 0; jj < cols; ++jj)
  {
    // this is going down the jth column of the rhs
    for (typename Rhs::InnerIterator rhs_it(rhs, jj); rhs_it; ++rhs_it)
    {
      const auto& rhs_val = rhs_it.value();
      for (typename Lhs::InnerIterator lhs_it(lhs, rhs_it.index());
           lhs_it; ++lhs_it)
      {
        res.template block<LhsScalar::RowsAtCompileTime,
                           RhsScalar::ColsAtCompileTime>
            (lhs_it.index()*LhsScalar::RowsAtCompileTime,
             jj * RhsScalar::ColsAtCompileTime).noalias() +=
            lhs_it.value() * rhs_val;
      }
    }
  }
}

template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockDiagonalRhsProduct(const Lhs& lhs, const Rhs& rhs,
                                          ResultType& res)
{
  typedef typename Rhs::Index Index;
  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = rhs.outerSize();
  eigen_assert(lhs.outerSize() == rhs.innerSize());
  res.resize(rows, cols);
  res.reserve(lhs.nonZeros());
  for (Index jj = 0; jj < cols; ++jj)
  {
    res.startVec(jj);
    for (typename Lhs::InnerIterator lhs_it(lhs, jj); lhs_it; ++lhs_it)
    {
      res.insertBackByOuterInner(jj, lhs_it.index()) =
          lhs_it.value() * rhs.coeff(jj, jj);
    }
  }
  res.finalize();
}

template<typename Lhs, typename Rhs, typename ResultType>
static void SparseBlockProduct(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                               const bool upper_triangular = false)
{
  const typename ResultType::Scalar zero = ResultType::Scalar::Zero();
  // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
  //    const int nBlockRows = ResultType::Scalar::RowsAtCompileTime;
  //    const int nBlockCols = ResultType::Scalar::ColsAtCompileTime;
  typedef typename ResultType::Scalar ResultScalar;
  typedef typename Rhs::Index Index;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = rhs.outerSize();
  //int size = lhs.outerSize();
  eigen_assert(lhs.outerSize() == rhs.innerSize());

  // allocate a temporary buffer
  // TODO: The +10 here is due to an issue with ambivector when the size
  // is too small (such as 1, used in EstimateRelativePoses Gauss Newton).
  // This is only an issue in sparse mode.
  Eigen::internal::BlockAmbiVector<ResultScalar,Index>
      temp_vector(rows+10,zero);

  Index estimated_nnz_prod = lhs.nonZeros() + rhs.nonZeros();

  res.resize(rows, cols);

  res.reserve(estimated_nnz_prod);
  const double row_cols = double(lhs.rows() * rhs.cols());
  const double ratio_col_res =
      row_cols == 0 ? 0 : double(estimated_nnz_prod) / row_cols;
  int count = 0;
  for (Index jj = 0; jj < cols; ++jj)
  {
    // FIXME:
    //double ratioColRes = (double(rhs.innerVector(j).nonZeros()) +
    // double(lhs.nonZeros())/double(lhs.cols()))/double(lhs.rows());
    // let's do a more accurate determination of the nnz ratio for
    // the current column j of res
    temp_vector.init(ratio_col_res);
    // tempVector.init(0.01);
    temp_vector.setZero();

    // this is going down the jth column of the rhs
    for (typename Rhs::InnerIterator rhs_it(rhs, jj); rhs_it; ++rhs_it)
    {
      const auto& rhs_val = rhs_it.value();
      // FIXME should be written like this: tmp += rhsIt.value() *
      // lhs.col(rhsIt.index())
      temp_vector.restart();
      for (typename Lhs::InnerIterator lhs_it(lhs, rhs_it.index());
           lhs_it; ++lhs_it)
      {
        if (upper_triangular && lhs_it.index() > count) {
          break;
        }
        temp_vector.coeffRef(lhs_it.index()).noalias() +=
            lhs_it.value() * rhs_val;
      }
    }

    count++;

    res.startVec(jj);
    for (typename Eigen::internal::BlockAmbiVector<ResultScalar,
                                                   Index>::Iterator
         it(temp_vector, zero); it; ++it){
      res.insertBackByOuterInner(jj, it.index()) = it.value();
    }
  }
  res.finalize();
}


template<typename Lhs, typename Rhs, int block_y = 0, int block_x = 0>
static void SparseBlockAdd(const Lhs& lhs, const Rhs& rhs, Lhs& res,
                           const int rhs_coef = 1 )
{
  // this function cannot add in-place.

  const typename Lhs::Scalar zero = Lhs::Scalar::Zero();
  // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
  typedef typename Rhs::Scalar RhsBlockType;
  typedef typename Lhs::Scalar BlockType;
  typedef typename Lhs::Index Index;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = lhs.outerSize();
  //int size = lhs.outerSize();
  eigen_assert(lhs.innerSize() == rhs.innerSize() &&
               lhs.outerSize() == rhs.outerSize());

  // allocate a temporary buffer
  // TODO: The +10 here is due to an issue with ambivector when the size
  // is too small (such as 1, used in EstimateRelativePoses Gauss Newton).
  // This is only an issue in sparse mode.
  Eigen::internal::BlockAmbiVector<BlockType,Index> temp_vector(rows+10, zero);

  Index estimated_nnz_prod = lhs.nonZeros() + rhs.nonZeros();

  // mimics a resizeByInnerOuter:
  res.resize(rows, cols);

  res.reserve(estimated_nnz_prod);
  double ratio_col_res = double(estimated_nnz_prod) /
      double(lhs.rows()*rhs.cols());
  for (Index jj = 0; jj < cols; ++jj)
  {
    temp_vector.init(ratio_col_res);
    temp_vector.setZero();
    for (typename Rhs::InnerIterator rhsIt(rhs, jj); rhsIt; ++rhsIt)
    {
      temp_vector.coeffRef(rhsIt.index()).
          template block<
          RhsBlockType::RowsAtCompileTime, RhsBlockType::ColsAtCompileTime>(
            block_y, block_x).noalias() +=
          rhsIt.value() * (double)rhs_coef;
    }

    temp_vector.restart();

    for (typename Lhs::InnerIterator lhs_it(lhs, jj); lhs_it; ++lhs_it)
    {
      temp_vector.coeffRef(lhs_it.index()).noalias() += lhs_it.value();
    }

    res.startVec(jj);
    for (typename Eigen::internal::BlockAmbiVector<BlockType, Index>::Iterator
         it(temp_vector, zero); it; ++it)
      res.insertBackByOuterInner(jj, it.index()) = it.value();
  }
  res.finalize();
}

template<typename Lhs, typename Rhs, typename Res,
         int block_y = 0, int block_x = 0>
static void SparseBlockAddDenseResult(const Lhs& lhs, const Rhs& rhs,
                                      Res const & resMat,
                                      const int rhs_coef = 1 )
{
  // this is a little hack as per
  // (http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html) to
  // enable passing in a block expression as res. We pass in a const reference,
  // then cast the const-ness away to enable writing to it
  Res& res = const_cast<Res&>(resMat);

  // const typename Lhs::Scalar zero = Lhs::Scalar::Zero();
  // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
  typedef typename Lhs::Scalar BlockType;
  typedef typename Rhs::Scalar RhsBlockType;
  typedef typename Lhs::Index Index;

  // make sure to call innerSize/outerSize since we fake the storage order.
  // Index rows = lhs.innerSize();
  Index cols = lhs.outerSize();
  //int size = lhs.outerSize();
  eigen_assert(lhs.innerSize() == rhs.innerSize() &&
               lhs.outerSize() == rhs.outerSize());

  // reset the output
  res.setZero();
  for (Index j=0; j<cols; ++j)
  {
    for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
    {
      res.template block<
          RhsBlockType::RowsAtCompileTime, RhsBlockType::ColsAtCompileTime>
          ( rhsIt.index()*BlockType::RowsAtCompileTime + block_y,
            j * BlockType::ColsAtCompileTime + block_y ).noalias() +=
          rhsIt.value()*(double)rhs_coef;
    }

    for (typename Lhs::InnerIterator lhs_it(lhs, j); lhs_it; ++lhs_it)
    {
      res.template block<BlockType::RowsAtCompileTime,
                         BlockType::ColsAtCompileTime>
          (lhs_it.index() * BlockType::RowsAtCompileTime,
           j * BlockType::ColsAtCompileTime ) += lhs_it.value();
    }
  }
}

template<typename Lhs, typename Rhs, int block_y = 0, int block_x = 0>
static inline void SparseBlockSubtract(const Lhs& lhs, const Rhs& rhs,
                                       Lhs& res)
{
  SparseBlockAdd<Lhs, Rhs, -1, block_y, block_x>(lhs, rhs, res);
}

template<typename Lhs, typename Rhs, typename Res,
         int block_y = 0, int block_x = 0>
static void SparseBlockSubtractDenseResult(const Lhs& lhs, const Rhs& rhs,
                                           Res const & res)
{
  SparseBlockAddDenseResult<Lhs, Rhs, Res, block_y, block_x>(
        lhs, rhs, res, -1);
}


/// UNOPTIMIZED -- USED FOR TESTING ONLY
template<typename SparseMatrix, typename DenseMatrix>
static void LoadSparseFromDense(const DenseMatrix& dense,
                                SparseMatrix& sparse)
{
  typedef typename SparseMatrix::Scalar BlockType;
  const int block_rows = BlockType::RowsAtCompileTime;
  const int block_cols = BlockType::ColsAtCompileTime;

  for( int ii = 0 ; ii < sparse.rows() ; ii++ ){
    for( int jj = 0 ; jj < sparse.cols() ; jj++ ) {
      sparse.coeffRef(ii, jj) =
          dense.block(ii * block_rows, jj * block_cols, block_rows, block_cols);
    }
  }
}

template<typename SparseMatrix, typename DenseMatrix,
         int stride_rows = SparseMatrix::Scalar::RowsAtCompileTime,
         int stride_cols = SparseMatrix::Scalar::ColsAtCompileTime>
static void LoadDenseFromSparse(const SparseMatrix& sparse,
                                DenseMatrix const & dense_mat)
{
  DenseMatrix& dense = const_cast< DenseMatrix& >(dense_mat);

  typedef typename SparseMatrix::Scalar BlockType;
  #ifndef NDEBUG
  const int block_rows = BlockType::RowsAtCompileTime;
  const int block_cols = BlockType::ColsAtCompileTime;

  assert(dense.rows() == block_rows * sparse.rows() &&
         dense.cols() == block_cols * sparse.cols());
  #endif

  dense.setZero();
  for (int jj = 0; jj < sparse.cols(); ++jj)
  {
    for (typename SparseMatrix::InnerIterator sparse_it(sparse, jj);
         sparse_it; ++sparse_it)
    {
      dense.template block<BlockType::RowsAtCompileTime,
          BlockType::ColsAtCompileTime>(sparse_it.index() * stride_rows,
                                        jj * stride_cols) =
          sparse_it.value();
    }
  }
}

} // end namespace Eigen


#endif // SPARSEBLOCKMATRIXPRODUCT_H
