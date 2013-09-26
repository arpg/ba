#ifndef SPARSEBLOCKMATRIXPRODUCT_H
#define SPARSEBLOCKMATRIXPRODUCT_H

namespace Eigen {

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, typename ResultType>
  static void SparseBlockVectorProductDenseResult(
      const Lhs& lhs, const Rhs& rhs, ResultType& res, int rhs_stride = -1,
      int res_stride = -1)
  {
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
    typedef typename ResultType::Scalar ResultScalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;
    typedef typename Rhs::Index Index;

    if (res_stride == -1) {
      res_stride = LhsScalar::RowsAtCompileTime;
    }

    if (rhs_stride == -1) {
      rhs_stride = LhsScalar::ColsAtCompileTime;
    }

    // make sure to call innerSize/outerSize since we fake the storage order.
    const Index lhsCols = lhs.outerSize();
    eigen_assert(lhsCols*LhsScalar::ColsAtCompileTime == rhs.rows());

    res.setZero();
    //    for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
    for (Index ii=0; ii< lhsCols; ++ii)
    {
      for (typename Lhs::InnerIterator lhsIt(lhs, ii); lhsIt; ++lhsIt)
      {
        res.template block<LhsScalar::RowsAtCompileTime,1>(
              res_stride*lhsIt.index(),0).noalias() +=
            lhsIt.value() *
            rhs.template block<LhsScalar::ColsAtCompileTime,1>(
              rhs_stride*ii,0);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, typename ResultType>
  static void SparseBlockTransposeVectorProductDenseResultAtb(
      const Lhs& lhs, const Rhs& rhs, ResultType& res, int res_stride = -1)
  {
    // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
    typedef typename ResultType::Scalar ResultScalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;
    typedef typename Rhs::Index Index;

    if (res_stride == -1) {
      res_stride = LhsScalar::RowsAtCompileTime;
    }
    // make sure to call innerSize/outerSize since we fake the storage order.
    const Index lhsCols = lhs.outerSize();
    eigen_assert(lhs.outerSize()*LhsScalar::ColsAtCompileTime == rhs.rows());

    res.setZero();

    for (Index ii=0; ii<lhsCols; ++ii)
    {
      for (typename Lhs::InnerIterator lhsIt(lhs, ii); lhsIt; ++lhsIt){
        res.template block<LhsScalar::ColsAtCompileTime,1>(
          res_stride*ii, 0).noalias() += lhsIt.value().transpose() *
            rhs.template block<LhsScalar::RowsAtCompileTime,1>(
              LhsScalar::RowsAtCompileTime*lhsIt.index(),0);
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

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, typename ResultType>
  static void SparseBlockProductDenseResult(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
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

    res.setZero();
    for (Index j=0; j<cols; ++j)
    {
      // this is going down the jth column of the rhs
      for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
      {
        const auto& rhsVal = rhsIt.value();
        for (typename Lhs::InnerIterator lhsIt(lhs, rhsIt.index()); lhsIt; ++lhsIt)
        {
          res.template block<LhsScalar::RowsAtCompileTime,RhsScalar::ColsAtCompileTime>
              ( lhsIt.index()*LhsScalar::RowsAtCompileTime, j*RhsScalar::ColsAtCompileTime ).noalias() += lhsIt.value() * rhsVal;
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
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
    for (Index j=0; j<cols; ++j)
    {
      res.startVec(j);
      for (typename Lhs::InnerIterator lhs_it(lhs, j); lhs_it; ++lhs_it)
      {
        res.insertBackByOuterInner(j,lhs_it.index()) =
            lhs_it.value() * rhs.coeff(j, j);
      }
    }
    res.finalize();
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, typename ResultType>
  static void SparseBlockProduct(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                                 const bool upper_triangular = false)
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
    const double ratioColRes =
        dRowCols == 0 ? 0 : double(estimated_nnz_prod)/dRowCols;
    int count = 0;
    for (Index j=0; j<cols; ++j)
    {
      // FIXME:
      //double ratioColRes = (double(rhs.innerVector(j).nonZeros()) + double(lhs.nonZeros())/double(lhs.cols()))/double(lhs.rows());
      // let's do a more accurate determination of the nnz ratio for the current column j of res
      tempVector.init(ratioColRes);
      // tempVector.init(0.01);
      tempVector.setZero();
      // this is going down the jth column of the rhs
      for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
      {
        const auto& rhsVal = rhsIt.value();
        // FIXME should be written like this: tmp += rhsIt.value() * lhs.col(rhsIt.index())
        tempVector.restart();
        for (typename Lhs::InnerIterator lhsIt(lhs, rhsIt.index()); lhsIt; ++lhsIt)
        {
          if (upper_triangular && lhsIt.index() > count) {
            break;
          }
          tempVector.coeffRef(lhsIt.index()).noalias() += lhsIt.value() * rhsVal;
        }
      }

      count++;

      res.startVec(j);
      for (typename Eigen::internal::BlockAmbiVector<ResultScalar,Index>::Iterator it(tempVector,zero); it; ++it){
        res.insertBackByOuterInner(j,it.index()) = it.value();
      }
    }
    res.finalize();
  }



  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, int block_y = 0, int block_x = 0>
  static void SparseBlockAdd(const Lhs& lhs, const Rhs& rhs, Lhs& res, const int nRhsCoef = 1 )
  {
      // cannot add in-place.
      assert(&lhs!=&res && &rhs != &res);

      const typename Lhs::Scalar zero = Lhs::Scalar::Zero();
      // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
      typedef typename Rhs::Scalar RhsBlockType;
      typedef typename Lhs::Scalar BlockType;
      typedef typename Lhs::Index Index;

      // make sure to call innerSize/outerSize since we fake the storage order.
      Index rows = lhs.innerSize();
      Index cols = lhs.outerSize();
      //int size = lhs.outerSize();
      eigen_assert(lhs.innerSize() == rhs.innerSize() && lhs.outerSize() == rhs.outerSize());

      // allocate a temporary buffer
      Eigen::internal::BlockAmbiVector<BlockType,Index> tempVector(rows,zero);

      Index estimated_nnz_prod = lhs.nonZeros() + rhs.nonZeros();

      // mimics a resizeByInnerOuter:
      res.resize(rows, cols);

      res.reserve(estimated_nnz_prod);
      double ratioColRes = double(estimated_nnz_prod)/double(lhs.rows()*rhs.cols());
      for (Index j=0; j<cols; ++j)
      {
          tempVector.init(ratioColRes);
          tempVector.setZero();
          for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
          {
              // tempVector.coeffRef(rhsIt.index()).noalias() += rhsIt.value()*nRhsCoef;
            tempVector.coeffRef(rhsIt.index()).
                template block<
                RhsBlockType::RowsAtCompileTime, RhsBlockType::ColsAtCompileTime>(
                  block_y,block_x).noalias() +=
                rhsIt.value()*(double)nRhsCoef;
          }

          tempVector.restart();

          for (typename Lhs::InnerIterator lhsIt(lhs, j); lhsIt; ++lhsIt)
          {
              tempVector.coeffRef(lhsIt.index()).noalias() += lhsIt.value();
          }

          res.startVec(j);
          for (typename Eigen::internal::BlockAmbiVector<BlockType,Index>::Iterator it(tempVector,zero); it; ++it)
              res.insertBackByOuterInner(j,it.index()) = it.value();
      }
      res.finalize();
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, typename Res,
           int block_y = 0, int block_x = 0>
  static void SparseBlockAddDenseResult(const Lhs& lhs, const Rhs& rhs,
                                        Res const & resMat, const int nRhsCoef = 1 )
  {
      // this is a little hack as per (http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html) to
      // enable passing in a block expression as res. We pass in a const reference, then cast the const-ness
      // away to enable writing to it
      Res& res = const_cast<Res&>(resMat);

      const typename Lhs::Scalar zero = Lhs::Scalar::Zero();
      // return sparse_sparse_product_with_pruning_impl2(lhs,rhs,res);
      typedef typename Lhs::Scalar BlockType;
      typedef typename Rhs::Scalar RhsBlockType;
      typedef typename Lhs::Index Index;

      // make sure to call innerSize/outerSize since we fake the storage order.
      Index rows = lhs.innerSize();
      Index cols = lhs.outerSize();
      //int size = lhs.outerSize();
      eigen_assert(lhs.innerSize() == rhs.innerSize() && lhs.outerSize() == rhs.outerSize());

      // reset the output
      res.setZero();
      for (Index j=0; j<cols; ++j)
      {
          for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
          {
//              res.template block<BlockType::RowsAtCompileTime,BlockType::ColsAtCompileTime>
//                      ( rhsIt.index()*BlockType::RowsAtCompileTime, j*BlockType::ColsAtCompileTime ).noalias() += rhsIt.value()*nRhsCoef;
            res.template block<
                RhsBlockType::RowsAtCompileTime, RhsBlockType::ColsAtCompileTime>
                ( rhsIt.index()*BlockType::RowsAtCompileTime + block_y,
                  j*BlockType::ColsAtCompileTime + block_y ).noalias() +=
                rhsIt.value()*(double)nRhsCoef;
          }

          for (typename Lhs::InnerIterator lhsIt(lhs, j); lhsIt; ++lhsIt)
          {
              res.template block<BlockType::RowsAtCompileTime,BlockType::ColsAtCompileTime>
                      ( lhsIt.index()*BlockType::RowsAtCompileTime, j*BlockType::ColsAtCompileTime ) += lhsIt.value();
          }
      }
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, int block_y = 0, int block_x = 0>
  static inline void SparseBlockSubtract(const Lhs& lhs, const Rhs& rhs,
                                         Lhs& res)
  {
    SparseBlockAdd<Lhs, Rhs, -1, block_y, block_x>(lhs,rhs,res);
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  template<typename Lhs, typename Rhs, typename Res,
           int block_y = 0, int block_x = 0>
  static void SparseBlockSubtractDenseResult(const Lhs& lhs, const Rhs& rhs, Res const & res)
  {
      SparseBlockAddDenseResult<Lhs, Rhs, Res, block_y, block_x>(lhs,rhs,res,-1);
  }


  //////////////////////////////////////////////////////////////////////////////////////////
  /// UNOPTIMIZED -- USED FOR TESTING ONLY
  template<typename SparseMatrix, typename DenseMatrix>
  static void LoadSparseFromDense(const DenseMatrix& dense,
                                  SparseMatrix& sparse)
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
