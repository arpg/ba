#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Eigen>

namespace ba {
template<typename BlockType>
class BlockSparseMatrix
{
public:
  void Clear()
  {

  }

  void Reserve(const int num_values)
  {
    values_.reserve(num_values);
    row_indices_.reserve(num_values);
  }

  void ReserveColumns(const Eigen::VectorXi& col_sizes)
  {
    values_.resize(col_sizes);
    row_indices_.resize(col_sizes);
    row_indices_.assign(col_sizes, -1);
    col_ptrs_.resize(col_sizes.rows());
    col_end_ptrs_.resize(col_sizes.rows());
    int current_val_idx = 0;
     // Allocate the column pointers
    for (const int ii : col_sizes) {
      col_ptrs_[ii] = current_val_idx;
      col_end_ptrs_[ii] = current_val_idx;
      current_val_idx += col_sizes[ii];
    }
  }

  void Resize(const int rows, const int cols)
  {
    rows_ = rows;
    cols_ = cols;
    col_ptrs_ = std::vector<int>(cols, -1);
    row_indices_.clear();
  }

  BlockType& Insert(const int row, const int column)
  {
    // If we are at the end of the values array
    if (column == colums_ - 1) {
      if (col_end_ptrs_[column] == values_.size()) {
        std::cout << "Attempted to insert beyond the end of the values "
                     "array." << std::endl;
        assert(false);
      }
    } else {
      if (col_end_ptrs_[column] == col_ptrs_[column + 1]) {
        std::cout << "Attempted to insert beyond the end of the allocated"
                     "column space. Incorrect ReserveColumns call " <<
                     std::endl;
        assert(false);
      }
    }

    const int value_index = col_end_ptrs_[column];
    // Increment the item count for this column.
    col_end_ptrs_[column]++;
    row_indices_[value_index] = row;
    // Return a reference to the newly added block.
    return values_[value_index];
  }

  void OrderedInsert(const int row, const int column, const BlockType& block)
  {
    if (column < current_col_) {
      std::cout << "Attempted to insert column " << column << " in order when"
                   " the current column is " << current_col_ << std::endl;
      assert(false);
    }

    // Skip any empty colums (due to the unordered insert)
    while (current_col_ == column) {
      col_ptrs_[current_col_] = -1;
    }
  }

private:
  int current_col_;
  int rows_;
  int colums_;
  std::vector<int> col_ptrs_;
  std::vector<int> col_end_ptrs_;
  std::vector<int> row_indices_;
  std::vector<BlockType, Eigen::aligned_allocator<BlockType>> values_;
};
}
