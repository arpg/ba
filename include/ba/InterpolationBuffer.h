/*
 This file is part of the BA Project.

 Copyright (C) 2013 George Washington University,
 Nima Keivan,
 Gabe Sibley

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef INTERPOLATIONBUFFER_H
#define INTERPOLATIONBUFFER_H

namespace ba {
////////////////////////////////////////////////////////////////////////////////
/// Templated interpolation buffer. Used to smoothly interpolate between stored
/// elements. The interplation is delimited by the time value.
/// ScalarType: the type used for the arithmetic (float or double)
/// ElementType: The type of the element in the interpolation buffer. Needs to
/// provide:
///     ElementType& operator *(const ScalarType rhs) : result of operation with
///     a scalar
///     ElementType& operator +(const ElementType& rhs) : result of addition
///     with an element
template<typename ElementType, typename ScalarType>
struct InterpolationBufferT {
  std::vector<ElementType> elements;
  ScalarType start_time;
  ScalarType end_time;
  ScalarType average_dt;
  InterpolationBufferT(unsigned int size = 1000)
      : start_time(-1),
        end_time(-1),
        average_dt(-1)
  {
    elements.reserve(size);
  }
  //////////////////////////////////////////////////////////////////////////////
  /// \brief Adds an element to the interpolation buffer, updates the average
  /// and end times
  /// \param element The new element to add
  ///
  void AddElement(const ElementType& element) {
    assert(element.time > end_time);
    const size_t nElems = elements.size();
    const ScalarType dt = nElems == 0 ? 0 : element.time - elements.back().time;

    // update the average dt
    average_dt = average_dt == -1 ? dt :
                 (average_dt * nElems + dt) / (nElems + 1);
    // add the element and update the end time
    elements.push_back(element);
    end_time = element.time;
    start_time = elements.front().time;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// \brief Gets the next element in the buffer, depending on the maximum time
  /// interval specified
  /// \param dMaxTime The maximum time interval. The returned element time will
  ///  be <= to this
  /// \param indexOut The output index of the returned element
  /// \param output The output element
  /// \return True if the function returned an intermediate element, false if we
  ///  had to interpolate
  /// in which case we've reached the end
  ///
  bool GetNext(const ScalarType max_time, size_t& index_out,
               ElementType& output) {
    // if we have reached the end, interpolate and signal the end
    if (index_out + 1 >= elements.size()) {
      output = GetElement(max_time, &index_out);
      return false;
    } else if (elements[index_out + 1].time > max_time) {
      output = GetElement(max_time, &index_out);
      return false;
    } else {
      index_out++;
      output = elements[index_out];
      return true;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an element exists for the given time
  /// \param dTime The time for which we check the element
  /// \return True if an element exists for this time
  ///
  bool HasElement(const ScalarType time) {
    return (time >= start_time && time <= end_time);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// \brief Returns an interpolated element. Call HasElement before this
  /// function to make sure
  ///        an element exists for this time
  /// \param dTime The time for which we require the element
  /// \return The element
  ///
  ElementType GetElement(const ScalarType time) {
    size_t index;
    return GetElement(time, &index);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// \brief Returns an interpolated element. Call HasElement before this
  ///  function to make sure
  ///        an element exists for this time
  /// \param dTime The time for which we require the element
  /// \param pIndex The output index
  /// \return The element
  ///
  ElementType GetElement(const ScalarType time, size_t* pIndex) {
    // assert(dTime >= StartTime && dTime <= EndTime);
    // guess what the index would be
    size_t guess_idx = (time - start_time) / average_dt;
    const size_t n_elements = elements.size();
    guess_idx = std::min(std::max((unsigned int) guess_idx, 0u),
                         (unsigned int) elements.size() - 1u);

    // now using the guess, find a direction to go
    if (elements[guess_idx].time > time) {
      // we need to go backwards
      if (guess_idx == 0) {
        *pIndex = guess_idx;
        return elements.front();
      }

      while ((guess_idx - 1) > 0 && elements[guess_idx - 1].time > time) {
        guess_idx--;
      }
      const ScalarType interpolator = (time - elements[guess_idx - 1].time)
          / (elements[guess_idx].time - elements[guess_idx - 1].time);

      *pIndex = guess_idx - 1;
      ElementType res = elements[guess_idx - 1] * (1 - interpolator)
          + elements[guess_idx] * interpolator;

      res.time = time;
      return res;
    } else {
      // we need to go forwards
      if (guess_idx == n_elements - 1) {
        *pIndex = guess_idx;
        return elements.back();
      }

      while ((guess_idx + 1) < n_elements &&
             elements[guess_idx + 1].time < time) {
        guess_idx++;
      }
      const ScalarType interpolator = (time - elements[guess_idx].time)
          / (elements[guess_idx + 1].time - elements[guess_idx].time);

      *pIndex = guess_idx;
      ElementType res = elements[guess_idx] * (1 - interpolator)
          + elements[guess_idx + 1] * interpolator;
      res.time = time;
      return res;
    }
  }

  std::vector<ElementType> GetRange(const ScalarType star_time,
                                    const ScalarType end_time) {
    std::vector<ElementType> measurements;
    size_t index;
    // get all the imu measurements between these two poses, and add them to a
    // vector
    if (HasElement(star_time)) {
      measurements.push_back(GetElement(star_time, &index));
      ElementType meas;
      while (GetNext(end_time, index, meas)) {
        measurements.push_back(meas);
      }
      // push back the last item
      measurements.push_back(meas);
    }
    return measurements;
  }
};
}

#endif // INTERPOLATIONBUFFER_H
