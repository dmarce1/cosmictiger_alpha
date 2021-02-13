#pragma once

#include <cosmictiger/vector.hpp>

template<class T>
class stack_vector {
   vector<T> data;
   vector<int> bounds;
   size_t begin() const {
      return bounds[bounds.size() - 2];
   }
   size_t end() const {
      return bounds.back();
   }
public:
   CUDA_EXPORT inline stack_vector() {
      THREAD;
      if (tid == 0) {
         bounds.resize(2);
         bounds[0] = 0;
         bounds[1] = 0;
      }SYNC;
   }
   CUDA_EXPORT inline void push(const T &a) {
      THREAD;
      data.push_back(a);
      if (tid == 0) {
         bounds.back()++;}
      SYNC;
   }
   CUDA_EXPORT inline size_t size() const {
      return end() - begin();
   }
   CUDA_EXPORT inline void resize(size_t sz) {
      THREAD;
      data.resize(begin() + sz);
      if (tid == 0) {
         bounds.back() = data.size();
      }SYNC;
   }
   CUDA_EXPORT inline T operator[](size_t i) const {
      return data[begin() + i];
   }
   CUDA_EXPORT inline T& operator[](size_t i) {
      return data[begin() + i];
   }
   CUDA_EXPORT inline void copy_top() {
      THREAD;
      BLOCK;
      const auto sz = size();
      bounds.push_back(end() + sz);
      data.resize(data.size() + sz);
      for (size_t i = begin() + tid; i < end(); i += blocksize) {
         data[i] = data[i - sz];
      }
   }
   CUDA_EXPORT inline void pop_top() {
      data.resize(begin());
      bounds.pop_back();
   }
#ifndef __CUDA_ARCH__
   std::function<void()> to_device(cudaStream_t stream) {
      auto f1 = data.to_device(stream);
      auto f2 = bounds.to_device(stream);
      return [f1, f2]() {
         f1();
         f2();
      };
   }
#endif
};

