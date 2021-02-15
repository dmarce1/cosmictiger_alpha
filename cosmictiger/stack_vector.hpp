#pragma once

#include <cosmictiger/vector.hpp>

template<class T>
class stack_vector {
   vector<T> data;
   vector<int> bounds;
   CUDA_EXPORT  inline size_t begin() const {
      assert(bounds.size()>=2);
      return bounds[bounds.size() - 2];
   }
   CUDA_EXPORT inline size_t end() const {
      assert(bounds.size()>=2);
      return bounds.back();
   }
public:
   CUDA_EXPORT inline stack_vector() {
      THREAD;
      data.reserve(1024);
      bounds.reserve(32);
      if (tid == 0) {
         bounds.resize(2);
         bounds[0] = 0;
         bounds[1] = 0;
      }SYNC;
   }
   CUDA_EXPORT inline void push(const T &a) {
      THREAD;
      assert(bounds.size()>=2);
      data.push_back(a);
      if (tid == 0) {
         bounds.back()++;}
      SYNC;
   }
   CUDA_EXPORT inline size_t size() const {
      assert(bounds.size()>=2);
       return end() - begin();
   }
   CUDA_EXPORT inline void resize(size_t sz) {
      THREAD;
      assert(bounds.size()>=2);
      data.resize(begin() + sz);
      if (tid == 0) {
         bounds.back() = data.size();
      }SYNC;
   }
   CUDA_EXPORT inline T operator[](size_t i) const {
      assert(i<size());
      return data[begin() + i];
   }
   CUDA_EXPORT inline T& operator[](size_t i) {
      assert(i<size());
      return data[begin() + i];
   }
   CUDA_EXPORT inline stack_vector copy_top() const {
      THREAD;
      BLOCK;
      stack_vector res;
      res.resize(size());
      SYNC;
      for( size_t i = tid; i < size(); i+= blocksize) {
         res[i] = (*this)[i];
      }SYNC;
      return res;
   }
   CUDA_EXPORT inline void push_top() {
      THREAD;
      BLOCK;
      const auto sz = size();
      bounds.push_back(end() + sz);
      data.resize(data.size() + sz);
      SYNC;
      for (size_t i = begin() + tid; i < end(); i += blocksize) {
         data[i] = data[i - sz];
      }SYNC;
   }
   CUDA_EXPORT inline void pop_top() {
      assert(bounds.size()>=2);
      data.resize(begin());
      bounds.pop_back();
   }
#ifndef __CUDACC__
   std::function<void()> to_device(cudaStream_t stream) {
      assert(bounds.size()>=2);
      auto f1 = data.to_device(stream);
      auto f2 = bounds.to_device(stream);
      return [f1, f2]() {
         f2();
         f1();
      };
   }
#endif
};

