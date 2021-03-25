#pragma once

#include <cosmictiger/vector.hpp>

template<class T>
class stack_vector {
   vector<T> data;
   vector<int> bounds;
   CUDA_EXPORT inline int begin() const {
      assert(bounds.size() >= 2);
      return bounds[bounds.size() - 2];
   }
   CUDA_EXPORT inline int end() const {
      assert(bounds.size() >= 2);
      return bounds.back();
   }
public:
   CUDA_EXPORT inline int depth() const {
      return bounds.size() - 2;
   }
   CUDA_EXPORT inline stack_vector() {
      THREAD;
      bounds.reserve(TREE_MAX_DEPTH + 1);
      bounds.resize(2,vectorPOD);
      if (tid == 0) {
         bounds[0] = 0;
         bounds[1] = 0;
      }
   }
   CUDA_EXPORT inline void push(const T &a) {
      THREAD;
      assert(bounds.size() >= 2);
      data.push_back(a);
      if (tid == 0) {
         bounds.back()++;}

   }
   CUDA_EXPORT inline int size() const {
      assert(bounds.size() >= 2);
      return end() - begin();
   }
   CUDA_EXPORT inline void resize(int sz) {
      THREAD;
      assert(bounds.size() >= 2);
      data.resize(begin() + sz,vectorPOD);
      if (tid == 0) {
         bounds.back() = data.size();
      }
   }
   CUDA_EXPORT inline T operator[](int i) const {
      assert(i < size());
      return data[begin() + i];
   }
   CUDA_EXPORT inline T& operator[](int i) {
      assert(i < size());
      return data[begin() + i];
   }
   CUDA_EXPORT inline stack_vector copy_top() const {
      THREAD;
      BLOCK;
      stack_vector res;
      res.resize(size());
      for (int i = tid; i < size(); i += blocksize) {
         res[i] = (*this)[i];
      }
      return res;
   }
   CUDA_EXPORT inline void push_top() {
      THREAD;
      BLOCK;
      const auto sz = size();
      bounds.push_back(end() + sz);
      data.resize(data.size() + sz,vectorPOD);
      for (int i = begin() + tid; i < end(); i += blocksize) {
         data[i] = data[i - sz];
      }
   }
   CUDA_EXPORT inline void pop_top() {
      assert(bounds.size() >= 2);
      data.resize(begin(),vectorPOD);
      bounds.pop_back();
   }
#ifndef __CUDACC__
   std::function<void()> to_device(cudaStream_t stream) {
      assert(bounds.size() >= 2);
      auto f1 = data.to_device(stream);
      auto f2 = bounds.to_device(stream);
      return [f1, f2]() {
         f2();
         f1();
      };
   }
#endif
};

