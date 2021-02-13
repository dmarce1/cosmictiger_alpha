/*
 * vector.hpp
 *
 *  Created on: Feb 11, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_VECTOR_HPP_
#define COSMICTIGER_VECTOR_HPP_

#ifdef __CUDA_ARCH__
#define BLOCK const int& blocksize = blockDim.x
#define THREAD const int& tid = threadIdx.x
#define SYNC __syncthreads()
#else
#define BLOCK constexpr int blocksize = 1
#define THREAD constexpr int tid = 0
#define SYNC
#endif

#include <functional>
#include <cosmictiger/memory.hpp>

template<class T>
class vector {
   T *ptr;
   size_t cap;
   size_t sz;
   bool dontfree;
   T *new_ptr;CUDA_EXPORT
   inline
   void destruct(size_t b, size_t e) {
      THREAD;
      BLOCK;
      SYNC;
      for (size_t i = b + tid; i < e; i += blocksize) {
         (*this)[i].T::~T();
      }SYNC;
   }
   CUDA_EXPORT
   inline
   void construct(size_t b, size_t e) {
      THREAD;
      BLOCK;
      SYNC;
      for (size_t i = b + tid; i < e; i += blocksize) {
         new (ptr + i) T();
      }SYNC;
   }
public:
#ifndef __CUDACC__
   std::function<void()> to_device(cudaStream_t stream) {
      assert(cap);
      cuda_allocator allocator;
    //  printf( "allocating\n");
      new_ptr = (T*) allocator.allocate(cap*sizeof(T));
//      static std::atomic<size_t> alloced(0);
  //    alloced += cap * sizeof(T);
    //  printf( " to device %li\n", (size_t) alloced);
      CHECK_POINTER(new_ptr);
    //  printf( "1\n");
      if (sz) {
         assert(ptr);
         assert(sz <= cap);
         CUDA_CHECK(cudaMemcpyAsync(new_ptr, ptr, sizeof(T) * sz, cudaMemcpyHostToDevice, stream));
      }
    //  printf( "2\n");
      auto* dptr = ptr;
      auto sz_ = sz;
      auto new_ptr_ = new_ptr;
      auto cap_ = cap;
      auto func = [dptr, sz_, new_ptr_]() {
         cuda_allocator allocator;
         if (dptr) {
            for( size_t i = 0; i < sz_; i++) {
               dptr[i].T::~T();
            }
            allocator.deallocate(new_ptr_);
   //         alloced -= cap_ * sizeof(T);
            CUDA_FREE(dptr);
         }
      };
   //   printf( "3\n");
      dontfree = true;
      ptr = new_ptr;
      return func;
   }
#endif
   CUDA_EXPORT inline vector() {
      THREAD;
      SYNC;
      if (tid == 0) {
         dontfree = false;
         ptr = nullptr;
         cap = 0;
         sz = 0;
      }SYNC;
   }
   CUDA_EXPORT inline vector(const vector &other) {
      THREAD;
      BLOCK;
      SYNC;
      if (tid == 0) {
         dontfree = false;
         sz = 0;
         ptr = nullptr;
         cap = 0;
      }SYNC;
      reserve(other.cap);
      if (tid == 0) {
         sz = other.sz;
         cap = other.sz;
      }SYNC;
      construct(0, other.sz);
      SYNC;
      for (size_t i = tid; i < other.sz; i += blocksize) {
         (*this)[i] = other[i];
      }SYNC;
   }
   CUDA_EXPORT
   inline vector& operator=(const vector &other) {
      THREAD;
      BLOCK;
      reserve(other.cap);
      resize(other.size());
      for (int i = tid; i < other.size(); i += blocksize) {
         (*this)[i] = other[i];
      }
      return *this;
   }
   CUDA_EXPORT
   inline vector& operator=(vector &&other) {
      THREAD;
      if (tid == 0) {
         ptr = other.ptr;
         sz = other.sz;
         cap = other.cap;
         dontfree = other.dontfree;
         other.ptr = nullptr;
         other.sz = 0;
         other.cap = 0;
         other.dontfree = false;
      }SYNC;
      return *this;
   }
   CUDA_EXPORT inline vector(vector &&other) {
      THREAD;
      SYNC;
      if (tid == 0) {
         dontfree = other.dontfree;
         ptr = other.ptr;
         sz = other.sz;
         cap = other.cap;
         other.ptr = nullptr;
         other.sz = 0;
         other.cap = 0;
         other.dontfree = false;
      }SYNC;
   }
   CUDA_EXPORT
   inline
   void reserve(size_t new_cap) {
      THREAD;
      BLOCK;
      size_t i = 256;
      while (i < new_cap ) {
         i *= 2;
      }
      new_cap = i;
      if (new_cap > cap) {
#ifdef __CUDA_ARCH__
//        printf( "INcreasing capacity from %li to %li\n", cap, new_cap);
#endif
         SYNC;
         if (tid == 0) {
            CUDA_MALLOC(new_ptr, new_cap);
         }SYNC;
         for (size_t i = tid; i < sz; i += blocksize) {
            new (new_ptr + i) T();
            new_ptr[i] = std::move((*this)[i]);
         }
         destruct(0, sz);
         if (tid == 0) {
            cap = new_cap;
            if (ptr && !dontfree) {
               CUDA_FREE(ptr);
            }
            dontfree = false;
            ptr = new_ptr;
         }SYNC;
      }
   }
   CUDA_EXPORT
   inline
   void resize(size_t new_size) {
      THREAD;
      reserve(new_size);
      auto oldsz = sz;
      SYNC;
      destruct(new_size, oldsz);
      if (tid == 0) {
         sz = new_size;
      }SYNC;
      construct(oldsz, new_size);
      SYNC;
   }
   CUDA_EXPORT
   inline T operator[](size_t i) const {
      assert(i < sz);
      return ptr[i];
   }
   CUDA_EXPORT
   inline T& operator[](size_t i) {
      assert(i < sz);
      return ptr[i];
   }
   CUDA_EXPORT
   inline size_t size() const {
      return sz;
   }
   CUDA_EXPORT
   inline
   void push_back(const T &dat) {
      THREAD;
      resize(size() + 1);
      SYNC;
      if (tid == 0) {
         ptr[size() - 1] = dat;
      }SYNC;
   }
   CUDA_EXPORT
   inline
   void push_back(T &&dat) {
      THREAD;
      resize(size() + 1);
      SYNC;
      if (tid == 0) {
         ptr[size() - 1] = std::move(dat);
      }SYNC;
   }
   CUDA_EXPORT
   inline T* data() {
      return ptr;
   }
   CUDA_EXPORT
   inline const T& data() const {
      return ptr;
   }
   CUDA_EXPORT inline ~vector() {
      THREAD;
      SYNC;
      destruct(0, sz);
      SYNC;
      if (tid == 0 && ptr && !dontfree) {
         CUDA_FREE(ptr);
      }SYNC;

   }
   CUDA_EXPORT inline void pop_back() {
      assert(size());
      resize(size()-1);
   }
   CUDA_EXPORT
   inline T back() const {
      return ptr[size() - 1];
   }
   CUDA_EXPORT
   inline T& back() {
      return ptr[size() - 1];
   }
   CUDA_EXPORT
   inline
   void swap(vector &other) {
      THREAD;
      if (tid == 0) {
         auto tmp1 = sz;
         auto tmp2 = cap;
         auto tmp3 = ptr;
         auto tmp4 = dontfree;
         sz = other.sz;
         cap = other.cap;
         ptr = other.ptr;
         dontfree = other.dontfree;
         other.sz = tmp1;
         other.cap = tmp2;
         other.ptr = tmp3;
         other.dontfree = tmp4;
      }SYNC;
   }
};

#endif /* COSMICTIGER_VECTOR_HPP_ */
