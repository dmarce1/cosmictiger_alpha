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
#define FENCE() __threadfence_block()
#define SYNC() __syncwarp()
#else
#define BLOCK constexpr int blocksize = 1
#define THREAD constexpr int tid = 0
#define SYNC()
#define FENCE()
#endif

#include <functional>
#include <cosmictiger/memory.hpp>

template<class T>
class vector {
   T *ptr;
   int cap;
   int sz;
   bool dontfree;
   T *new_ptr;
   CUDA_EXPORT
   inline
   void destruct(int b, int e) {
      THREAD;
      BLOCK;
      for (int i = b + tid; i < e; i += blocksize) {
         (*this)[i].T::~T();
      }
   }
   CUDA_EXPORT
   inline
   void construct(int b, int e) {
      THREAD;
      BLOCK;
      for (int i = b + tid; i < e; i += blocksize) {
         new (ptr + i) T();
      }
   }
public:
#ifndef __CUDACC__
   std::function<void()> to_device(cudaStream_t stream) {
      assert(cap);
      dontfree = true;
      if (ptr) {
   //      CUDA_CHECK(cudaMemPrefetchAsync(ptr, sizeof(T) * sz, 0, stream));
         auto *dptr = ptr;
         auto sz_ = sz;
         auto func = [dptr, sz_]() {
            unified_allocator alloc;
            if (dptr) {
               for (int i = 0; i < sz_; i++) {
                  dptr[i].T::~T();
               }
               alloc.deallocate(dptr);
            }
         };
         //   printf( "3\n");
         //     ptr = new_ptr;
         return func;
      } else {
         return []() {
         };
      }
   }
#endif
   CUDA_EXPORT inline vector() {
      THREAD;
      if (tid == 0) {
         dontfree = false;
         ptr = nullptr;
         cap = 0;
         sz = 0;
      }
   }
   CUDA_EXPORT inline vector(const vector &other) {
      THREAD;
      BLOCK;
      if (tid == 0) {
         dontfree = false;
         sz = 0;
         ptr = nullptr;
         cap = 0;
      };
      reserve(other.cap);
      if (tid == 0) {
         sz = other.sz;
         cap = other.sz;
      }
      construct(0, other.sz);

      for (int i = tid; i < other.sz; i += blocksize) {
         (*this)[i] = other[i];
      }
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
         if (ptr && !dontfree) {
#ifndef __CUDA_ARCH__
            unified_allocator alloc;
            alloc.deallocate(ptr);
#else
            CUDA_FREE(ptr);
#endif
         }
         ptr = other.ptr;
         sz = other.sz;
         cap = other.cap;
         dontfree = other.dontfree;
         other.ptr = nullptr;
         other.sz = 0;
         other.cap = 0;
         other.dontfree = false;
      }
      return *this;
   }
   CUDA_EXPORT inline vector(vector &&other) {
      THREAD;
      if (tid == 0) {
         dontfree = other.dontfree;
         ptr = other.ptr;
         sz = other.sz;
         cap = other.cap;
         other.ptr = nullptr;
         other.sz = 0;
         other.cap = 0;
         other.dontfree = false;
      }
   }
   CUDA_EXPORT
   inline
   void reserve(int new_cap) {
      THREAD;
      BLOCK;
      if (new_cap > cap) {
         int i = 1;
         while (i < new_cap) {
            i *= 2;
         }
         new_cap = i;
#ifdef __CUDA_ARCH__
        printf( "INcreasing capacity from %i to %i\n", cap, new_cap);
#endif
         if (tid == 0) {
#ifndef __CUDA_ARCH__
            unified_allocator alloc;
            new_ptr = (T*) alloc.allocate(new_cap * sizeof(T));
#else
            CUDA_MALLOC(new_ptr, new_cap);
#endif
         }
         SYNC();
         for (int i = tid; i < sz; i += blocksize) {
            new (new_ptr + i) T();
            new_ptr[i] = std::move((*this)[i]);
         }
         destruct(0, sz);
         SYNC();
         if (tid == 0) {
            cap = new_cap;
            if (ptr && !dontfree) {
#ifndef __CUDA_ARCH__
               unified_allocator alloc;
               alloc.deallocate(ptr);
#else
               CUDA_FREE(ptr);
#endif
            }
            dontfree = false;
            ptr = new_ptr;
         }
      }
   }
   CUDA_EXPORT
   inline
   void resize(int new_size) {
      THREAD;
      reserve(new_size);
      auto oldsz = sz;
      destruct(new_size, oldsz);
      if (tid == 0) {
         sz = new_size;
      }
      construct(oldsz, new_size);

   }
   CUDA_EXPORT
   inline T operator[](int i) const {
      assert(i < sz);
      return ptr[i];
   }
   CUDA_EXPORT
   inline T& operator[](int i) {
      assert(i < sz);
      return ptr[i];
   }
   CUDA_EXPORT
   inline int size() const {
      return sz;
   }
   CUDA_EXPORT
   inline
   void push_back(const T &dat) {
      THREAD;
      resize(size() + 1);
      if (tid == 0) {
         ptr[size() - 1] = dat;
      }
   }
   CUDA_EXPORT
   inline
   void push_back(T &&dat) {
      THREAD;
      resize(size() + 1);
      if (tid == 0) {
         ptr[size() - 1] = std::move(dat);
      }
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
      //    printf( "destroying\n");
      THREAD;
      destruct(0, sz);
      if (tid == 0 && ptr && !dontfree) {
#ifndef __CUDA_ARCH__
         unified_allocator alloc;
         alloc.deallocate(ptr);
#else
         CUDA_FREE(ptr);
#endif
      }
   }
   CUDA_EXPORT inline void pop_back() {
      assert(size());
      resize(size() - 1);
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
      }
   }
};

#endif /* COSMICTIGER_VECTOR_HPP_ */
