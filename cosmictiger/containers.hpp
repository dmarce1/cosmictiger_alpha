#pragma once

#include <cosmictiger/memory.hpp>
#include <cosmictiger/cuda.hpp>
#include <cassert>

#ifdef __CUDA_ARCH__
#define THREAD_INDEX \
      const int& tid = threadIdx.x
#else
#define THREAD_INDEX \
   const int tid = 0
#endif

#ifdef __CUDA_ARCH__
#define BLOCK_SIZE \
      const int& block_size = blockDim.x
#else
#define BLOCK_SIZE \
   const int block_size = 1
#endif

template<class T>
class vector {
   T *ptr_;
   size_t sz_;
   size_t cap_;
   CUDA_EXPORT
   inline void init() {
      THREAD_INDEX;
      if (tid == 0) {
         ptr_ = nullptr;
         sz_ = 0;
         cap_ = 0;
      }CUDA_SYNC();
   }
   CUDA_EXPORT
   inline void free() {
      THREAD_INDEX;
      BLOCK_SIZE;
      for (int i = tid; i < sz_; i += block_size) {
         ptr_[i].~T();
         ::operator delete(ptr_ + i);
      }
      CUDA_SYNC();
      if (ptr_ && tid == 0) {
         CUDA_FREE(ptr_);
      }CUDA_SYNC();

   }
public:

   constexpr inline CUDA_EXPORT vector() :
         ptr_(nullptr), sz_(0), cap_(0) {
   }
   inline CUDA_EXPORT vector(size_t sz) {
      init();
      resize(sz_);
   }
   inline CUDA_EXPORT vector(size_t sz, T init) {
      init();
      resize(sz_, init);
   }

   inline CUDA_EXPORT vector(size_t sz, T &&init) {
      THREAD_INDEX;
      BLOCK_SIZE;
      init();
      resize(sz_);
      for (int i = tid; i < sz_; i += block_size) {
         (*this)[i] = std::move(init);
      }
   }
   inline CUDA_EXPORT ~vector() {
      free();
   }
   inline CUDA_EXPORT vector(const vector &other) {
      init();
      *this = other;
   }
   inline CUDA_EXPORT vector(vector &&other) {
      init();
      *this = std::move(other);
   }
   inline CUDA_EXPORT void swap(vector &other) {
      THREAD_INDEX;
      if (tid == 0) {
         const size_t tmp_sz = sz_;
         const size_t tmp_cap = cap_;
         T *const tmp_ptr = ptr_;
         sz_ = other.sz_;
         cap_ = other.cap_;
         ptr_ = other.ptr_;
         other.sz_ = tmp_sz;
         other.cap_ = tmp_cap;
         other.ptr_ = tmp_ptr;
      }CUDA_SYNC();
   }
   inline CUDA_EXPORT vector& operator=(const vector &other) {
      THREAD_INDEX;
      BLOCK_SIZE;
      resize(other.size());
      for (int i = tid; i < other.size(); i += block_size) {
         (*this)[i] = other[i];
      }
      CUDA_SYNC();
      return *this;
   }

   inline CUDA_EXPORT vector& operator=(vector &&other) {
      THREAD_INDEX;
      free();
      if (tid == 0) {
         ptr_ = other.ptr_;
         sz_ = other.sz_;
         cap_ = other.cap_;
      }
      other.free();
      CUDA_SYNC();
      return *this;
   }

   CUDA_EXPORT
   void reserve(size_t cap) {
      THREAD_INDEX;
      BLOCK_SIZE;
      size_t new_cap = 1;
      while (new_cap < cap) {
         new_cap *= 2;
      }
      if (new_cap > cap_) {
         T *new_ptr;
         if (tid == 0) {
            cap_ = new_cap;
            CUDA_MALLOC(new_ptr, new_cap);
            if (new_cap < sz_) {
               sz_ = new_cap;
            }
         }
         CUDA_SYNC();
         for (int i = tid; i < sz_; i += block_size) {
            new_ptr[i] = std::move(ptr_[i]);
         }
         CUDA_SYNC();
         free();
         if (tid == 0) {
            ptr_ = new_ptr;
            cap_ = new_cap;
         }
      }
   }
   CUDA_EXPORT
   inline void resize(size_t new_size) {
      THREAD_INDEX;
      BLOCK_SIZE;
      for (int i = new_size + tid; i < sz_; i += block_size) {
         ptr_[i].~T();
      }
      CUDA_SYNC();
      reserve(new_size);
      CUDA_SYNC();
      for (int i = sz_ + tid; i < new_size; i += block_size) {
         new (ptr_ + i) T();
      }
      CUDA_SYNC();
      if (tid == 0) {
         sz_ = new_size;
      }CUDA_SYNC();
   }
   CUDA_EXPORT
   inline void resize(size_t new_size, const T &data) {
      THREAD_INDEX;
      BLOCK_SIZE;
      for (int i = new_size + tid; i < sz_; i += block_size) {
         ptr_[i].~T();
      }
      CUDA_SYNC();
      reserve(new_size);
      CUDA_SYNC();
      for (int i = sz_ + tid; i < new_size; i += block_size) {
         new (ptr_ + i) T(data);
      }
      CUDA_SYNC();
      if (tid == 0) {
         sz_ = new_size;
      }CUDA_SYNC();
   }
   CUDA_EXPORT
   inline void resize(size_t new_size, T &&data) {
      BLOCK_SIZE;
      THREAD_INDEX;
      for (int i = new_size + tid; i < sz_; i += block_size) {
         ptr_[i].~T();
      }
      CUDA_SYNC();
      reserve(new_size);
      CUDA_SYNC();
      for (int i = sz_ + tid; i < new_size; i += block_size) {
         new (ptr_ + i) T(std::move(data));
      }
      CUDA_SYNC();
      if (tid == 0) {
         sz_ = new_size;
      }CUDA_SYNC();
   }
   CUDA_EXPORT
   inline T operator[](int i) const {
      assert(i >= sz_);
      assert(i < sz_);
      return ptr_[i];
   }
   CUDA_EXPORT
   inline T& operator[](int i) {
      assert(i >= sz_);
      assert(i < sz_);
      return ptr_[i];
   }
   CUDA_EXPORT
   inline size_t size() const {
      return sz_;
   }
   CUDA_EXPORT
   void push_back(const T &data) {
      resize(size() + 1);
      ptr_[size() - 1] = data;
   }
   CUDA_EXPORT
   void push_back(T &&data) {
      resize(size() + 1);
      ptr_[size() - 1] = std::move(data);
   }
   CUDA_EXPORT
   void clear() {
      THREAD_INDEX;
      BLOCK_SIZE;
      for (int i = tid; i < sz_; i += block_size) {
         ptr_[i].~T();
      }
      sz_ = 0;
   }
   CUDA_EXPORT
   T back() const {
      return ptr_[size() - 1];
   }
   CUDA_EXPORT
   T& back() {
      return ptr_[size() - 1];
   }
   CUDA_EXPORT
   T front() const {
      return ptr_[0];
   }
   CUDA_EXPORT
   T& front() {
      return ptr_[0];
   }
};

template<class T, size_t N>
class array {
   T a[N];
public:
   CUDA_EXPORT
   inline T operator[](int i) const {
      assert(i >= 0);
      assert(i < N);
      return a[i];
   }
   CUDA_EXPORT
   inline T& operator[](int i) {
      assert(i >= 0);
      assert(i < N);
      return a[i];
   }
   CUDA_EXPORT
   inline constexpr size_t size() const {
      return N;
   }
   CUDA_EXPORT
   inline T* data() {
      return a;
   }
   CUDA_EXPORT
   inline const T* data() const {
      return a;
   }
};
template<class A, class B>
struct pair {
   A first;
   B second;
};
