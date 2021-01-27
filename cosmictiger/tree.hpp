#pragma once

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/thread_control.hpp>

#include <memory>

struct tree {
   struct sort_vars {
      std::vector<size_t> bounds;
      range box;
      int depth;
      template<class A>
      void serialize(A &&arc, unsigned) {
         arc & depth;
         arc & box;
         arc & bounds;
      }
   };
   struct id_type {
      uintptr_t ptr;
      int rank;
      inline constexpr id_type() :
            rank(-1), ptr(0) {
      }
      inline bool operator==(const id_type &other) {
         return rank == other.rank && ptr == other.ptr;
      }
      template<class A>
      void serialize(A &&arc, unsigned) {
         arc & rank;
         arc & ptr;
      }
   };
private:
   std::array<id_type, NCHILD> children;
   size_t parts_begin;
   size_t parts_end;
   static constexpr id_type invalid_id();
public:
   static hpx::future<tree::id_type> create(int,std::shared_ptr<tree::sort_vars> vars);
   tree();
   tree (std::shared_ptr<sort_vars>);
};
