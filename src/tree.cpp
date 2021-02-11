#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>

#include <cmath>

#define KICK_MAX_MULTIS 1024
#define KICK_MAX_PARTS 1024
#define KICK_MAX_CHECKS 1024
particle_set *tree::particles;

static finite_vector_allocator<sizeof(kick_params_type)> kick_params_alloc;

void tree::set_particle_set(particle_set *parts) {
   particles = parts;
}

std::atomic<int> tree::cuda_node_count(0);
std::atomic<int> tree::cpu_node_count(0);

inline auto cuda_depth() {
   return int(log(global().cuda.devices[0].multiProcessorCount * OVERSUBSCRIPTION) / log(2));
}

inline hpx::future<sort_return> tree::create_child(sort_params &params) {
   static std::atomic<int> threads_used(hpx_rank() == 0 ? 1 : 0);
   tree_ptr id;
   id.rank = 0;
   id.ptr = (uintptr_t) params.allocs->tree_alloc.allocate();
   CHECK_POINTER(id.ptr);
   const auto nparts = (*params.bounds)[params.key_end] - (*params.bounds)[params.key_begin];
   bool thread = false;
   if (nparts > TREE_MIN_PARTS2THREAD) {
      if (++threads_used <= hpx::thread::hardware_concurrency()) {
         thread = true;
      } else {
         threads_used--;
      }
   }
#ifdef TEST_STACK
   thread = false;
#endif
   if (!thread) {
      sort_return rc = ((tree*) (id.ptr))->sort(params);
      rc.check = id;
      return hpx::make_ready_future(std::move(rc));
   } else {
      params.allocs = std::make_shared<tree_alloc>();
      return hpx::async([id, params]() {
         auto rc = ((tree*) (id.ptr))->sort(params);
         rc.check = id;
         threads_used--;
         return rc;
      });
   }
}

sort_return tree::sort(sort_params params) {
   const auto &opts = global().opts;
   if (params.iamroot()) {
      params.set_root();
   }
   {
      const auto bnds = params.get_bounds();
      parts.first = bnds.first;
      parts.second = bnds.second;
   }
   if (params.depth == TREE_MAX_DEPTH) {
      printf("Exceeded maximum tree depth\n");
      abort();
   }

   multi = params.allocs->multi_alloc.allocate();
   if (params.depth == cuda_depth()) {
      cuda_node_count++;
   } else if (params.depth < cuda_depth()) {
      cpu_node_count++;
   }
#ifdef TEST_TREE
   const auto &box = params.box;
   bool failed = false;
   for (size_t i = parts.first; i < parts.second; i++) {
      particle p = particles->part(i);
      if (!box.contains(p.x)) {
         printf("Particle out of range !\n");
         printf("Box\n");
         for (int dim = 0; dim < NDIM; dim++) {
            printf("%e %e |", box.begin[dim].to_float(), box.end[dim].to_float());
         }
         printf("\n");
         printf("Particle\n");
         for (int dim = 0; dim < NDIM; dim++) {
            printf("%e ", p.x[dim].to_float());
         }
         printf("\n");
         //       abort();
         failed = true;
      }
   }
   if (failed) {
      // abort();
   }
#endif
#ifdef TEST_STACK
   {
      uint8_t dummy;
      printf("Stack usaged = %li Depth = %li \n", &dummy - params.stack_ptr, params.depth);
   }
#endif
   if (parts.second - parts.first > opts.bucket_size) {
      std::array<fast_future<sort_return>, NCHILD> futs;
      {
         const auto size = parts.second - parts.first;
         auto child_params = params.get_children();
         if (params.key_end - params.key_begin == 1) {
#ifndef TEST_TREE
            const auto &box = params.box;
#endif
            int radix_depth = (int(log(double(size) / opts.bucket_size) / log(2) + TREE_RADIX_CUSHION));
            radix_depth = std::min(std::max(radix_depth, TREE_RADIX_MIN), TREE_RADIX_MAX) + params.depth;
            const auto radix_begin = morton_key(box.begin, radix_depth);
            std::array<fixed64, NDIM> tmp;
            for (int dim = 0; dim < NDIM; dim++) {
               tmp[dim] = box.end[dim] - fixed32::min();
            }
            const auto radix_end = morton_key(tmp, radix_depth) + 1;
            auto bounds = particles->local_sort(parts.first, parts.second, radix_depth, radix_begin, radix_end);
            assert(bounds[0] >= parts.first);
            assert(bounds[bounds.size() - 1] <= parts.second);
            auto bndptr = std::make_shared<decltype(bounds)>(std::move(bounds));
            for (int ci = 0; ci < NCHILD; ci++) {
               child_params[ci].bounds = bndptr;
            }
            child_params[LEFT].key_begin = 0;
            child_params[LEFT].key_end = child_params[RIGHT].key_begin = (radix_end - radix_begin) / 2;
            child_params[RIGHT].key_end = (radix_end - radix_begin);
         }
         for (int ci = 0; ci < NCHILD; ci++) {
            futs[ci] = create_child(child_params[ci]);
         }
      }
      std::array<multipole*, NCHILD> Mc;
      std::array<fixed32*, NCHILD> Xc;
      std::array<float, NCHILD> Rc;
      auto &M = *(multi);
      for (int ci = 0; ci < NCHILD; ci++) {
         sort_return rc = futs[ci].get();
         children[ci] = rc.check;
         Mc[ci] = ((tree*) rc.check)->multi;
         Xc[ci] = ((tree*) rc.check)->pos.data();
         children[ci] = rc.check;
      }
      std::array<double, NDIM> com = { 0, 0, 0 };
      const auto &MR = *Mc[RIGHT];
      const auto &ML = *Mc[LEFT];
      M() = ML() + MR();
      double rleft = 0.0;
      double rright = 0.0;
      for (int dim = 0; dim < NDIM; dim++) {
         com[dim] = (ML() * Xc[LEFT][dim].to_double() + MR() * Xc[RIGHT][dim].to_double()) / (ML() + MR());
         pos[dim] = com[dim];
         rleft += sqr(Xc[LEFT][dim].to_double() - com[dim]);
         rright += sqr(Xc[RIGHT][dim].to_double() - com[dim]);
      }
      std::array<double, NDIM> xl, xr;
      for (int dim = 0; dim < NDIM; dim++) {
         xl[dim] = Xc[LEFT][dim].to_double() - com[dim];
         xr[dim] = Xc[RIGHT][dim].to_double() - com[dim];
      }
      M = (ML >> xl) + (MR >> xr);
      rleft = std::sqrt(rleft) + ((tree*) children[LEFT])->radius;
      rright = std::sqrt(rright) + ((tree*) children[RIGHT])->radius;
      radius = std::max(rleft, rright);
      float rmax = 0.0;
      const auto corners = params.box.get_corners();
      for (int ci = 0; ci < NCORNERS; ci++) {
         double d = 0.0;
         for (int dim = 0; dim < NDIM; dim++) {
            d += sqr(com[dim] - corners[ci][dim].to_double());
         }
         rmax = std::max((float) std::sqrt(d), rmax);
      }
      radius = std::min(radius, rmax);
      //    printf("x      = %e\n", pos[0].to_float());
      //   printf("y      = %e\n", pos[1].to_float());
      //  printf("z      = %e\n", pos[2].to_float());
      // printf("radius = %e\n", radius);
   } else {
      std::array<double, NDIM> com = { 0, 0, 0 };
      for (auto i = parts.first; i < parts.second; i++) {
         for (int dim = 0; dim < NDIM; dim++) {
            com[dim] += particles->pos(dim, i).to_double();
         }
      }
      for (int dim = 0; dim < NDIM; dim++) {
         com[dim] /= (parts.second - parts.first);
         pos[dim] = com[dim];

      }
      auto &M = *(multi);
      M = 0.0;
      radius = 0.0;
      for (auto i = parts.first; i < parts.second; i++) {
         double this_radius = 0.0;
         M() += 1.0;
         for (int n = 0; n < NDIM; n++) {
            const auto xn = particles->pos(n, i).to_double() - com[n];
            this_radius += xn * xn;
            for (int m = n; m < NDIM; m++) {
               const auto xm = particles->pos(m, i).to_double() - com[m];
               const auto xnm = xn * xm;
               M(n, m) += xnm;
               for (int l = m; l > NDIM; l++) {
                  const auto xl = particles->pos(l, i).to_double() - com[l];
                  M(n, m, l) -= xnm * xl;
               }
            }
         }
         this_radius = std::sqrt(this_radius);
         radius = std::max(radius, (float) (this_radius));
      }
      parts.first = parts.first;
      parts.second = parts.second;
   }
   sort_return rc;
   return rc;
}

hpx::lcos::local::mutex tree::mtx;
hpx::lcos::local::mutex tree::gpu_mtx;

void tree::cleanup() {
   shutdown_daemon = true;
   while (daemon_running) {
      hpx::this_thread::yield();
   }
}

fast_future<kick_return> tree_ptr::kick(kick_params_type *params_ptr, bool try_thread) {
   kick_params_type &params = *params_ptr;
   const auto part_begin = ((tree*) (*this))->parts.first;
   const auto part_end = ((tree*) (*this))->parts.second;
   //  printf( "%i\n", cuda_depth);
   if (params_ptr->depth == cuda_depth()) {
      return fast_future<kick_return>(((tree*) ptr)->send_kick_to_gpu(params_ptr));
   } else {
      static std::atomic<int> thread_cnt(hpx_rank() == 0 ? 1 : 0);
      bool thread = false;
#ifndef TEST_CHECKLIST_TIME
      if (try_thread) {
         int cnt = thread_cnt++;
         if (cnt < hpx::thread::hardware_concurrency() * OVERSUBSCRIPTION) {
            thread = true;
         } else {
            thread_cnt--;
         }
         thread = true;
      }
#endif
      if (!thread) {
         return fast_future<kick_return>(((tree*) ptr)->kick(params_ptr));
      } else {
         kick_params_type *new_params;
         new_params = (kick_params_type*) kick_params_alloc.allocate();
         new (new_params) kick_params_type;
         new_params->dstack.copy_to(params_ptr->dstack.get_top_list(), params_ptr->dstack.get_top_count());
         new_params->estack.copy_to(params_ptr->estack.get_top_list(), params_ptr->estack.get_top_count());
         new_params->L[params_ptr->depth] = params_ptr->L[params_ptr->depth];
         new_params->Lpos[params_ptr->depth] = params_ptr->Lpos[params_ptr->depth];
         new_params->depth = params_ptr->depth;
         new_params->theta = params_ptr->theta;
         new_params->eta = params_ptr->eta;
         new_params->scale = params_ptr->scale;
         new_params->t0 = params_ptr->t0;
         new_params->rung = params_ptr->rung;
         auto func = [this, new_params]() {
            auto rc = ((tree*) ptr)->kick(new_params);
            thread_cnt--;
            kick_params_alloc.deallocate(new_params);
            return rc;
         };
         auto fut = hpx::async(std::move(func));
         return fast_future<kick_return>(std::move(fut));
      }
   }
}

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

//int num_kicks = 0;
hpx::future<kick_return> tree::kick(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;

#ifdef TEST_CHECKLIST_TIME
   static timer tm;
#endif
   kick_return rc;
   rc.flops = 0;
   auto &L = params.L[params.depth];
   const auto &Lpos = params.Lpos[params.depth];
   array<accum_real, NDIM> dx;
   for (int dim = 0; dim < NDIM; dim++) {
#ifdef ACCUMULATE_DOUBLE_PRECISION
      const auto x1 = pos[dim].to_double();
      const auto x2 = Lpos[dim].to_double();
#else
      const auto x1 = pos[dim].to_float();
      const auto x2 = Lpos[dim].to_float();
#endif
      dx[dim] = x1 - x2;
   }
   L <<= dx;

   const auto theta2 = params.theta * params.theta;
   array<tree_ptr*, N_INTERACTION_TYPES> all_checks;
   array<int*, N_INTERACTION_TYPES> list_counts;
   all_checks[CC_CP_DIRECT] = params.dstack.get_top_list();
   all_checks[CC_CP_EWALD] = params.estack.get_top_list();
   all_checks[PC_PP_DIRECT] = params.dstack.get_top_list();
   all_checks[PC_PP_EWALD] = params.estack.get_top_list();
   list_counts[CC_CP_DIRECT] = &params.dstack.get_top_count();
   list_counts[CC_CP_EWALD] = &params.estack.get_top_count();
   list_counts[PC_PP_DIRECT] = &params.dstack.get_top_count();
   list_counts[PC_PP_EWALD] = &params.estack.get_top_count();
   auto &multis = params.multi_interactions;
   auto &parti = params.part_interactions;
   auto &next_checks = params.next_checks;
   auto &opened_checks = params.opened_checks;
   int ninteractions = is_leaf() ? 4 : 2;
   for (int type = 0; type < ninteractions; type++) {
      auto *checks = (all_checks[type]);
      const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
      const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
      params.npart = params.nmulti = params.nnext = 0;

      do {
         params.nnext = params.nopen = 0;
#ifdef TEST_CHECKLIST_TIME
            tm.start();
#endif
         for (int ci = 0; ci < *(list_counts[type]); ci++) {
            const auto other_radius = checks[ci].get_radius();
            const auto other_pos = checks[ci].get_pos();
            float d2 = 0.f;
            const float R2 = sqr(other_radius + radius);
            for (int dim = 0; dim < NDIM; dim++) {
               d2 += sqr(fixed<int32_t>(other_pos[dim]) - fixed<int32_t>(pos[dim])).to_float();
            }
            if (ewald_dist) {
               d2 = std::max(d2, EWALD_MIN_DIST2);
            }
            const bool far = R2 < theta2 * d2;
            if (far) {
               multis[params.nmulti++] = checks[ci];
            } else if (!checks[ci].is_leaf()) {
               const auto child_checks = checks[ci].get_children().get();
               next_checks[params.nnext++] = child_checks[LEFT];
               next_checks[params.nnext++] = child_checks[RIGHT];
            } else if (!checks[ci].opened++) {
               next_checks[params.nnext++] = checks[ci];
            } else {
               parti[params.npart++] = checks[ci];
            }
         }
#ifdef TEST_CHECKLIST_TIME
         tm.stop();
#endif
         if (type == PC_PP_DIRECT || type == CC_CP_DIRECT) {
            params.dstack.pop();
            params.dstack.copy_to(next_checks.data(), params.nnext);
         } else {
            params.estack.pop();
            params.estack.copy_to(next_checks.data(), params.nnext);
         }
      } while (direct && *(list_counts[type]));

      switch (type) {
      case CC_CP_DIRECT:
         rc.flops += cpu_cc_direct(params_ptr);
         break;
      case CC_CP_EWALD:
         if (params_ptr->nmulti) {
            auto tmp = send_ewald_to_gpu(params_ptr).get();
            //     printf( "%i\n", tmp);
            rc.flops += tmp;
         } else {
            if (params.depth < cuda_depth()) {
               cpu_node_count--;
            }
         }
         break;
      case PC_PP_DIRECT:
         break;
      case PC_PP_EWALD:
         break;
      }

   }
   if (!is_leaf()) {
      // printf("4\n");
      const bool try_thread = parts.second - parts.first > TREE_MIN_PARTS2THREAD;
      params.dstack.copy_top();
      params.estack.copy_top();
      params.depth++;
      params.L[params.depth] = L;
      params.Lpos[params.depth] = pos;
      array<hpx::future<kick_return>, NCHILD> futs;
      futs[LEFT] = children[LEFT].kick(params_ptr, try_thread);
      //  printf("5\n");
      params.dstack.pop();
      params.estack.pop();
      params.L[params.depth] = L;
      futs[RIGHT] = children[RIGHT].kick(params_ptr, false);
      params.depth--;
      //  printf("6\n");
      return hpx::when_all(futs.begin(), futs.end()).then(
            [rc](hpx::future<std::vector<hpx::future<kick_return>>> futs) {
               auto tmp = futs.get();
               auto rc1 = tmp[LEFT].get();
               auto rc2 = tmp[RIGHT].get();
               kick_return rc3 = rc;
               rc3.rung = std::max(rc1.rung, rc2.rung);
               rc3.flops += rc1.flops + rc2.flops;
               return rc3;
            });
   } else {
      rc.rung = 0;
      return hpx::make_ready_future(rc);
   }
#ifdef TEST_CHECKLIST_TIME
   if (depth == 0) {
      printf("Checklist time = %e\n", tm.read());
   }
#endif
}

lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> tree::gpu_queue;
lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> tree::gpu_ewald_queue;
std::atomic<bool> tree::daemon_running(false);
std::atomic<bool> tree::shutdown_daemon(false);

using cuda_exec_return =
std::pair<std::function<bool()>, kick_return*>;

cuda_exec_return cuda_execute_kick_kernel(kick_params_type **params, int grid_size);
void cuda_execute_ewald_cc_kernel(kick_params_type *params_ptr);

void tree::gpu_daemon() {
   using namespace std::chrono_literals;

   printf("Starting gpu kick daemon\n");
   daemon_running = true;
   int eminsize = 1;
   int cminsize = 1;
   while (!shutdown_daemon) {
      hpx::this_thread::yield();
      int grid_size = std::min(eminsize, (int) cpu_node_count);
      while (gpu_ewald_queue.size() >= grid_size && gpu_ewald_queue.size()) {
         std::vector < hpx::lcos::local::promise < int32_t >> promises;
         static finite_vector_allocator<sizeof(kick_params_type*) * KICK_GRID_SIZE> all_params_alloc;
         kick_params_type **all_params = (kick_params_type**) all_params_alloc.allocate();
         cpu_node_count -= grid_size;
         for (int i = 0; i < grid_size; i++) {
            auto tmp = gpu_ewald_queue.pop();
            all_params[i] = tmp.params;
            promises.push_back(std::move(tmp.promise));
         }
         printf("Executing %i ewald blocks\n", grid_size);
         auto exec_ret = cuda_execute_ewald_kernel(all_params, grid_size);
         eminsize = std::min(2*eminsize, KICK_GRID_SIZE);
         grid_size = std::max(std::min(eminsize, (int) cpu_node_count),1);
         hpx::apply([all_params, exec_ret, grid_size](std::vector<hpx::lcos::local::promise<int32_t>> &&promises) {
            while (!exec_ret()) {
               hpx::this_thread::yield();
            }
            for (int i = 0; i < grid_size; i++) {
               promises[i].set_value(all_params[i]->flops);
            }
            all_params_alloc.deallocate(all_params);
         },std::move(promises));
      }
      while (gpu_queue.size() >= std::min((int) cuda_node_count, KICK_GRID_SIZE) && cuda_node_count) {
         int grid_size = std::min((int) cuda_node_count, KICK_GRID_SIZE);
         cuda_node_count -= grid_size;
         std::vector < hpx::lcos::local::promise < kick_return >> promises;
         static finite_vector_allocator<sizeof(kick_params_type*) * KICK_GRID_SIZE> all_params_alloc;
         kick_params_type **all_params = (kick_params_type**) all_params_alloc.allocate();
         for (int i = 0; i < grid_size; i++) {
            auto tmp = gpu_queue.pop();
            all_params[i] = tmp.params;
            promises.push_back(std::move(tmp.promise));
         }
         printf("Executing %i blocks\n", grid_size);
         auto exec_ret = cuda_execute_kick_kernel(all_params, grid_size);
         hpx::apply([all_params, exec_ret, grid_size](std::vector<hpx::lcos::local::promise<kick_return>> &&promises) {
            while (!exec_ret.first()) {
               hpx::this_thread::yield();
            }
            for (int i = 0; i < grid_size; i++) {
               promises[i].set_value(exec_ret.second[i]);
               all_params[i]->kick_params_type::~kick_params_type();
               kick_params_alloc.deallocate(all_params[i]);
            }
            CUDA_FREE(exec_ret.second);
            all_params_alloc.deallocate(all_params);
         },std::move(promises));
      }
   }
   daemon_running = false;
   shutdown_daemon = false;
   printf("Shutting down GPU kick daemon\n");
}

hpx::future<kick_return> tree::send_kick_to_gpu(kick_params_type *params) {
   if (!daemon_running) {
      std::lock_guard < hpx::lcos::local::mutex > lock(gpu_mtx);
      if (!daemon_running) {
         shutdown_daemon = false;
         daemon_running = true;
         hpx::apply([]() {
            gpu_daemon();
         });
      }
   }
   gpu_kick gpu;
   tree_ptr me;
   me.ptr = (uintptr_t)(this);
   me.rank = hpx_rank();
   kick_params_type *new_params;
   new_params = (kick_params_type*) kick_params_alloc.allocate();
   new (new_params) kick_params_type();
   new_params->tptr = me;
   new_params->dstack.copy_to(params->dstack.get_top_list(), params->dstack.get_top_count());
   new_params->estack.copy_to(params->estack.get_top_list(), params->estack.get_top_count());
   new_params->L[params->depth] = params->L[params->depth];
   new_params->Lpos[params->depth] = params->Lpos[params->depth];
   new_params->depth = params->depth;
   new_params->theta = params->theta;
   new_params->eta = params->eta;
   new_params->scale = params->scale;
   new_params->t0 = params->t0;
   new_params->rung = params->rung;
   gpu.params = new_params;
   auto fut = gpu.promise.get_future();
   gpu_queue.push(std::move(gpu));

   return std::move(fut);

}

hpx::future<int32_t> tree::send_ewald_to_gpu(kick_params_type *params) {
   if (!daemon_running) {
      std::lock_guard < hpx::lcos::local::mutex > lock(gpu_mtx);
      if (!daemon_running) {
         shutdown_daemon = false;
         daemon_running = true;
         hpx::apply([]() {
            gpu_daemon();
         });
      }
   }
   gpu_ewald gpu;
   tree_ptr me;
   me.ptr = (uintptr_t)(this);
   me.rank = hpx_rank();
   params->tptr = me;
   gpu.params = params;
   auto fut = gpu.promise.get_future();
   gpu_ewald_queue.push(std::move(gpu));

   return std::move(fut);

}

int tree::cpu_cc_direct(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   auto &L = params.L[params.depth];
   const auto &multis = params.multi_interactions;
   int nmulti = params.nmulti;
   int flops = 0;
   if (nmulti != 0) {
      static const auto one = simd_float(1.0);
      static const auto half = simd_float(0.5);
      array<simd_fixed32, NDIM> X, Y;
      multipole_type<simd_float> M;
      expansion<simd_float> Lacc;
      Lacc = simd_float(0);
      const auto cnt1 = nmulti;
      const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
      nmulti = cnt2;
      for (int dim = 0; dim < NDIM; dim++) {
         X[dim] = simd_fixed32(pos[dim]);
      }
      array<simd_float, NDIM> dX;
      for (int j = 0; j < cnt1; j += simd_float::size()) {
         for (int k = 0; k < simd_float::size(); k++) {
            if (j + k < cnt1) {
               for (int dim = 0; dim < NDIM; dim++) {
                  Y[dim][k] = ((const tree*) multis[j + k])->pos[dim];
               }
               for (int i = 0; i < MP; i++) {
                  M[i][k] = (*((const tree*) multis[j + k])->multi)[i];
               }
            } else {
               for (int dim = 0; dim < NDIM; dim++) {
                  Y[dim][k] = ((const tree*) multis[cnt1 - 1])->pos[dim];
               }
               for (int i = 0; i < MP; i++) {
                  M[k][i] = 0.0;
               }
            }
         }
         for (int dim = 0; dim < NDIM; dim++) {
            dX[dim] = (simd_fixed32(X[dim]) - simd_fixed32(Y[dim])).to_float();
         }
         auto tmp = multipole_interaction(Lacc, M, dX, false);
         if (j == 0) {
            flops = 3 + tmp;
         }
      }
      flops *= cnt1;
      for (int i = 0; i < LP; i++) {
         L[i] += Lacc[i].sum();
         flops++;
      }
   }
   return flops;
}

