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
float tree::theta;
int8_t tree::rung;

void tree::set_particle_set(particle_set *parts) {
   particles = parts;
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

void tree::set_kick_parameters(float theta_, int8_t rung_) {
   theta = theta_;
   rung = rung_;
}

hpx::lcos::local::mutex tree::mtx;
hpx::lcos::local::mutex tree::gpu_mtx;
std::stack<std::shared_ptr<kick_workspace_t>> tree::kick_works;

void tree::cleanup() {
   shutdown_daemon = true;
   daemon_running = false;
   while (kick_works.size()) {
      kick_works.pop();
   }
}

std::shared_ptr<kick_workspace_t> tree::get_workspace() {
   std::unique_lock<hpx::lcos::local::mutex> lock(mtx);
   if (kick_works.empty()) {
      lock.unlock();
      auto work = std::make_shared<kick_workspace_t>();
      return std::move(work);
   } else {
      auto work = std::move(kick_works.top());
      kick_works.pop();
      return std::move(work);
   }
}

void tree::cleanup_workspace(std::shared_ptr<kick_workspace_t> &&work) {
   std::lock_guard<hpx::lcos::local::mutex> lock(mtx);
   //  printf("120947\n");
   // printf("120412421947\n");
   kick_works.push(std::move(work));
}

fast_future<kick_return> tree_ptr::kick(kick_stack &stack, int depth, bool try_thread) {
   if (depth == 8) {
      return fast_future<kick_return>(((tree*) ptr)->send_kick_to_gpu(stack, depth));
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
         return fast_future<kick_return>(((tree*) ptr)->kick(stack, depth));
      } else {
         auto new_stack = std::make_shared<kick_stack>();
         stack.dchecks.copy_top(new_stack->dchecks);
         stack.echecks.copy_top(new_stack->echecks);
         new_stack->L[depth] = stack.L[depth];
         new_stack->Lpos[depth] = stack.Lpos[depth];
         auto func = [this, depth, new_stack]() {
            auto rc = ((tree*) ptr)->kick(*new_stack, depth);
            thread_cnt--;
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
hpx::future<kick_return> tree::kick(kick_stack &stacks, int depth) {
#ifdef TEST_CHECKLIST_TIME
   static timer tm;
#endif
   // num_kicks++;
   // printf( "%li\n", num_kicks);
   kick_return rc;
   const auto theta2 = theta * theta;
   array<tree_ptr*, N_INTERACTION_TYPES> all_checks;
   array<int*, N_INTERACTION_TYPES> list_counts;
   all_checks[CC_CP_DIRECT] = stacks.dchecks.top_list();
   all_checks[CC_CP_EWALD] = stacks.echecks.top_list();
   all_checks[PC_PP_DIRECT] = stacks.dchecks.top_list();
   all_checks[PC_PP_EWALD] = stacks.echecks.top_list();
   list_counts[CC_CP_DIRECT] = &stacks.dchecks.counts.top();
   list_counts[CC_CP_EWALD] = &stacks.echecks.counts.top();
   list_counts[PC_PP_DIRECT] = &stacks.dchecks.counts.top();
   list_counts[PC_PP_EWALD] = &stacks.echecks.counts.top();
   auto workspace = get_workspace();
   auto &multis = workspace->multi_interactions;
   auto &parti = workspace->part_interactions;
   auto &next_checks = workspace->next_checks;
   int ninteractions = is_leaf() ? 4 : 2;
   for (int type = 0; type < ninteractions; type++) {
      auto *checks = (all_checks[type]);
      next_checks.resize(0);
      multis.resize(0);
      parti.resize(0);
      const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
      const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;

      do {
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
               multis.push_back(checks[ci]);
            } else if (!checks[ci].is_leaf()) {
               const auto child_checks = checks[ci].get_children().get();
               next_checks.push_back(child_checks[LEFT]);
               next_checks.push_back(child_checks[RIGHT]);
            } else if (!checks[ci].opened++) {
               next_checks.push_back(checks[ci]);
            } else {
               parti.push_back(checks[ci]);
            }
         }
#ifdef TEST_CHECKLIST_TIME
         tm.stop();
#endif
         if (type == PC_PP_DIRECT || type == CC_CP_DIRECT) {
            stacks.dchecks.pop();
            stacks.dchecks.push(next_checks);
         } else {
            stacks.echecks.pop();
            stacks.echecks.push(next_checks);
         }
      } while (direct && *(list_counts[type]));

      /*********** DO INTERACTIONS *********************/

   }

// printf("3\n");
   cleanup_workspace(std::move(workspace));

   if (!is_leaf()) {
      // printf("4\n");
      const bool try_thread = parts.second - parts.first > TREE_MIN_PARTS2THREAD;
      stacks.dchecks.copy_top(stacks.dchecks);
      stacks.echecks.copy_top(stacks.echecks);
      array<hpx::future<kick_return>, NCHILD> futs;
      futs[LEFT] = children[LEFT].kick(stacks, depth + 1, try_thread);
      //  printf("5\n");
      stacks.dchecks.pop();
      stacks.echecks.pop();
      futs[RIGHT] = children[RIGHT].kick(stacks, depth + 1, false);
      //  printf("6\n");
      return hpx::when_all(futs.begin(), futs.end()).then([](hpx::future<std::vector<hpx::future<kick_return>>> futs) {
         auto tmp = futs.get();
         kick_return rc;
         rc.rung = std::max(tmp[LEFT].get().rung, tmp[RIGHT].get().rung);
         return rc;
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

std::queue<gpu_kick> tree::gpu_queue;
bool tree::daemon_running = false;
bool tree::shutdown_daemon = false;

using cuda_exec_return =
std::pair<std::function<bool()>, std::shared_ptr<finite_vector<kick_return, KICK_GRID_SIZE>>>;

cuda_exec_return cuda_execute_kick_kernel(cuda_workspace_t *workspace, int grid_size);

void tree::gpu_daemon() {
   using namespace std::chrono_literals;

   printf("Starting gpu kick daemon\n");
   daemon_running = true;
   int tries = 0;
   int min_grid_size = KICK_GRID_SIZE;
   while (!shutdown_daemon) {
      hpx::this_thread::yield();
      if (tries == min_grid_size) {
         min_grid_size = std::max(1, min_grid_size / 2);
         tries = 0;
      } else {
         tries++;
      }
      std::lock_guard<hpx::lcos::local::mutex> lock(gpu_mtx);
      while (gpu_queue.size() >= min_grid_size) {
         int grid_size = min_grid_size;
         min_grid_size = KICK_GRID_SIZE;
         std::vector<hpx::lcos::local::promise<kick_return>> promises;
         cuda_workspace_t *workspace;
         CUDA_MALLOC(workspace, 1);
         new (workspace) cuda_workspace_t(grid_size);
         for (int i = 0; i < grid_size; i++) {
            auto work_item = std::move(gpu_queue.front());
            gpu_queue.pop();
            workspace->stacks[i] = (std::move(work_item.stack));
            workspace->depths[i] = (work_item.depth);
            workspace->roots[i] = (work_item.tree);
            promises.push_back(std::move(work_item.promise));
         }
         printf("Executing %i blocks\n", grid_size);
         auto exec_ret = cuda_execute_kick_kernel(workspace, grid_size);
         hpx::apply([exec_ret, grid_size, workspace](std::vector<hpx::lcos::local::promise<kick_return>> &&promises) {
            while (!exec_ret.first()) {
               hpx::this_thread::yield();
            }
            for (int i = 0; i < grid_size; i++) {
               promises[i].set_value((*exec_ret.second)[i]);
            }
            workspace->cuda_workspace_t::~cuda_workspace_t();
            CUDA_FREE(workspace);
         },std::move(promises));
      }
   }
   shutdown_daemon = false;
}

hpx::future<kick_return> tree::send_kick_to_gpu(kick_stack &stack, int depth) {
   std::unique_lock<hpx::lcos::local::mutex> lock(gpu_mtx);
   if (!daemon_running) {
      shutdown_daemon = false;
      daemon_running = true;
      lock.unlock();
      hpx::apply([]() {
         gpu_daemon();
      });
   } else {
      lock.unlock();
   }
   gpu_kick gpu;
   tree_ptr me;
   me.ptr = (uintptr_t)(this);
   me.rank = hpx_rank();
   gpu.tree = me;
   stack.dchecks.copy_top(gpu.stack.dchecks);
   stack.echecks.copy_top(gpu.stack.echecks);
   gpu.stack.L[depth] = stack.L[depth];
   gpu.stack.Lpos[depth] = stack.Lpos[depth];
   gpu.depth = depth;
   auto fut = gpu.promise.get_future();

   lock.lock();
   gpu_queue.push(std::move(gpu));

   return std::move(fut);

}

