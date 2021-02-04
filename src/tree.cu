#include <cosmictiger/tree.hpp>
#include <functional>

//CUDA_KERNEL cuda_kick()

CUDA_KERNEL cuda_kick_kernel(finite_vector<kick_return, KICK_GRID_SIZE> *rc,
      finite_vector<kick_stack, KICK_GRID_SIZE> stacks, finite_vector<tree_ptr, KICK_GRID_SIZE> roots,
      finite_vector<int, KICK_GRID_SIZE> depths) {

}

std::pair<std::function<bool()>, std::shared_ptr<finite_vector<kick_return, KICK_GRID_SIZE>>> cuda_execute_kick_kernel(
      finite_vector<kick_stack, KICK_GRID_SIZE> &&stacks, finite_vector<tree_ptr, KICK_GRID_SIZE> &&roots,
      finite_vector<int, KICK_GRID_SIZE> &&depths, int grid_size) {
   std::vector < std::function < kick_return() >> returns;
   finite_vector<kick_return, KICK_GRID_SIZE> *rcptr;
   CUDA_MALLOC(rcptr, 1);
   new (rcptr) finite_vector<kick_return, KICK_GRID_SIZE>();
   rcptr->resize(grid_size);
   cudaStream_t stream;
   cudaEvent_t event;
   CUDA_CHECK(cudaStreamCreate(&stream));
   CUDA_CHECK(cudaEventCreate(&event));

   /*******************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, 0, stream>>>(rcptr, std::move(stacks),std::move(roots),std::move(depths));/**/
/**/CUDA_CHECK(cudaEventRecord(event, stream));/********************************************************************************/
   /*******************************************************************************************************************************/

   struct cuda_kick_future_shared {
      cudaStream_t stream;
      cudaEvent_t event;
      std::shared_ptr<finite_vector<kick_return, KICK_GRID_SIZE>> returns;
      finite_vector<kick_return, KICK_GRID_SIZE> *rcptr;
      int grid_size;
      mutable bool ready;
   public:
      cuda_kick_future_shared() {
         ready = false;
      }
      bool operator()() const {
         if (!ready) {
            if (cudaEventQuery(event) == cudaSuccess) {
               ready = true;
               CUDA_CHECK(cudaStreamSynchronize(stream));
               CUDA_CHECK(cudaEventDestroy(event));
               CUDA_CHECK(cudaStreamDestroy(stream));
               *returns = std::move(*rcptr);
               rcptr->finite_vector<kick_return, KICK_GRID_SIZE>::~finite_vector<kick_return, KICK_GRID_SIZE>();
               CUDA_FREE(rcptr);
            }
         }
         return ready;
      }
   };

   cuda_kick_future_shared fut;
   fut.returns = std::make_shared<finite_vector<kick_return, KICK_GRID_SIZE>>();
   fut.stream = stream;
   fut.event = event;
   fut.rcptr = rcptr;
   fut.grid_size = grid_size;
   std::function < bool() > ready_func = [fut]() {
      return fut();
   };
   return std::make_pair(std::move(ready_func), std::move(fut.returns));
}

