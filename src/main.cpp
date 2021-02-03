#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/tree.hpp>

int hpx_main(int argc, char *argv[]) {
   options opts;
   printf( "%li\n", sizeof(std::shared_ptr<int>));
   printf( "%li\n", sizeof(sort_params));
    if (process_options(argc, argv, opts)) {
      hpx_init();
      const auto cuda = cuda_init();
      global_init(opts, cuda);

      if (opts.test != "") {
         test_run(opts.test);
      } else {

      }
   }

   tree::cleanup();
   return hpx::finalize();
}

int main(int argc, char *argv[]) {
   std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };
   cfg.push_back("hpx.stacks.small_size=0x20000");
   hpx::init(argc, argv, cfg);
}
