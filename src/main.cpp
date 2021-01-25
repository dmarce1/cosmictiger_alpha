#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>

int hpx_main() {
   hpx_init();
   particle_set::create();

   return hpx::finalize();
}

int main(int argc, char *argv[]) {
   std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };
   hpx::init(argc, argv, cfg);
}
