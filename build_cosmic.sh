

set -x

source ~/scripts/sourceme.sh gperftools
source ~/scripts/sourceme.sh hwloc
source ~/scripts/sourceme.sh vc
source ~/scripts/sourceme.sh silo
source ~/scripts/sourceme.sh $1/hpx_60

#rm -rf $1
#mkdir $1
cd $1
rm CMakeCache.txt
rm -r CMakeFiles


cmake -DCMAKE_PREFIX_PATH="$HOME/local/release/hpx_60" \
      -DCMAKE_CXX_COMPILER=mpic++  \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_FLAGS="-fPIC -std=c++0x -L$HOME/local/boost/lib -march=native" \
      -DCMAKE_C_FLAGS="-fPIC -L$HOME/local/boost/lib" \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      -DCMAKE_INSTALL_PREFIX="$HOME/local/$1/octotiger"                                   \
      -DBOOST_ROOT=$HOME/local/boost \
      -DHPX_IGNORE_COMPILER_COMPATIBILITY=on \
      ..

