

set -x

source ~/scripts/sourceme.sh gperftools
source ~/scripts/sourceme.sh hwloc
source ~/scripts/sourceme.sh vc
source ~/scripts/sourceme.sh silo
source ~/scripts/sourceme.sh $1/hpx

#rm -rf $1
#mkdir $1
cd $1
rm CMakeCache.txt
rm -r CMakeFiles


cmake \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      ..

