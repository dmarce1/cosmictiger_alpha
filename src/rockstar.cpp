#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/global.hpp>

int rockstar_bh_sort(vector<halo_tree>& trees, vector<halo_part>& parts, range box, int begin, int end, int depth);

void rockstar_bh(vector<halo_part>& parts) {
	range box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = std::numeric_limits<float>::max();
		box.end[dim] = -std::numeric_limits<float>::max();
	}
	for (int pi = 0; pi < parts.size(); pi++) {
		for (int dim = 0; dim < NDIM; dim++) {
			box.begin[dim] = std::min(box.begin[dim], parts[pi].x[dim]);
			box.end[dim] = std::max(box.end[dim], parts[pi].x[dim]);
		}
	}

	vector<halo_tree> trees;
	rockstar_bh_sort(trees, parts, box, 0, parts.size(), 0);
}

int rockstar_bh_sort(vector<halo_tree>& trees, vector<halo_part>& parts, range box, int begin, int end, int depth) {
	const float h = global().opts.hsoft * 2.f;
	halo_tree tree;
	const int tree_index = trees.size();
	trees.resize(trees.size() + 1);
	if (end - begin > 1) {
		int xdim = depth % NDIM;
		int lo = begin;
		int hi = end;
		int mid;
		const float xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
		while (lo < hi) {
			if (parts[lo].x[xdim] >= xmid) {
				while (lo != hi) {
					hi--;
					if (parts[hi].x[xdim] < xmid) {
						swap(parts[hi], parts[lo]);
						break;
					}
				}
			}
			lo++;
		}
		mid = hi;
		range boxl = box;
		range boxr = box;
		boxl.end[xdim] = boxr.end[xdim] = xmid;
		tree.children[LEFT] = rockstar_bh_sort(trees, parts, boxl, begin, mid, depth + 1);
		tree.children[RIGHT] = rockstar_bh_sort(trees, parts, boxr, mid, end, depth + 1);
		auto& left = trees[tree.children[LEFT]];
		auto& right = trees[tree.children[RIGHT]];
		const auto mtot = left.mass + right.mass;
		assert(mtot > 0.0);
		for (int dim = 0; dim < NDIM; dim++) {
			tree.x[dim] = left.x[dim] * left.mass + right.x[dim] * right.mass;
		}
		const float minv = 1.0f / mtot;
		for (int dim = 0; dim < NDIM; dim++) {
			tree.x[dim] *= minv;
		}
		float rleft = 0.0f;
		float rright = 0.0f;
		for (int dim = 0; dim < NDIM; dim++) {
			rleft += sqr(tree.x[dim] - right.x[dim]);
			rright += sqr(tree.x[dim] - left.x[dim]);
		}
		rleft = sqrt(rleft);
		rright = sqrt(rright);
		tree.radius = std::max(rleft, rright) + std::max(left.radius, right.radius);
	} else {
		tree.children[LEFT] = tree.children[RIGHT] = -1;
		if (end - begin) {
			tree.radius = h;
			tree.mass = 1.0;
			tree.x = parts[begin].x;
		} else {
			tree.mass = 0.0;
			tree.radius = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				tree.x[dim] = 1e6;
			}
		}

	}
	trees[tree_index] = tree;
	return tree_index;
}
