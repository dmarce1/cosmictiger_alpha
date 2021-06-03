/*  
 Copyright (c) 2016 Dominic C. Marcello

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EXPAN222SION_H_
#define EXPAN222SION_H_

#include <cosmictiger/array.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/spherical_harmonic.hpp>
#include <cosmictiger/cuda.hpp>


#define LORDER ORDER
#define LP (LORDER*(LORDER+1)/2)

struct force {
	float phi;
//	float padding;
	array<float, NDIM> g;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & phi;
		arc & g;
	}
};

template<class T>
using expansion = sphericalY<T,LORDER>;

expansion<float>& shift_expansion(expansion<float>& L, const array<float, NDIM> &dX, bool do_phi);
CUDA_DEVICE expansion<float>& cuda_shift_expansion(expansion<float>& L, const array<float, NDIM> &dX, bool do_phi);
CUDA_EXPORT void shift_expansion(const expansion<float> &L, array<float, NDIM> &g, float &phi, const array<float, NDIM> &dX, bool do_phi);
CUDA_DEVICE void expansion_init();
__host__ void expansion_init_cpu();

/* namespace fmmx */
#endif /* expansion_H_ */
