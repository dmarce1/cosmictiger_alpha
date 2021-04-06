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
#include <cosmictiger/cuda.hpp>
constexpr int LP = 35;

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
class expansion {

	array<T, LP> data;
public:
	expansion<T>& operator*=(T r) {
		for (int i = 0; i != LP; ++i) {
			data[i] *= r;
		}
		return *this;
	}
	CUDA_EXPORT
	expansion();CUDA_EXPORT
	T operator ()() const;CUDA_EXPORT
	T& operator ()();CUDA_EXPORT
	T operator ()(int i) const;CUDA_EXPORT
	T& operator ()(int i);CUDA_EXPORT
	T operator ()(int i, int j) const;CUDA_EXPORT
	T& operator ()(int i, int j);CUDA_EXPORT
	T operator ()(int i, int j, int k) const;CUDA_EXPORT
	T& operator ()(int i, int j, int k);CUDA_EXPORT
	T operator ()(int i, int j, int k, int l) const;CUDA_EXPORT
	T& operator ()(int i, int j, int k, int l);CUDA_EXPORT
	expansion<T>& operator =(const expansion<T> &expansion);CUDA_EXPORT
	expansion<T>& operator =(T expansion);CUDA_EXPORT
	void translate_L2(array<T, NDIM>& g, T& phi, const array<T, NDIM> &dX) const;CUDA_EXPORT
	expansion<T> operator<<(const array<T, NDIM> &dX) const;CUDA_EXPORT
	expansion<T>& operator<<=(const array<T, NDIM> &dX);CUDA_EXPORT
	void compute_D(const array<T, NDIM> &Y);CUDA_EXPORT
	std::array<expansion<T>, NDIM> get_derivatives() const;CUDA_EXPORT
	inline T& operator[](int i) {
		return data[i];
	}
	CUDA_EXPORT
	inline T operator[](int i) const {
		return data[i];
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		for (int i = 0; i < LP; i++) {
			arc & data[i];
		}
	}
	CUDA_EXPORT
	void scale_back();
};

template<class T>
CUDA_EXPORT inline void expansion<T>::scale_back() {
	return;
/*	constexpr auto DSCALE2 = DSCALE * DSCALE;
	constexpr auto DSCALE3 = DSCALE2 * DSCALE;
	constexpr auto DSCALE4 = DSCALE3 * DSCALE;
	constexpr auto DSCALE5 = DSCALE4 * DSCALE;
	data[0] *= T(DSCALE);
	data[1] *= T(DSCALE2);
	data[2] *= T(DSCALE2);
	data[3] *= T(DSCALE2);
	data[4] *= T(DSCALE3);
	data[5] *= T(DSCALE3);
	data[6] *= T(DSCALE3);
	data[7] *= T(DSCALE3);
	data[8] *= T(DSCALE3);
	data[9] *= T(DSCALE3);
	data[10] *= T(DSCALE4);
	data[11] *= T(DSCALE4);
	data[12] *= T(DSCALE4);
	data[13] *= T(DSCALE4);
	data[14] *= T(DSCALE4);
	data[15] *= T(DSCALE4);
	data[16] *= T(DSCALE4);
	data[17] *= T(DSCALE4);
	data[18] *= T(DSCALE4);
	data[19] *= T(DSCALE4);
	data[20] *= T(DSCALE5);
	data[21] *= T(DSCALE5);
	data[22] *= T(DSCALE5);
	data[23] *= T(DSCALE5);
	data[24] *= T(DSCALE5);
	data[25] *= T(DSCALE5);
	data[26] *= T(DSCALE5);
	data[27] *= T(DSCALE5);
	data[28] *= T(DSCALE5);
	data[29] *= T(DSCALE5);
	data[30] *= T(DSCALE5);
	data[31] *= T(DSCALE5);
	data[32] *= T(DSCALE5);
	data[33] *= T(DSCALE5);
	data[34] *= T(DSCALE5);*/
}

template<class T>
CUDA_EXPORT inline expansion<T>::expansion() {
}

template<class T>
CUDA_EXPORT inline T expansion<T>::operator ()() const {
	return data[0];
}
template<class T>
CUDA_EXPORT inline T& expansion<T>::operator ()() {
	return data[0];
}

template<class T>
CUDA_EXPORT inline T expansion<T>::operator ()(int i) const {
	return data[1 + i];
}
template<class T>
CUDA_EXPORT inline T& expansion<T>::operator ()(int i) {
	return data[1 + i];
}

template<class T>
CUDA_EXPORT inline T expansion<T>::operator ()(int i, int j) const {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return data[4 + map2[i][j]];
}
template<class T>
CUDA_EXPORT inline T& expansion<T>::operator ()(int i, int j) {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return data[4 + map2[i][j]];
}

template<class T>
CUDA_EXPORT inline T expansion<T>::operator ()(int i, int j, int k) const {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4,
			7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } };

	return data[10 + map3[i][j][k]];
}
template<class T>
CUDA_EXPORT inline T& expansion<T>::operator ()(int i, int j, int k) {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4,
			7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } };

	return data[10 + map3[i][j][k]];
}

template<class T>
CUDA_EXPORT inline T& expansion<T>::operator ()(int i, int j, int k, int l) {
	static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 },
			{ 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3,
			6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } } }, { { { 2, 4, 5 }, { 4,
			7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, { { 5, 8, 9 }, { 8, 12, 13 },
			{ 9, 13, 14 } } } };
	return data[20 + map4[i][j][k][l]];
}

template<class T>
CUDA_EXPORT inline T expansion<T>::operator ()(int i, int j, int k, int l) const {
	static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 },
			{ 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3,
			6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } } }, { { { 2, 4, 5 }, { 4,
			7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, { { 5, 8, 9 }, { 8, 12, 13 },
			{ 9, 13, 14 } } } };
	return data[20 + map4[i][j][k][l]];
}

template<class T>
CUDA_EXPORT inline expansion<T>& expansion<T>::operator =(const expansion<T> &expansion) {
	for (int i = 0; i < LP; i++) {
		data[i] = expansion[i];
	}
	return *this;
}

template<class T>
CUDA_EXPORT inline expansion<T>& expansion<T>::operator =(T expansion) {
	for (int i = 0; i < LP; i++) {
		data[i] = expansion;
	}
	return *this;
}

template<class T>
CUDA_EXPORT inline expansion<T> expansion<T>::operator<<(const array<T, NDIM> &dX) const {
	expansion<T> you = *this;
	you <<= dX;
	return you;
}

template<class T>
struct expansion_factors: public expansion<T> {
	expansion_factors() {
		for (int i = 0; i < LP; i++) {
			(*this)[i] = T(0.0);
		}
		(*this)() += T(1);
		for (int a = 0; a < NDIM; ++a) {
			(*this)(a) += T(1.0);
			for (int b = 0; b < NDIM; ++b) {
				(*this)(a, b) += T(0.5);
				for (int c = 0; c < NDIM; ++c) {
					(*this)(a, b, c) += T(1.0 / 6.0);
					for (int d = 0; d < NDIM; ++d) {
						(*this)(a, b, c, d) += T(1.0 / 24.0);
					}
				}
			}
		}
	}
};

template<class T>
CUDA_EXPORT inline expansion<T>& expansion<T>::operator<<=(const array<T, NDIM> &dX) {
#ifndef __CUDA_ARCH__
	const static expansion_factors<T> factor;
#endif
	expansion<T> &me = *this;
	for (int a = 0; a < 3; a++) {
		me() += me(a) * dX[a];
		for (int b = 0; b <= a; b++) {
			me() += me(a, b) * dX[a] * dX[b] * factor(a, b);
			for (int c = 0; c <= b; c++) {
				me() += me(a, b, c) * dX[a] * dX[b] * dX[c] * factor(a, b, c);
				for (int d = 0; d <= c; d++) {
					me() += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * factor(a, b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			me(a) += me(a, b) * dX[b];
			for (int c = 0; c <= b; c++) {
				me(a) += me(a, b, c) * dX[b] * dX[c] * factor(b, c);
				for (int d = 0; d <= c; d++) {
					me(a) += me(a, b, c, d) * dX[b] * dX[c] * dX[d] * factor(b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b <= a; b++) {
			for (int c = 0; c < NDIM; c++) {
				me(a, b) += me(a, b, c) * dX[c];
				for (int d = 0; d <= c; d++) {
					me(a, b) += me(a, b, c, d) * dX[c] * dX[d] * factor(c, d);
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = 0; b <= a; b++) {
			for (int c = 0; c <= b; c++) {
				for (int d = 0; d < 3; d++) {
					me(a, b, c) += me(a, b, c, d) * dX[d];
				}
			}
		}
	}

	return me;
}

template<class T>
CUDA_EXPORT inline void expansion<T>::translate_L2(array<T, NDIM>& g, T& phi, const array<T, NDIM> &dX) const {
	const static expansion_factors<T> factor;

	const auto &me = *this;
	force f;
	phi = (*this)();
	for (int a = 0; a < 3; a++) {
		phi += me(a) * dX[a];
		for (int b = a; b < 3; b++) {
			phi += me(a, b) * dX[a] * dX[b] * factor(a, b);
			for (int c = b; c < 3; c++) {
				phi += me(a, b, c) * dX[a] * dX[b] * dX[c] * factor(a, b, c);
				for (int d = c; d < 3; d++) {
					phi += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * factor(a, b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		g[a] = -(*this)(a);
		for (int b = 0; b < 3; b++) {
			g[a] -= me(a, b) * dX[b];
			for (int c = b; c < 3; c++) {
				g[a] -= me(a, b, c) * dX[b] * dX[c] * factor(b, c);
				for (int d = c; d < 3; d++) {
					g[a] -= me(a, b, c, d) * dX[b] * dX[c] * dX[d] * factor(b, c, d);
				}
			}
		}
	}
}

expansion<float>& shift_expansion(expansion<float>& L, const array<float, NDIM> &dX, bool do_phi);
CUDA_DEVICE expansion<float>& cuda_shift_expansion(expansion<float>& L, const array<float, NDIM> &dX, bool do_phi);
CUDA_EXPORT void shift_expansion(const expansion<float> &L, array<float, NDIM> &g, float &phi, const array<float, NDIM> &dX, bool do_phi);
CUDA_DEVICE void expansion_init();
__host__ void expansion_init_cpu();

/* namespace fmmx */
#endif /* expansion_H_ */
