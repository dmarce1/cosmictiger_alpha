#include <cosmictiger/fourier.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/vector.hpp>

static vector<vector<cmplx>> Y;
static std::vector<std::shared_ptr<spinlock_type>> mutexes;
static int N;
static int begin;
static std::vector<hpx::id_type> localities;
static int end;
static int span;
static int rank;
static int nranks;

vector<vector<cmplx>> fourier3d_transpose(int xbegin, int xend, int zbegin, int zend, vector<vector<cmplx>>);
void fourier3d_transpose_xz();
void fft1d(cmplx* Y, int N);

HPX_PLAIN_ACTION(fourier3d_initialize);
HPX_PLAIN_ACTION(fourier3d_destroy);
HPX_PLAIN_ACTION(fourier3d_transpose);
HPX_PLAIN_ACTION(fourier3d_accumulate);
HPX_PLAIN_ACTION(fourier3d_transpose_xz);
HPX_PLAIN_ACTION(fourier3d_read);
HPX_PLAIN_ACTION(fourier3d_mirror);

int slab_to_rank(int xi) {
	return std::min((int) ((xi + int(N % nranks != 0)) * nranks / N), nranks - 1);
}

void fourier3d_mirror() {
	std::vector<hpx::future<void>> futs;
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_mirror_action>(localities[i]));
		}
	}
	for (int xi = begin; xi < end; xi++) {
		if (xi >= N / 2) {
			int xi0 = (N - xi) % N;
			auto future = hpx::async<fourier3d_read_action>(localities[slab_to_rank(xi)], xi0, xi0 + 1, 0, N, 0, N);
			futs.push_back(future.then([xi](hpx::future<vector<cmplx>> fut) {
				auto data = fut.get();
				for( int yi = 0; yi < N; yi++) {
					for( int zi = 0; zi < N; zi++) {
						Y[xi-begin][yi*N+zi] = data[((N-yi)%N)*N+((N-zi)%N)].conj();
					}
				}
			}));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

vector<cmplx> fourier3d_read(int xb, int xe, int yb, int ye, int zb, int ze) {
	vector<cmplx> data;
	const int yspan = ye - yb;
	const int zspan = ze - zb;
	std::vector<hpx::future<void>> futs;
	data.resize((xe - xb) * yspan * zspan);
	if (xb < 0) {
		auto data1 = fourier3d_read(0, xe, yb, ye, zb, ze);
		auto data2 = fourier3d_read(xb + N, N, yb, ye, zb, ze);
		for (int yi = yb; yi < ye; yi++) {
			for (int zi = zb; zi < ze; zi++) {
				for (int xi = xb; xi < 0; xi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data2[yspan * zspan * (xi - xb)
							+ zspan * (yi - yb) + (zi - zb)];
				}
				for (int xi = 0; xi < xe; xi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data1[yspan * zspan * (xi - 0)
							+ zspan * (yi - yb) + (zi - zb)];
				}

			}
		}
	} else if (xe > N) {
		auto data1 = fourier3d_read(xb, N, yb, ye, zb, ze);
		auto data2 = fourier3d_read(0, xe - N, yb, ye, zb, ze);
		for (int yi = yb; yi < ye; yi++) {
			for (int zi = zb; zi < ze; zi++) {
				for (int xi = N; xi < xe; xi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data2[yspan * zspan * (xi - N)
							+ zspan * (yi - yb) + (zi - zb)];
				}
				for (int xi = xb; xi < N; xi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data1[yspan * zspan * (xi - xb)
							+ zspan * (yi - yb) + (zi - zb)];
				}

			}
		}

	} else if (yb < 0) {
		auto data1 = fourier3d_read(xb, xe, 0, ye, zb, ze);
		auto data2 = fourier3d_read(xb, xe, yb + N, N, zb, ze);
		const int yspan1 = ye;
		const int yspan2 = -yb;
		for (int xi = xb; xi < xe; xi++) {
			for (int zi = zb; zi < ze; zi++) {
				for (int yi = yb; yi < 0; yi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data2[yspan1 * zspan * (xi - xb)
							+ zspan * (yi - yb) + (zi - zb)];
				}
				for (int yi = 0; yi < ye; yi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data1[yspan2 * zspan * (xi - xb)
							+ zspan * (yi - 0) + (zi - zb)];
				}

			}
		}
	} else if (ye > N) {
		auto data1 = fourier3d_read(xb, xe, yb, N, zb, ze);
		auto data2 = fourier3d_read(xb, xe, 0, ye - N, zb, ze);
		const int yspan1 = N - yb;
		const int yspan2 = ye - N;
		for (int xi = xb; xi < xe; xi++) {
			for (int zi = zb; zi < ze; zi++) {
				for (int yi = N; yi < ye; yi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data2[yspan1 * zspan * (xi - xb)
							+ zspan * (yi - N) + (zi - zb)];
				}
				for (int yi = yb; yi < N; yi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data1[yspan2 * zspan * (xi - xb)
							+ zspan * (yi - yb) + (zi - zb)];
				}

			}
		}
	} else if (zb < 0) {
		auto data1 = fourier3d_read(xb, xe, yb, ye, 0, ze);
		auto data2 = fourier3d_read(xb, xe, yb, ye, zb + N, N);
		const int zspan1 = ze;
		const int zspan2 = -zb;
		for (int xi = xb; xi < xe; xi++) {
			for (int yi = yb; yi < ye; yi++) {
				for (int zi = zb; zi < 0; zi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data2[yspan * zspan * (xi - xb)
							+ zspan2 * (yi - yb) + (zi - zb)];
				}
				for (int zi = 0; zi < ze; zi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data1[yspan * zspan * (xi - xb)
							+ zspan1 * (yi - yb) + (zi - 0)];
				}
			}
		}
	} else if (ze > N) {
		auto data1 = fourier3d_read(xb, xe, yb, ye, zb, N);
		auto data2 = fourier3d_read(xb, xe, yb, ye, 0, ze - N);
		const int zspan1 = N - zb;
		const int zspan2 = ze - N;
		for (int xi = xb; xi < xe; xi++) {
			for (int yi = yb; yi < ye; yi++) {
				for (int zi = N; zi < ze; zi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data2[yspan * zspan * (xi - xb)
							+ zspan2 * (yi - yb) + (zi - N)];
				}
				for (int zi = zb; zi < N; zi++) {
					data[yspan * zspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = data1[yspan * zspan * (xi - xb)
							+ zspan1 * (yi - yb) + (zi - zb)];
				}
			}
		}
	} else if (slab_to_rank(xb) != rank || slab_to_rank(xe) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe);
		for (int other_rank = rankb; other_rank < ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			vector<cmplx> rank_data(this_xspan * yspan * zspan);
			auto fut = hpx::async<fourier3d_read_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze);
			futs.push_back(
					fut.then(
							[&data, this_xb, this_xe, yspan, zspan, xb, yb, ye, zb, ze](hpx::future<vector<cmplx>> fut) {
								auto rank_data = fut.get();
								for( int xi = this_xb; xi < this_xe; xi++) {
									for( int yi = yb; yi < ye; yi++) {
										for( int zi = zb; zi < ze; zi++) {
											data[zspan*yspan*(xi-xb)+zspan*(yi-yb)+(zi-zb)] = rank_data[zspan*yspan*(xi-this_xb)+zspan*(yi-yb)+(zi-zb)];
										}
									}
								}
							}));
		}
	} else {
		for (int xi = xb; xi < xe; xi++) {
			for (int yi = yb; yi < ye; yi++) {
				for (int zi = zb; zi < ze; zi++) {
					data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = Y[xi - begin][+zspan * (yi - yb)
							+ (zi - zb)];
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

void fourier3d_accumulate(int xb, int xe, int yb, int ye, int zb, int ze, vector<cmplx> data) {
	const int yspan = ye - yb;
	const int zspan = ze - zb;
	std::vector<hpx::future<void>> futs;
	if (slab_to_rank(xb) != rank || slab_to_rank(xe) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe);
		for (int other_rank = rankb; other_rank < ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			vector<cmplx> rank_data(this_xspan * yspan * zspan);
			for (int xi = this_xb; xi < this_xe; xi++) {
				for (int yi = yb; yi < ye; yi++) {
					for (int zi = zb; zi < ze; zi++) {
						rank_data[zspan * yspan * (xi - this_xb) + zspan * (yi - yb) + (zi - zb)] = data[zspan * yspan
								* (xi - xb) + yspan * (yi - yb) + (zi - zb)];
					}
				}
			}
			futs.push_back(
					hpx::async<fourier3d_accumulate_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze,
							std::move(rank_data)));
		}
	} else {
		for (int xi = xb; xi < xe; xi++) {
			std::lock_guard<spinlock_type> lock(*mutexes[xi]);
			for (int yi = yb; yi < ye; yi++) {
				for (int zi = zb; zi < ze; zi++) {
					Y[xi - xb][N * (yi - yb) + (zi - zb)] += data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)];
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_execute() {
	for (int i = 0; i < span; i++) {
		fft2d(Y[i].data(), N);
	}
	fourier3d_transpose_xz();
	for (int i = 0; i < span; i++) {
		fft1d(Y[i].data(), N);
	}
	fourier3d_transpose_xz();
}

void fourier3d_inv_execute() {
	const float factor = 1.0f / (N * N * N);
	for (int i = 0; i < span; i++) {
		for (int j = 0; j < N * N; j++) {
			Y[i][j].real() *= factor;
			Y[i][j].imag() *= factor;
		}
	}
	fourier3d_execute();
}

void fourier3d_initialize(int N_) {
	std::vector<hpx::future<void>> futs;
	rank = hpx_rank();
	nranks = hpx_size();
	localities = hpx_localities();
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_initialize_action>(localities[i], N));
		}
	}
	N = N_;
	begin = rank * N / nranks;
	end = (rank + 1) * N / nranks;
	span = end - begin;
	Y.resize(span);
	mutexes.resize(span);
	for (int i = 0; i < span; i++) {
		mutexes[i] = std::make_shared<spinlock_type>();
		Y[i].resize(N * N);
		for (int j = 0; j < N * N; j++) {
			Y[i][j].real() = Y[i][j].imag() = 0.0;
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_destroy() {
	std::vector<hpx::future<void>> futs;
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_destroy_action>(localities[i]));
		}
	}
	Y = decltype(Y)();
	mutexes = decltype(mutexes)();
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_transpose_xz() {
	std::vector<hpx::future<void>> futs;
	rank = hpx_rank();
	nranks = hpx_size();
	localities = hpx_localities();
	if (rank == 0) {
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_transpose_xz_action>(localities[i]));
		}
	}
	vector<vector<cmplx>> data;
	for (int other = rank; other < nranks; other++) {
		const int zbegin = other * N / nranks;
		const int zend = (other + 1) * N / nranks;
		const int zspan = zend - zbegin;
		data.resize(span);
		for (int i = 0; i < span; i++) {
			data[i].resize(N * zspan);
			for (int j = 0; j < N; j++) {
				for (int k = zbegin; k < zend; k++) {
					data[i][j * zspan + k - zbegin] = Y[i][j * N + k];
				}
			}
		}
		auto future = hpx::async<fourier3d_transpose_action>(localities[other], zbegin, zend, begin, end,
				std::move(data));
		futs.push_back(future.then([zspan,zbegin,zend](hpx::future<vector<vector<cmplx>>> fut) {
			auto data = fut.get();
			for (int i = 0; i < span; i++) {
				for (int j = 0; j < N; j++) {
					for (int k = zbegin; k < zend; k++) {
						Y[i][j * N + k] = data[i][j * zspan + k - zbegin];
					}
				}
			}
		}));
	}

	hpx::wait_all(futs.begin(), futs.end());
}

vector<vector<cmplx>> fourier3d_transpose(int xbegin, int xend, int zbegin, int zend, vector<vector<cmplx>> data) {
	const int xspan = xend - xbegin;
	for (int xi = 0; xi < span; xi++) {
		for (int yi = 0; yi < N; yi++) {
			for (int zi = zbegin; zi < zend; zi++) {
				std::swap(data[zi - zbegin][yi * xspan + xi], Y[xi][yi * N + zi]);
			}
		}
	}
	return std::move(data);
}
