#include <cosmictiger/fourier.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/vector.hpp>

static vector<vector<cmplx>> Y;
static vector<std::shared_ptr<spinlock_type>> mutexes;
static int N;
static int begin;
static std::vector<hpx::id_type> localities;
static int end;
static int span;
static int rank;
static int nranks;

vector<cmplx> fourier3d_transpose(int xbegin, int xend, int zi, vector<cmplx>);
void fourier3d_transpose_xz();
void fft1d(cmplx* Y, int N);

HPX_PLAIN_ACTION(fourier3d_initialize);
HPX_PLAIN_ACTION(fourier3d_destroy);
HPX_PLAIN_ACTION(fourier3d_transpose);
HPX_PLAIN_ACTION(fourier3d_accumulate);
HPX_PLAIN_ACTION(fourier3d_transpose_xz);
HPX_PLAIN_ACTION(fourier3d_read);
HPX_PLAIN_ACTION(fourier3d_read_real);
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
	int yspan = ye - yb;
	int zspan = ze - zb;
	data.resize(zspan * yspan * (xe - xb));
	std::vector<hpx::future<void>> futs;
	if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			auto fut = hpx::async<fourier3d_read_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze);
			futs.push_back(
					fut.then(
							[&data, this_xb, this_xe, yspan, zspan, xb, yb, ye, zb, ze](hpx::future<vector<cmplx>> fut) {
								auto rank_data = fut.get();
								std::vector<hpx::future<void>> these_futs;
								int nthreads = hardware_concurrency();
								for( int proc = 0; proc < nthreads; proc++) {
									const auto func = [xb,this_xb,proc,this_xe,nthreads,yb,ye,zb,ze,&data,&rank_data,zspan, yspan]() {
										for( int xi = this_xb + proc; xi < this_xe; xi+=nthreads) {
											for( int yi = yb; yi < ye; yi++) {
												for( int zi = zb; zi < ze; zi++) {
													data[zspan*yspan*(xi-xb)+zspan*(yi-yb)+(zi-zb)] = rank_data[zspan*yspan*(xi-this_xb)+zspan*(yi-yb)+(zi-zb)];
												}
											}
										}
									};
									these_futs.push_back(hpx::async(func));
								}
								hpx::wait_all(these_futs.begin(),these_futs.end());
							}));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [xb,proc,xe,nthreads,yb,ye,zb,ze,yspan,zspan,&data]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = Y[xi - begin][zspan * (yi - yb)
							+ (zi - zb)];
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

std::vector<float> fourier3d_read_real(int xb, int xe, int yb, int ye, int zb, int ze) {
	std::vector<float> data;
	int yspan = ye - yb;
	int zspan = ze - zb;
	data.resize(zspan * yspan * (xe - xb));
	std::vector<hpx::future<void>> futs;
	if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			auto fut = hpx::async<fourier3d_read_real_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze);
			futs.push_back(
					fut.then(
							[&data, this_xb, this_xe, yspan, zspan, xb, yb, ye, zb, ze](hpx::future<std::vector<float>> fut) {
								auto rank_data = fut.get();
								std::vector<hpx::future<void>> these_futs;
								int nthreads = hardware_concurrency();
								for( int proc = 0; proc < nthreads; proc++) {
									const auto func = [xb,this_xb,proc,this_xe,nthreads,yb,ye,zb,ze,&data,&rank_data,zspan, yspan]() {
										for( int xi = this_xb + proc; xi < this_xe; xi+=nthreads) {
											for( int yi = yb; yi < ye; yi++) {
												for( int zi = zb; zi < ze; zi++) {
													data[zspan*yspan*(xi-xb)+zspan*(yi-yb)+(zi-zb)] = rank_data[zspan*yspan*(xi-this_xb)+zspan*(yi-yb)+(zi-zb)];
												}
											}
										}
									};
									these_futs.push_back(hpx::async(func));
								}
								hpx::wait_all(these_futs.begin(),these_futs.end());
							}));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [xb,proc,xe,nthreads,yb,ye,zb,ze,yspan,zspan,&data]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							data[zspan * yspan * (xi - xb) + zspan * (yi - yb) + (zi - zb)] = Y[xi - begin][zspan * (yi - yb)
							+ (zi - zb)].real();
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

void fourier3d_accumulate(int xb, int xe, int yb, int ye, int zb, int ze, vector<cmplx> data) {
	const int yspan = ye - yb;
	const int zspan = ze - zb;
	std::vector<hpx::future<void>> futs;
	if (slab_to_rank(xb) != rank || slab_to_rank(xe - 1) != rank) {
		const int rankb = slab_to_rank(xb);
		const int ranke = slab_to_rank(xe - 1);
		for (int other_rank = rankb; other_rank <= ranke; other_rank++) {
			const int this_xb = std::max(xb, other_rank * N / nranks);
			const int this_xe = std::min(xe, (other_rank + 1) * N / nranks);
			const int this_xspan = this_xe - this_xb;
			vector<cmplx> rank_data(this_xspan * yspan * zspan);
			int nthreads = hardware_concurrency();
			std::vector<hpx::future<void>> these_futs;
			for (int proc = 0; proc < nthreads; proc++) {
				const auto func = [this_xb,this_xe,proc,nthreads,xb,yb,ye,zb,ze,zspan,yspan,&data,&rank_data]() {
					for (int xi = this_xb + proc; xi < this_xe; xi+= nthreads) {
						for (int yi = yb; yi < ye; yi++) {
							for (int zi = zb; zi < ze; zi++) {
								rank_data[zspan * yspan * (xi - this_xb) + zspan * (yi - yb) + (zi - zb)] = data[zspan * yspan
								* (xi - xb) + yspan * (yi - yb) + (zi - zb)];
							}
						}
					}
				};
				these_futs.push_back(hpx::async(func));
			}
			hpx::wait_all(these_futs.begin(), these_futs.end());
			futs.push_back(
					hpx::async<fourier3d_accumulate_action>(localities[other_rank], this_xb, this_xe, yb, ye, zb, ze,
							std::move(rank_data)));
		}
	} else {
		int nthreads = hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			const auto func = [proc,nthreads,xb,xe,yb,ye,zb,ze,&data,zspan,yspan]() {
				for (int xi = xb + proc; xi < xe; xi+=nthreads) {
					std::lock_guard<spinlock_type> lock(*mutexes[xi - xb]);
					for (int yi = yb; yi < ye; yi++) {
						for (int zi = zb; zi < ze; zi++) {
							Y[xi - xb][N * (yi - yb) + (zi - zb)] += data[zspan * yspan * (xi - xb) + zspan * (yi - yb)
							+ (zi - zb)];
						}
					}
				}
			};
			futs.push_back(hpx::async(func));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fourier3d_do2dpart() {
	for (int i = 0; i < span; i++) {
		fft2d(Y[i].data(), N);
	}
}

void fourier3d_do1dpart() {
	for (int i = 0; i < span; i++) {
		fft1d(Y[i].data(), N);
	}
}

HPX_PLAIN_ACTION(fourier3d_do1dpart);
HPX_PLAIN_ACTION(fourier3d_do2dpart);

void fourier3d_execute() {
	printf("Executing Fourier\n");
	std::vector<hpx::future<void>> futs;
	printf("doing 2d transform\n");
	for (int i = 0; i < nranks; i++) {
		futs.push_back(hpx::async<fourier3d_do2dpart_action>(localities[i]));
	}
	hpx::wait_all(futs.begin(), futs.end());
	printf("Transposing\n");
	fourier3d_transpose_xz();
	futs.resize(0);
	printf("doing 1d transform\n");
	for (int i = 0; i < nranks; i++) {
		futs.push_back(hpx::async<fourier3d_do1dpart_action>(localities[i]));
	}
	hpx::wait_all(futs.begin(), futs.end());
	printf("Transposing\n");
	fourier3d_transpose_xz();
	printf("Done executing Fourier\n");
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
		fourier3d_destroy();
		for (int i = 1; i < nranks; i++) {
			futs.push_back(hpx::async<fourier3d_initialize_action>(localities[i], N_));
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
	for (int other = rank; other < nranks; other++) {
		for (int xi = begin; xi < end; xi++) {
			vector<cmplx> data;
			const int zend = (other + 1) * N / nranks;
			const int zbegin = other * N / nranks + ((other == rank) ? (xi - begin) : 0);
			const int zspan = zend - zbegin;
			data.resize(N * zspan);
			for (int j = 0; j < N; j++) {
				for (int k = zbegin; k < zend; k++) {
					data[j * zspan + k - zbegin] = Y[xi - begin][j * N + k];
				}
			}
			auto future = hpx::async<fourier3d_transpose_action>(localities[other], zbegin, zend, xi, std::move(data));
			futs.push_back(future.then([zspan,zbegin,zend,xi](hpx::future<vector<cmplx>> fut) {
				auto data = fut.get();
				for (int j = 0; j < N; j++) {
					for (int k = zbegin; k < zend; k++) {
						Y[xi-begin][j * N + k] = data[j * zspan + k - zbegin];
					}
				}
			}));
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

vector<cmplx> fourier3d_transpose(int xbegin, int xend, int zi, vector<cmplx> data) {
	const int xspan = xend - xbegin;
	for (int xi = 0; xi < xspan; xi++) {
		for (int yi = 0; yi < N; yi++) {
			auto& a = data[yi * xspan + xi];
			auto& b = Y[xi + xbegin - begin][yi * N + zi];
			swap(a, b);
		}
	}
	return std::move(data);
}
