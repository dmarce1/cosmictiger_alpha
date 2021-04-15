/*
 * unrolled.hpp
 *
 *  Created on: Apr 15, 2021
 *      Author: dmarce1
 */

#ifndef DEQUE_HPP_
#define DEQUE_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/hpx.hpp>

#include <stack>

template<class T>
class unrolled {
private:
	static constexpr int entry_size = 27;
	static constexpr int allocation_size = 4*1024;
	struct unrolled_entry {
		array<T, entry_size> data;
		unrolled_entry* next;
		unrolled_entry* prev;
		int count;
	};
	static std::stack<unrolled_entry*> freelist;
	static std::stack<unrolled_entry*> allocations;
#ifndef __CUDACC__
	static hpx::lcos::local::spinlock mtx;
	static unrolled_entry* allocate() {
		std::lock_guard<hpx::lcos::local::spinlock> lock(mtx);
		if (freelist.empty()) {
			unrolled_entry* new_entries;
			CUDA_MALLOC(new_entries, allocation_size);
			allocations.push(new_entries);
			for (int i = 0; i < allocation_size; i++) {
				freelist.push(new_entries + i);
			}
		}
		unrolled_entry* ptr;
		ptr = freelist.top();
		freelist.pop();
		return ptr;
	}
	static void deallocate(unrolled_entry* d) {
		std::lock_guard<hpx::lcos::local::spinlock> lock(mtx);
		freelist.push(d);
	}
#endif
	unrolled_entry* head;
	unrolled_entry* tail;
	int size_;
public:
	static void free_all() {
		while (allocations.size()) {
			CUDA_FREE(allocations.top());
			allocations.pop();
		}
		while(freelist.size()) {
			freelist.pop();
		}
	}
	class iterator {
	private:
		unrolled_entry* current;
		int index;
	public:
		CUDA_EXPORT
		inline iterator& operator++() {
			index++;
			if (index == entry_size && current->next) {
				index = 0;
				current = current->next;
			}
			return *this;
		}
		CUDA_EXPORT
		inline bool operator==(iterator other) const {
			return current == other.current && index == other.index;
		}
		CUDA_EXPORT
		inline bool operator!=(iterator other) const {
			return !(*this == other);
		}
		CUDA_EXPORT
		inline T operator*() const {
			return current->data[index];
		}
		friend class unrolled<T> ;
	};CUDA_EXPORT
	inline iterator begin() {
		iterator i;
		i.current = head;
		i.index = 0;
		return i;
	}
	CUDA_EXPORT
	inline iterator end() {
		iterator i;
		i.current = tail;
		i.index = tail->count;
		return i;
	}
	CUDA_EXPORT
	inline unrolled() {
		head = tail = nullptr;
		size_ = 0;
	}
	CUDA_EXPORT
	inline int size() const {
		return size_;
	}
#ifndef __CUDACC__
	inline ~unrolled() {
		clear();
	}
	inline void clear() {
		while (size()) {
			pop_top();
		}
	}
	inline void push_top(const T& data) {
		if (head == nullptr) {
			head = allocate();
			tail = head;
			head->prev = head->next = nullptr;
			head->count = 0;
		} else if (tail->count == entry_size) {
			tail->next = allocate();
			tail->next->prev = tail;
			tail = tail->next;
			tail->count = 0;
		}
		tail->data[tail->count] = data;
		tail->count++;
		size_++;
	}
	inline void pop_top() {
		tail->count--;
		if (tail->count == 0) {
			auto* freeme = tail;
			if (tail->prev == nullptr) {
				head = tail = nullptr;
			} else {
				tail->prev->next = nullptr;
				tail = tail->prev;
			}
			deallocate(freeme);
		}
		size_--;
	}
#else
	CUDA_EXPORT
	~unrolled() {
		printf( "Error - calling unrolled destructor from CUDACC\n");
	}
#endif
};

template<class T>
std::stack<typename unrolled<T>::unrolled_entry*> unrolled<T>::freelist;

template<class T>
std::stack<typename unrolled<T>::unrolled_entry*> unrolled<T>::allocations;

#ifndef __CUDACC__
template<class T>
hpx::lcos::local::spinlock unrolled<T>::mtx;
#endif

#endif /* DEQUE_HPP_ */

