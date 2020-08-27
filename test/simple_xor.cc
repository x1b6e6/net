#include <algorithm>
#include <cassert>
#include <tuple>

#include <iostream>

#include <Net.hh>

template <typename T>
constexpr T abs(T x) {
	return x > T{} ? x : -x;
}

template <typename T, size_t S>
constexpr T diff(const std::array<T, S>& a, const std::array<T, S>& b) {
	T o{};
	for (size_t i = 0; i < S; ++i) {
		o += abs(a[i] - b[i]);
	}
	return o;
}

int main() {
	using net_type = net::Net<2, 3, 2>;
	using tuple_type = std::tuple<float, net_type>;

	constexpr auto to_use = 25;
	constexpr auto nets_size = to_use * to_use;
	constexpr auto max_generations = 10000;

	std::array<tuple_type, nets_size> nets;

	for (auto& n : nets) {
		std::get<net_type>(n).rand();
	}

	std::array<float, 2> xor_data_in[4] = {
		{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};

	std::array<float, 2> xor_data_out[4] = {
		{0.f, 1.f}, {1.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}};

	size_t generation;
	for (generation = 1; generation < max_generations; ++generation) {
		for (auto& n : nets) {
			auto& result = std::get<float>(n);
			auto& nn = std::get<net_type>(n);
			result = 0.f;
			for (size_t i = 0; i < 4; ++i) {
				auto res = nn(xor_data_in[i]);
				result += diff(res, xor_data_out[i]);
			}
		}
		std::sort(std::begin(nets), std::end(nets),
				  [](const tuple_type& a, const tuple_type& b) {
					  return std::get<float>(a) < std::get<float>(b);
				  });
		auto min_error = std::get<float>(nets[0]);
		if (min_error < 1.f)
			break;
		for (size_t i = 0; i < to_use; ++i) {
			for (size_t j = i + 1; j < to_use; ++j) {
				size_t child_id = to_use - 1 + i * to_use + j;
				auto& child = std::get<net_type>(nets[child_id]);
				auto& parent1 = std::get<net_type>(nets[i]);
				auto& parent2 = std::get<net_type>(nets[j]);

				child =
					parent1 + parent2 + (static_cast<int>(min_error) * 2 + 1);
			}
		}
	}
	assert(generation != max_generations);

	return 0;
}

// vim: set ts=4 sw=4 :