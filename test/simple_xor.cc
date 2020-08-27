#include <algorithm>
#include <cassert>
#include <iostream>
#include <span>
#include <tuple>

#include <Net.hh>

using net_type = net::SimpleNet<2, 3, 2>;
using tuple_type = std::tuple<float, net_type>;

constexpr auto to_use = 25;
constexpr auto max_generations = 10000;
constexpr auto min_score =
	7.5f; /* maximum score is (is_true{1} + is_false{1}) * tests{4} = 8 */

constexpr auto nets_size = to_use * to_use;

auto check_true(const net_type::result_type& res) {
	return res[0] - res[1];
};

auto check_false(const net_type::result_type& res) {
	return res[1] - res[0];
};

int main() {
	std::array<tuple_type, nets_size> nets;

	for (auto& n : nets) {
		std::get<net_type>(n).rand();
	}

	net_type::feed_type xor_data_in[4] = {
		{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};

	size_t generation;
	for (generation = 1; generation < max_generations; ++generation) {
		for (auto& n : nets) {
			auto& result = std::get<float>(n);
			auto& nn = std::get<net_type>(n);
			result = 0.f;
			net_type::result_type res;

			res = nn(xor_data_in[0]);
			result += check_false(res);

			res = nn(xor_data_in[1]);
			result += check_true(res);

			res = nn(xor_data_in[2]);
			result += check_true(res);

			res = nn(xor_data_in[3]);
			result += check_false(res);
		}
		std::sort(std::begin(nets), std::end(nets),
				  [](const tuple_type& a, const tuple_type& b) {
					  return std::get<float>(a) > std::get<float>(b);
				  });
		auto score = std::get<float>(nets[0]);
		std::cout << score << '\n';
		if (score >= min_score)
			break;
		for (size_t i = 0; i < to_use; ++i) {
			for (size_t j = i + 1; j < to_use; ++j) {
				size_t child_id = to_use - 1 + i * to_use + j;
				auto& child = std::get<net_type>(nets[child_id]);
				auto& parent1 = std::get<net_type>(nets[i]);
				auto& parent2 = std::get<net_type>(nets[j]);

				child = parent1 + parent2 + 5;
			}
		}
	}
	assert(generation != max_generations);

	return 0;
}

// vim: set ts=4 sw=4 :
