#include <algorithm>
#include <cassert>
#include <iostream>
#include <span>
#include <tuple>

#include <Net.hh>

#include "common.hh"

using namespace std::literals;

// type helper SimpleNet
using net_type = net::SimpleNet<2, 3, 2>;

// float is score of net
using tuple_type = std::tuple<float, net_type>;

// use this count for generating new generation
constexpr auto to_use = 25;

// stop when score greater or equal min_score
constexpr auto min_score =
	7.5f; /* maximum score is (is_true{1} + is_false{1}) * tests{4} = 8 */

// helper function for computing count of networks
constexpr std::size_t compute_size(std::size_t x) {
	if (x == 1)
		return 0;
	return x - 1 + compute_size(x - 1);
}

// compute efficient count of networks
constexpr auto nets_size = to_use + compute_size(to_use);

// return score for expected true result
auto check_true(const net_type::result_type& res) {
	return res[0] - res[1];
};

// return score for expected false result
auto check_false(const net_type::result_type& res) {
	return res[1] - res[0];
};

// compare networks by score
constexpr bool compare(const tuple_type& a, const tuple_type& b) {
	return std::get<float>(a) > std::get<float>(b);
}

int main() {
	// set time limit
	TimeLimit timelimit(5s);

	// allocate nets
	std::array<tuple_type, nets_size> nets;

	// randomize networks
	for (auto& n : nets) {
		std::get<net_type>(n).rand();
	}

	// input datas for networks
	net_type::feed_type xor_data_in[4] = {
		{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};

	for (;;) {
		for (auto& n : nets) {
			// get reference to network score
			auto& result = std::get<float>(n);

			// get reference to network
			auto& nn = std::get<net_type>(n);

			// reset score
			result = 0.f;

			// temporary variable for storing result
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
		// sort networks by score
		std::sort(std::begin(nets), std::end(nets), compare);

		// compare best score with min_score
		auto best_score = std::get<float>(nets[0]);
		if (best_score >= min_score)
			std::exit(0);

		// generate new generation
		std::size_t child_id = to_use;
		for (size_t i = 0; i < to_use; ++i) {
			for (size_t j = i + 1; j < to_use; ++j) {
				auto& child = std::get<net_type>(nets[child_id]);
				auto& parent1 = std::get<net_type>(nets[i]);
				auto& parent2 = std::get<net_type>(nets[j]);

				child = parent1 + parent2 + 5;

				++child_id;
			}
		}
	}
}

// vim: set ts=4 sw=4 :
