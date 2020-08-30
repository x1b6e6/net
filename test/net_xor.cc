#include <cassert>
#include <iostream>

#include <Net.hh>

using net_type = net::Net<2, 3, 2>;

constexpr auto max_generations = 10000;
constexpr auto min_score =
	7.5f; /* maximum score is (is_true{1} + is_false{1}) * tests{4} = 8 */

net_type::score_type check_true(const net_type::result_type& res) {
	return res[0] - res[1];
};

net_type::score_type check_false(const net_type::result_type& res) {
	return res[1] - res[0];
};

int main() {
	net_type n(25);

	net_type::feed_type xor_data_in[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

	int generation;
	for (generation = 1; generation < max_generations; ++generation) {
		n.reset_score();

		n.feed(xor_data_in[0]);
		n.count_score(check_false);

		n.feed(xor_data_in[1]);
		n.count_score(check_true);

		n.feed(xor_data_in[2]);
		n.count_score(check_true);

		n.feed(xor_data_in[3]);
		n.count_score(check_false);

		auto score = n.best_score<std::greater>();
		std::cout << score << '\n';
		if (score >= min_score)
			break;

		n.next<std::greater>(5);
	}
	assert(generation != max_generations);

	return 0;
}
