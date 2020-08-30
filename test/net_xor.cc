#include <cassert>
#include <iostream>

#include <Net.hh>

#include "common.hh"

using namespace std::literals;

using net_type = net::Net<2, 3, 2>;

constexpr auto min_score =
	7.5f; /* maximum score is (is_true{1} + is_false{1}) * tests{4} = 8 */

net_type::score_type check_true(const net_type::result_type& res) {
	return res[0] - res[1];
};

net_type::score_type check_false(const net_type::result_type& res) {
	return res[1] - res[0];
};

int main() {
	TimeLimit timelimit(5s);
	net_type n(25);

	net_type::feed_type xor_data_in[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

	for (;;) {
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
			std::exit(0);

		n.next<std::greater>(5);
	}
}

// vim: set ts=4 sw=4 :
