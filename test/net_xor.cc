#include <cassert>
#include <iostream>

#include <Net.hh>

#include "common.hh"

using namespace std::literals;

// using this type
using net_type = net::Net<2, 3, 2>;

// stop train when score greater or equal min_score
constexpr auto min_score =
	7.5f; /* maximum score is (is_true{1} + is_false{1}) * tests{4} = 8 */

// return score for expected true result
net_type::score_type check_true(const net_type::result_type& res) {
	return res[0] - res[1];
};

// return score for expected false result
net_type::score_type check_false(const net_type::result_type& res) {
	return res[1] - res[0];
};

int main() {
	// terminate programm after 5 seconds
	TimeLimit timelimit(5s);

	// create net::Net object with to_use=25 immutable=0
	net_type n(25);

	// randomize networks
	n.rand();

	// input data variants
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

		// compare the best score with min_score
		auto best_score = n.best_score<std::greater>();
		if (best_score >= min_score)
			std::exit(0);

		// generate new generate with 5 mutations
		n.next<std::greater>(5);
	}
}

// vim: set ts=4 sw=4 :
