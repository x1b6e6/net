#include <Net.hh>

using net_type = net::Net<2, 2>;

int main() {
	try {
		// check throwing exception at invalid argument
		net_type n(1);
		return 1;
	} catch (std::invalid_argument& e) {
	}

	net_type n(5);

	net_type::feed_type data;

	n.feed(data);
	n.count_score([](const net_type::result_type&) { return 0; });
	n.next();
	n.next<std::less>(5);

	auto a = n.rand();
	auto b = n.reset_score();
	auto c = n.best_score();
	auto d = n.result();
	auto e = n.best_result();
	auto f = n.score();

	return 0;
}

// vim: set ts=4 sw=4 :
