#include <cassert>
#include <sstream>

#include <Net.hh>

int main() {
	net::SimpleNet<2, 3, 5> n;

	n++;
	++n;
	n + 5;
	n + n;
	n.rand();
	n.merge(n);
	n.mutation(2);

	std::stringstream ss;

	ss << n;

	net::SimpleNet<2, 3, 5> n2;
	ss >> n2;

	assert(n == n2);

	return 0;
}

// vim: set ts=4 sw=4 :
