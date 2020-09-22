#include <Net.hh>

void nothing(int& val) {
	static int c = 0;
	val = c++;
}
void cnothing(const int& val) {
	static int c = 0;
	c = val;
}

int main() {
	net::array<int, 5> arr;

	std::for_each(arr.begin(), arr.end(), nothing);
	std::for_each(arr.cbegin(), arr.cend(), cnothing);
	std::for_each(arr.rbegin(), arr.rend(), nothing);
	std::for_each(arr.crbegin(), arr.crend(), cnothing);

	return 0;
}