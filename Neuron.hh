#pragma once

#include <array>
#include <cstddef>

namespace net {

using store_type = float;

namespace {
constexpr float abs(float x) {
	return x > 0.f ? x : -x;
}
constexpr float sigmoid(float x) {
	return x / (1 + abs(x));
}
}  // namespace

template <std::size_t IN>
class Neuron {
   public:
	using feed_type = std::array<store_type, IN>;
	using result_type = store_type;

	constexpr static auto data_size = IN * 2;

	constexpr Neuron(store_type* data = nullptr) : ndata(data) {}

	constexpr void operator=(const Neuron& other) { ndata = other.ndata; }

	constexpr result_type operator()(const feed_type& data) const {
		result_type o = 0.f;
		for (std::size_t i = 0; i < IN; ++i) {
			o += data[i] * ndata[i << 1] + ndata[(i << 1) + 1];
		}
		return sigmoid(o);
	}

   private:
	store_type* ndata;
};

}  // namespace net

// vim: set ts=4 sw=4 :
