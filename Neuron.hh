#pragma once

#include <array>
#include <cstddef>

namespace net {

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
	constexpr static auto data_size = IN * 2;

	constexpr Neuron(float* data = nullptr) : ndata(data) {}

	constexpr void operator=(const Neuron& other) { ndata = other.ndata; }

	constexpr float operator()(const std::array<float, IN>& data) const {
		float o = 0.f;
		for (int i = 0; i < IN; ++i) {
			o += data[i] * ndata[i << 1] + ndata[(i << 1) + 1];
		}
		return sigmoid(o);
	}

   private:
	float* ndata;
};

}  // namespace net

// vim: set ts=4 sw=4 :
