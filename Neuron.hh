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
	constexpr Neuron(float* data) : ndata(data) {}

	constexpr float operator()(const std::array<float, IN>& data) const {
		float o = 0.f;
		for (int i = 0; i < IN; ++i) {
			o += data[i] * ndata[i << 1] + ndata[(i << 1) + 1];
		}
		return sigmoid(o);
	}
	constexpr static std::size_t data_size() { return 2 * IN; }

   private:
	float* ndata;
};

}  // namespace net

// vim: set ts=4 sw=4 :
