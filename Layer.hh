#pragma once

#include <array>
#include <memory>

#include "Neuron.hh"

namespace net {

template <std::size_t IN, std::size_t OUT>
class Layer {
   public:
	using neuron_type = Neuron<IN>;

	constexpr Layer(float* data = nullptr) {
		if (data != nullptr) {
			for (int i = 0; i < OUT; ++i) {
				neurons[i] = neuron_type(data);
				data += neuron_type::data_size();
			}
		}
	}

	constexpr void operator=(const Layer& other) { neurons = other.neurons; }

	std::array<float, OUT> operator()(const std::array<float, IN>& data) const {
		std::array<float, OUT> o;

#pragma omp parallel for simd num_threads(OUT)
		for (int i = 0; i < OUT; ++i) {
			o[i] = neurons[i](data);
		}

		return o;
	}

	constexpr static std::size_t data_size() {
		return OUT * neuron_type::data_size();
	}

   private:
	std::array<neuron_type, OUT> neurons;
};

}  // namespace net

// vim: set ts=4 sw=4 :
