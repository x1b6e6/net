#pragma once

#include <array>
#include <memory>

#include "Neuron.hh"

namespace net {

template <std::size_t IN, std::size_t OUT>
class Layer {
   public:
	using neuron_type = Neuron<IN>;

	constexpr static auto data_size = OUT * neuron_type::data_size;

	constexpr Layer(float* data = nullptr) {
		if (data != nullptr) {
			for (int i = 0; i < OUT; ++i) {
				neurons[i] = neuron_type(data);
				data += neuron_type::data_size;
			}
		}
	}

	constexpr void operator=(const Layer& other) { neurons = other.neurons; }

	std::array<float, OUT> operator()(const std::array<float, IN>& data) const {
		std::array<float, OUT> o;

		for (int i = 0; i < OUT; ++i) {
			o[i] = neurons[i](data);
		}

		return o;
	}

   private:
	std::array<neuron_type, OUT> neurons;
};

}  // namespace net

// vim: set ts=4 sw=4 :
