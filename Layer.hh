#pragma once

#include <array>
#include <memory>

#include "Neuron.hh"

namespace net {

template <std::size_t IN, std::size_t OUT>
class Layer {
   public:
	using neuron_type = Neuron<IN>;
	using feed_type = typename neuron_type::feed_type;
	using result_type = std::array<typename neuron_type::result_type, OUT>;

	constexpr static auto data_size = OUT * neuron_type::data_size;

	constexpr Layer(store_type* data = nullptr) {
		if (data != nullptr) {
			for (std::size_t i = 0; i < OUT; ++i) {
				neurons[i] = neuron_type(data);
				data += neuron_type::data_size;
			}
		}
	}

	constexpr void operator=(const Layer& other) { neurons = other.neurons; }

	result_type operator()(const feed_type& data) const {
		result_type o;

		for (std::size_t i = 0; i < OUT; ++i) {
			o[i] = neurons[i](data);
		}

		return o;
	}

   private:
	neuron_type neurons[OUT];
};

}  // namespace net

// vim: set ts=4 sw=4 :
