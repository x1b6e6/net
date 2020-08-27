#pragma once

#include <array>
#include <memory>

#include "Neuron.hh"

namespace net {

template <std::size_t...>
class Layer;

template <std::size_t IN, std::size_t OUT>
class Layer<IN, OUT> {
   public:
	using neuron_type = Neuron<IN>;
	using feed_type = typename neuron_type::feed_type;
	using result_type = std::array<typename neuron_type::result_type, OUT>;
	using my_result_type = result_type;

	constexpr static auto data_size = neuron_type::data_size * OUT;

	constexpr Layer(store_type* data = nullptr) {
		if (data != nullptr) {
			for (std::size_t i = 0; i < OUT; ++i) {
				neurons[i] = neuron_type(data);
				data += neuron_type::data_size;
			}
		}
	}

	constexpr Layer& operator=(const Layer& other) {
		for (std::size_t i = 0; i < OUT; ++i) {
			neurons[i] = other.neurons[i];
		}
		return *this;
	}

	constexpr result_type operator()(const feed_type& data) const {
		result_type o{};

		for (std::size_t i = 0; i < OUT; ++i) {
			o[i] = neurons[i](data);
		}

		return o;
	}

   private:
	neuron_type neurons[OUT];
};

template <std::size_t IN, std::size_t OUT, std::size_t... Ss>
class Layer<IN, OUT, Ss...> {
   public:
	using base_type = Layer<OUT, Ss...>;
	using neuron_type = Neuron<IN>;
	using feed_type = typename neuron_type::feed_type;
	using result_type = typename base_type::result_type;
	using my_result_type = std::array<typename neuron_type::result_type, OUT>;

	constexpr static auto current_data_size = neuron_type::data_size * OUT;
	constexpr static auto data_size = base_type::data_size + current_data_size;

	constexpr Layer(store_type* data = nullptr) {
		if (data != nullptr) {
			base = base_type(data + current_data_size);
			for (std::size_t i = 0; i < OUT; ++i) {
				neurons[i] = neuron_type(data);
				data += neuron_type::data_size;
			}
		}
	}

	constexpr Layer& operator=(const Layer& other) {
		for (std::size_t i = 0; i < OUT; ++i) {
			neurons[i] = other.neurons[i];
		}
		return *this;
	}

	result_type operator()(const feed_type& data) const {
		my_result_type o{};

		for (std::size_t i = 0; i < OUT; ++i) {
			o[i] = neurons[i](data);
		}

		return base(o);
	}

   private:
	base_type base;
	neuron_type neurons[OUT];
};

}  // namespace net

// vim: set ts=4 sw=4 :
