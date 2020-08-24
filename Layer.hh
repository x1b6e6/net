#pragma once

#include <array>
#include <memory>

#include "Neuron.hh"

namespace net {

namespace {

template <std::size_t IN, std::size_t OUT>
class LayerBase : LayerBase<IN, OUT - 1> {
	using neuron_type = Neuron<IN>;
	using base_type = LayerBase<IN, OUT - 1>;

   public:
	constexpr LayerBase(float* data)
		: neuron(data), base_type(data + neuron_type::data_size()) {}

	constexpr float operator()(const std::array<float, IN>& data) const {
		return neuron(data);
	}

	constexpr static std::size_t data_size() {
		return neuron_type::data_size() + base_type::data_size();
	}

	constexpr const neuron_type& neuron_id(std::size_t i) const {
		if (i == 0)
			return neuron;
		return static_cast<const base_type*>(this)->neuron_id(i - 1);
	}

   private:
	neuron_type neuron;
};

template <std::size_t IN>
class LayerBase<IN, 1> {
	using neuron_type = Neuron<IN>;

   public:
	constexpr LayerBase(float* data) : neuron(data) {}
	constexpr float operator()(const std::array<float, IN>& data) const {
		return neuron(data);
	}
	constexpr static std::size_t data_size() {
		return neuron_type::data_size();
	}

	constexpr const neuron_type& neuron_id(std::size_t i) const {
		if (i != 0)
			throw std::runtime_error{"neuron_id"};
		return neuron;
	}

   private:
	neuron_type neuron;
};

}  // namespace

template <std::size_t IN, std::size_t OUT>
class Layer : LayerBase<IN, OUT> {
   public:
	using neuron_type = Neuron<IN>;
	using base_type = LayerBase<IN, OUT>;
	constexpr Layer(float* data) : base_type(data) {}

	std::array<float, OUT> operator()(const std::array<float, IN>& data) const {
		std::array<float, OUT> o;

#pragma omp simd
		for (int i = 0; i < OUT; ++i) {
			o[i] = neuron_id(i)(data);
		}

		return o;
	}

	constexpr static std::size_t data_size() { return base_type::data_size(); }

   private:
	constexpr const neuron_type& neuron_id(std::size_t i) const {
		return static_cast<const base_type*>(this)->neuron_id(i);
	}
};

}  // namespace net

// vim: set ts=4 sw=4 :
