#pragma once

#include <array>
#include <cstring>
#include <random>

namespace net {
using store_type = float;

namespace {
constexpr float abs(float x) {
	return x > 0.f ? x : -x;
}
constexpr float sigmoid(float x) {
	return x / (1 + abs(x));
}

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

}  // namespace

template <std::size_t... Ss>
class SimpleNet {
	using layer_type = Layer<Ss...>;

   public:
	using result_type = typename layer_type::result_type;
	using feed_type = typename layer_type::feed_type;

	constexpr static auto data_size = layer_type::data_size;

	constexpr SimpleNet() : layer(data) {}
	constexpr SimpleNet(const SimpleNet& other) : SimpleNet() { *this = other; }

	constexpr result_type operator()(const feed_type& data) {
		return layer(data);
	}

	constexpr SimpleNet& operator=(const SimpleNet& other) {
		std::memcpy(data, other.data, data_size * sizeof(store_type));

		return *this;
	}

	SimpleNet operator+(const SimpleNet& other) const { return merge(other); }
	SimpleNet& operator+(int mut) { return mutation(mut); }
	SimpleNet& operator++() { return mutation(); }
	SimpleNet operator++(int z) const {
		SimpleNet o{*this};
		mutation(z ? z : 1);
		return o;
	}

	void rand() {
		std::random_device rd;
		std::uniform_real_distribution<store_type> rand;
		for (auto& n : data) {
			n = rand(rd);
		}
	}

	SimpleNet merge(const SimpleNet& other) const {
		SimpleNet o(*this);

		std::random_device rd;
		std::uniform_int_distribution<std::size_t> rand_index(
			0, layer_type::data_size - 1);

		auto r = rand_index(rd);
		auto l = rand_index(rd);
		if (l > r)
			std::swap(l, r);

		std::memcpy(o.data + 0, data + 0, l * sizeof(store_type));
		std::memcpy(o.data + r, data + r,
					(layer_type::data_size - r) * sizeof(store_type));
		std::memcpy(o.data + l, other.data + l, (r - l) * sizeof(store_type));

		return o;
	}

	SimpleNet& mutation(std::size_t count = 1) {
		std::random_device rd;
		std::uniform_int_distribution<std::size_t> rand_index(
			0, layer_type::data_size - 1);
		std::uniform_real_distribution<store_type> rand_mutation(-50.f, 50.f);

		for (std::size_t i = 0; i < count; ++i) {
			auto mut_idx = rand_index(rd);
			data[mut_idx] += rand_mutation(rd);
		}

		return *this;
	}

   private:
	layer_type layer;
	store_type data[data_size];
	template <typename IStream>
	friend IStream& operator>>(IStream& s, SimpleNet<Ss...>& n) {
		return s.read(reinterpret_cast<char*>(n.data),
					  net::SimpleNet<Ss...>::data_size * sizeof(store_type));
	}

	template <typename OStream>
	friend OStream& operator<<(OStream& s, const SimpleNet<Ss...>& n) {
		return s.write(reinterpret_cast<const char*>(n.data),
					   net::SimpleNet<Ss...>::data_size * sizeof(store_type));
	}
};
}  // namespace net

// vim: set ts=4 sw=4 :
