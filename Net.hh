#pragma once

#include <array>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>

namespace net {
using store_type = float;

namespace {
constexpr auto abs(auto x) {
	return x > 0 ? x : -x;
}
constexpr auto sigmoid(auto x) {
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
	constexpr static auto in_size = IN;
	constexpr static auto out_size = OUT;

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
	using next_layer_type = Layer<OUT, Ss...>;
	using neuron_type = Neuron<IN>;
	using feed_type = typename neuron_type::feed_type;
	using result_type = typename next_layer_type::result_type;
	using my_result_type = std::array<typename neuron_type::result_type, OUT>;

	constexpr static auto current_data_size = neuron_type::data_size * OUT;
	constexpr static auto data_size =
		next_layer_type::data_size + current_data_size;
	constexpr static auto in_size = IN;
	constexpr static auto out_size = next_layer_type::out_size;

	constexpr Layer(store_type* data = nullptr) {
		if (data != nullptr) {
			next_layer = next_layer_type(data + current_data_size);
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

		return next_layer(o);
	}

   private:
	next_layer_type next_layer;
	neuron_type neurons[OUT];
};

template <std::size_t... Ss>
struct size_length;

template <std::size_t S, std::size_t... Ss>
struct size_length<S, Ss...>
	: std::integral_constant<std::size_t, 1 + size_length<Ss...>::value> {};

template <std::size_t S>
struct size_length<S> : std::integral_constant<std::size_t, 1> {};

}  // namespace

template <std::size_t... Ss>
requires(size_length<Ss...>::value >= 2) class SimpleNet {
   public:
	using layer_type = Layer<Ss...>;
	using result_type = typename layer_type::result_type;
	using feed_type = typename layer_type::feed_type;

	constexpr static auto data_size = layer_type::data_size;
	constexpr static auto in_size = layer_type::in_size;
	constexpr static auto out_size = layer_type::out_size;

	constexpr SimpleNet() : data(new store_type[data_size]), layer(data) {}
	constexpr SimpleNet(const SimpleNet& other) : SimpleNet() { *this = other; }
	constexpr ~SimpleNet() { delete[] data; }

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
		for (std::size_t i = 0; i < data_size; ++i) {
			data[i] = rand(rd);
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
	store_type* data;
	layer_type layer;

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

template <std::size_t... Ss>
requires(size_length<Ss...>::value >= 2) class Net {
   public:
	using net_type = SimpleNet<Ss...>;
	using score_type = store_type;
	using result_type = typename net_type::result_type;
	using feed_type = typename net_type::feed_type;

	struct tuple_type : std::tuple<score_type, net_type, result_type> {
		friend constexpr auto operator<=>(const tuple_type& a,
										  const tuple_type& b) {
			return std::get<score_type>(a) <=> std::get<score_type>(b);
		}
	};

	constexpr static auto in_size = net_type::in_size;
	constexpr static auto out_size = net_type::out_size;

	constexpr Net(std::size_t to_use_)
		: to_use(to_use_), nets_size(to_use_ * to_use_) {
		if (to_use < 3) {
			throw std::invalid_argument{"net::Net to_use should be >=2"};
		}
		nets.resize(nets_size);
	}

	Net& rand() {
		for (auto& n : nets) {
			auto& nn = std::get<net_type>(n);
			nn.rand();
		}
	}

	constexpr Net& feed(const feed_type& data) {
		/* TODO: add multithreading */
		for (auto& n : nets) {
			auto& nn = std::get<net_type>(n);
			auto& res = std::get<result_type>(n);

			res = nn(data);
		}
		return *this;
	}

	template <typename Fn>
	constexpr Net& count_score(Fn fn) {
		/* TODO: add multithreading */
		for (auto& n : nets) {
			auto& score = std::get<score_type>(n);
			const auto& res = std::get<result_type>(n);

			score += fn(res);
		}
		return *this;
	}

	template <template <typename> typename Compare = std::greater>
	constexpr Net& next(int mutation = 2,
						Compare<tuple_type> comp = Compare<tuple_type>()) {
		std::sort(std::begin(nets), std::end(nets), comp);

		for (std::size_t i = 0; i < to_use; ++i) {
			for (std::size_t j = i + 1; j < to_use; ++j) {
				std::size_t child_id = to_use - 1 + to_use * i + j;

				auto& child = std::get<net_type>(nets[child_id]);
				auto& parent1 = std::get<net_type>(nets[i]);
				auto& parent2 = std::get<net_type>(nets[j]);

				child = parent1 + parent2 + mutation;
			}
		}

		return *this;
	}

	template <template <typename> typename Compare = std::greater>
	constexpr score_type best_score(Compare<score_type> comp = {}) const {
		score_type best = std::get<score_type>(nets[0]);

		for (auto& n : nets) {
			auto& score = std::get<score_type>(n);
			if (comp(score, best)) {
				best = score;
			}
		}

		return best;
	}

	constexpr Net& reset_score() {
		/* TODO: add multithreading */
		for (auto& n : nets) {
			std::get<score_type>(n) = score_type{};
		}
		return *this;
	}

	constexpr result_type result() {
		result_type o;

		for (auto& n : nets) {
			const auto& res = std::get<result_type>(n);
			/* TODO: add multithreading */
			for (std::size_t i = 0; i < out_size; ++i) {
				o[i] += res[i];
			}
		}
		/* TODO: add multithreading */
		for (std::size_t i = 0; i < out_size; ++i) {
			o[i] /= nets_size;
		}

		return o;
	}

	constexpr result_type best_result() {
		return std::get<result_type>(nets[0]);
	}

   private:
	const std::size_t to_use;
	const std::size_t nets_size;
	std::vector<tuple_type> nets;
};

}  // namespace net

// vim: set ts=4 sw=4 :
