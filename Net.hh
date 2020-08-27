#pragma once

#include <cstring>
#include <random>

#include "Layer.hh"

namespace net {

namespace {
template <std::size_t...>
class NetBase;

template <std::size_t IN, std::size_t OUT>
class NetBase<IN, OUT> {
   public:
	using layer_type = Layer<IN, OUT>;
	using result_type = typename layer_type::result_type;
	using feed_type = typename layer_type::feed_type;

	constexpr static auto in_size = IN;
	constexpr static auto out_size = OUT;
	constexpr static auto data_size = layer_type::data_size;

	constexpr NetBase(store_type* data) : layer(data) {}

	constexpr result_type operator()(const feed_type& data) const {
		return layer(data);
	}

   private:
	layer_type layer;
};

template <std::size_t IN, std::size_t OUT, std::size_t... Ss>
class NetBase<IN, OUT, Ss...> : NetBase<OUT, Ss...> {
   public:
	using base_type = NetBase<OUT, Ss...>;
	using layer_type = Layer<IN, OUT>;
	using result_type = typename base_type::result_type;
	using feed_type = typename layer_type::feed_type;

	constexpr static auto in_size = IN;
	constexpr static auto out_size = IN;
	constexpr static auto data_size =
		layer_type::data_size + base_type::data_size;

	constexpr NetBase(store_type* data)
		: layer(data), base_type(data + layer_type::data_size) {}

	constexpr result_type operator()(const feed_type& data) const {
		return static_cast<const base_type*>(this)->operator()(layer(data));
	}

   private:
	layer_type layer;
};

}  // namespace

template <std::size_t... Ss>
class Net : NetBase<Ss...> {
	using base_type = NetBase<Ss...>;

   public:
	using result_type = typename base_type::result_type;
	using feed_type = typename base_type::feed_type;

	constexpr static auto data_size = base_type::data_size;

	constexpr Net() : base_type(data) {}
	constexpr Net(const Net& other) : Net() { *this = other; }

	constexpr result_type operator()(const feed_type& data) {
		return static_cast<base_type*>(this)->operator()(data);
	}

	constexpr Net& operator=(const Net& other) {
		std::memcpy(data, other.data, data_size * sizeof(store_type));

		return *this;
	}

	Net operator+(const Net& other) const { return merge(other); }
	Net& operator+(int mut) { return mutation(mut); }
	Net& operator++() { return mutation(); }
	Net operator++(int z) const {
		Net o{*this};
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

	Net merge(const Net& other) const {
		Net o(*this);

		std::random_device rd;
		std::uniform_int_distribution<std::size_t> rand_index(
			0, base_type::data_size - 1);

		auto r = rand_index(rd);
		auto l = rand_index(rd);
		if (l > r)
			std::swap(l, r);

		std::memcpy(o.data + 0, data + 0, l * sizeof(store_type));
		std::memcpy(o.data + r, data + r,
					(base_type::data_size - r) * sizeof(store_type));
		std::memcpy(o.data + l, other.data + l, (r - l) * sizeof(store_type));

		return o;
	}

	Net& mutation(std::size_t count = 1) {
		std::random_device rd;
		std::uniform_int_distribution<std::size_t> rand_index(
			0, base_type::data_size - 1);
		std::uniform_real_distribution<store_type> rand_mutation(-50.f, 50.f);

		for (std::size_t i = 0; i < count; ++i) {
			auto mut_idx = rand_index(rd);
			data[mut_idx] += rand_mutation(rd);
		}

		return *this;
	}

   private:
	store_type data[data_size];
	template <typename IStream>
	friend IStream& operator>>(IStream& s, Net<Ss...>& n) {
		return s.read(reinterpret_cast<char*>(n.data),
					  net::Net<Ss...>::data_size * sizeof(store_type));
	}

	template <typename OStream>
	friend OStream& operator<<(OStream& s, const Net<Ss...>& n) {
		return s.write(reinterpret_cast<const char*>(n.data),
					   net::Net<Ss...>::data_size * sizeof(store_type));
	}
};
}  // namespace net

// vim: set ts=4 sw=4 :
