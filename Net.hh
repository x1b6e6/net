#pragma once

#include <cstring>
#include <random>

#include "Layer.hh"

namespace net {

template <std::size_t... Ss>
class Net {
	using layer_type = Layer<Ss...>;

   public:
	using result_type = typename layer_type::result_type;
	using feed_type = typename layer_type::feed_type;

	constexpr static auto data_size = layer_type::data_size;

	constexpr Net() : layer(data) {}
	constexpr Net(const Net& other) : Net() { *this = other; }

	constexpr result_type operator()(const feed_type& data) {
		return layer(data);
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

	Net& mutation(std::size_t count = 1) {
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
