#pragma once

#include <cstring>
#include <random>

#include "Layer.hh"

namespace net {

namespace {
template <std::size_t...>
class NetBase;

template <std::size_t IN>
class NetBase<IN> {
   public:
	using result_type = std::array<float, IN>;

	constexpr static auto in_size = IN;
	constexpr static auto out_size = IN;
	constexpr static auto data_size = 0;

	constexpr NetBase(float*) {}

	constexpr std::array<float, IN> operator()(
		const std::array<float, IN>& data) const {
		return data;
	}
};

template <std::size_t IN, std::size_t OUT, std::size_t... Ss>
class NetBase<IN, OUT, Ss...> : NetBase<OUT, Ss...> {
   public:
	using base_type = NetBase<OUT, Ss...>;
	using layer_type = Layer<IN, OUT>;
	using result_type = typename base_type::result_type;
	using feed_type = std::array<float, IN>;

	constexpr static auto in_size = IN;
	constexpr static auto out_size = IN;
	constexpr static auto data_size =
		layer_type::data_size + base_type::data_size;

	constexpr NetBase(float* data)
		: layer(data), base_type(data + layer_type::data_size) {}

	constexpr auto operator()(const std::array<float, IN>& data) const {
		return static_cast<const base_type*>(this)->operator()(
			reinterpret_cast<const layer_type*>(this)->operator()(data));
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

	constexpr Net() : data(), base_type(data.data()) {}
	constexpr Net(const Net& other) : Net() { *this = other; }

	constexpr auto operator()(
		const std::array<float, base_type::in_size>& data) {
		return static_cast<base_type*>(this)->operator()(data);
	}

	constexpr Net& operator=(const Net& other) {
		data = other.data;
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
		std::uniform_real_distribution<float> rand;
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
		std::memcpy(o.data.data() + 0, data.data() + 0, l * sizeof(float));
		std::memcpy(o.data.data() + r, data.data() + r,
					(base_type::data_size - r) * sizeof(float));
		std::memcpy(o.data.data() + l, other.data.data() + l,
					(r - l) * sizeof(float));

		return o;
	}

	Net& mutation(std::size_t count = 1) {
		std::random_device rd;
		std::uniform_int_distribution<std::size_t> rand_index(
			0, base_type::data_size - 1);
		std::uniform_real_distribution<float> rand_mutation(-50.f, 50.f);

		for (int i = 0; i < count; ++i) {
			auto mut_idx = rand_index(rd);
			data[mut_idx] += rand_mutation(rd);
		}

		return *this;
	}

   private:
	std::array<float, data_size> data;
	template <typename IStream>
	friend IStream& operator>>(IStream& s, Net<Ss...>& n) {
		return s.read(reinterpret_cast<char*>(n.data.data()),
					  net::Net<Ss...>::data_size * sizeof(float));
	}

	template <typename OStream>
	friend OStream& operator<<(OStream& s, const Net<Ss...>& n) {
		return s.write(reinterpret_cast<const char*>(n.data.data()),
					   net::Net<Ss...>::data_size * sizeof(float));
	}
};
}  // namespace net

// vim: set ts=4 sw=4 :
