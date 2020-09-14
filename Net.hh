#pragma once

#include <array>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <utility>

namespace net {
// main type used for storing, input and output data
using store_type = float;

// container for input and output data
template <typename T, std::size_t S>
class array {
   public:
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using reference = value_type&;
	using const_reference = const value_type&;
	using pointer = T*;
	using const_pointer = const T*;

	// FIXME: using std::iterator instead of raw pointer
	using iterator = T*;
	using const_iterator = const T*;

	// TODO:
	// using reverse_iterator =
	// using const_reverse_iterator =

	constexpr array() : _data(new T[S]) {}
	constexpr array(const std::initializer_list<T>& list) : array() {
		std::memcpy(data(), list.begin(), list.size() * sizeof(T));
	}
	constexpr array(const array& other) : array() { *this = other; }
	constexpr ~array() { release(); }

	constexpr reference operator[](size_type index) {
		return *(data() + index);
	}
	constexpr const_reference operator[](size_type index) const {
		return *(data() + index);
	}

	constexpr reference at(size_type index) {
		if (index >= size())
			throw std::out_of_range("index out of bounds");
		return *(data() + index);
	}
	constexpr const_reference at(size_type index) const {
		if (index >= size())
			throw std::out_of_range("index out of bounds");
		return *(data() + index);
	}

	constexpr bool empty() const noexcept { return size() == 0; }
	constexpr size_type size() const noexcept { return S; }
	constexpr size_type max_size() const noexcept { return size(); }

	constexpr void operator=(const array& other) {
		std::memcpy(data(), other.data(), size() * sizeof(T));
	}

	constexpr void release() {
		if (_data != nullptr) {
			delete[] _data;
			_data = nullptr;
		}
	}

	constexpr reference front() { return operator[](0); }
	constexpr const_reference front() const { return operator[](0); }

	constexpr reference back() { return operator[](size() - 1); }
	constexpr const_reference back() const { return operator[](size() - 1); }

	constexpr pointer data() noexcept { return _data; }
	constexpr const_pointer data() const noexcept { return _data; }

	constexpr iterator begin() noexcept { return data(); }
	constexpr const_iterator begin() const noexcept { return data(); }
	constexpr const_iterator cbegin() const noexcept { return data(); }

	constexpr iterator end() noexcept { return data() + size(); }
	constexpr const_iterator end() const noexcept { return data() + size(); }
	constexpr const_iterator cend() const noexcept { return data() + size(); }

	// TODO:
	// rbegin()
	// rend()
	// crbegin()
	// crend()

	constexpr void fill(const T& val) { std::memset(data(), val, size()); }

	constexpr void swap(array& other) noexcept(std::is_nothrow_swappable_v<T>) {
		auto mit = begin();
		auto oit = other.begin();

		auto mend = end();

		while (mit != mend) {
			std::swap(*mit, *oit);

			++mit;
			++oit;
		}
	}

   private:
	T* _data;
};

namespace {
// absolute value
constexpr auto abs(auto x) {
	return x > 0 ? x : -x;
}

// sigmoid
constexpr auto sigmoid(auto x) {
	return x / (1 + abs(x));
}

// class Neuron contained pointers (no ownership) to his data
template <std::size_t IN>
class Neuron {
   public:
	// type of input
	using feed_type = array<store_type, IN>;
	// type of output
	using result_type = store_type;

	// number of store_type what it need
	constexpr static auto data_size = IN * 2;
	// size of input data
	constexpr static auto in_size = IN;
	// size of output data
	constexpr static auto out_size = 1;

	// construct Neuron
	constexpr Neuron(store_type* data = nullptr) : ndata(data) {}

	// computing result
	constexpr result_type operator()(const feed_type& data) const {
		return proccess(data);
	}

	// computing result
	constexpr result_type proccess(const feed_type& data) const {
		result_type o = 0.f;
		for (std::size_t i = 0; i < IN; ++i) {
			o += data[i] * ndata[i << 1] + ndata[(i << 1) + 1];
		}
		return sigmoid(o);
	}

   private:
	// current neuron data
	store_type* ndata;
};

// class Layer contain Neurons
// Warn: use recurrent deriving
// look like this:
// <1, 2, 3, 4>  => <2, 3, 4> ==> <3, 4>
//      \\              \\
//    <1, 2>          <2, 3>
//
//  \\ - deriving (base class at bottom)
// ==> - deriving (base class at right)
//  => - left class store right;
//
template <std::size_t...>
class Layer;

// end point of recurrent deriving of Layer
template <std::size_t IN, std::size_t OUT>
class Layer<IN, OUT> {
	// type used neurons
	using neuron_type = Neuron<IN>;

   public:
	// type of input data
	using feed_type = typename neuron_type::feed_type;
	// type of output data
	using result_type = array<typename neuron_type::result_type, OUT>;

	// number of store_type what it need
	constexpr static auto data_size = neuron_type::data_size * OUT;
	// size of input data
	constexpr static auto in_size = neuron_type::in_size;
	// size of output data
	constexpr static auto out_size = OUT;

	// construct Layer and inner neurons
	constexpr Layer(store_type* data)
		: neurons(reinterpret_cast<neuron_type*>(operator new[](
			  OUT * sizeof(neuron_type)))) {
		for (std::size_t i = 0; i < OUT; ++i) {
			new (neurons + i) neuron_type(data);
			data += neuron_type::data_size;
		}
	}

	// destruct Layer with inner neurons
	constexpr ~Layer() { operator delete[](neurons); }

	// compute result via proccess()
	constexpr result_type operator()(feed_type& data) const {
		return proccess(data);
	}

	// compute result
	constexpr result_type proccess(feed_type& data) const {
		result_type o{};

		for (std::size_t i = 0; i < OUT; ++i) {
			o[i] = neurons[i](data);
		}

		data.release();

		return o;
	}

   private:
	// neurons
	neuron_type* neurons;
};

// recurrent Layer
template <std::size_t IN, std::size_t OUT, std::size_t... Ss>
class Layer<IN, OUT, Ss...> {
	// type of base class
	using base_type = Layer<IN, OUT>;
	// type of stored class
	using next_layer_type = Layer<OUT, Ss...>;

	// number of store_type what base_type need
	constexpr static auto base_data_size = base_type::data_size;

   public:
	// type of input data
	using feed_type = typename base_type::feed_type;
	// type of output data
	using result_type = typename next_layer_type::result_type;

	// number of store_type what it need
	constexpr static auto data_size =
		base_type::data_size + next_layer_type::data_size;
	// size of input data
	constexpr static auto in_size = base_type::in_size;
	// size of output data
	constexpr static auto out_size = next_layer_type::out_size;

	// construct base and stored Layer
	constexpr Layer(store_type* data)
		: base(data), next_layer(data + base_data_size) {}

	// compute result
	constexpr result_type operator()(feed_type& data) const {
		return proccess(data);
	}

	constexpr result_type proccess(feed_type& data) const {
		auto tmp = base(data);
		data.release();
		return next_layer(tmp);
	}

   private:
	base_type base;
	next_layer_type next_layer;
};
}  // namespace

// class SimpleNet contain first layer (but it contain next layer and etc).
// class SimpleNet contain data for neurons
template <std::size_t... Ss>
requires(sizeof...(Ss) >= 2) class SimpleNet {
   public:
	// type of first layer
	using layer_type = Layer<Ss...>;
	// type of output data
	using result_type = typename layer_type::result_type;
	// type of input data
	using feed_type = typename layer_type::feed_type;

	// number of store_type what all layers are contain
	constexpr static auto data_size = layer_type::data_size;
	// size of input data
	constexpr static auto in_size = layer_type::in_size;
	// size of output data
	constexpr static auto out_size = layer_type::out_size;

	// construct SimpleNet
	// allocate neuron data
	// allocate first layer
	constexpr SimpleNet()
		: data(new store_type[data_size]), layer(new layer_type(data)) {}

	// copy constructor
	constexpr SimpleNet(const SimpleNet& other) : SimpleNet() { *this = other; }

	// deallocate store
	constexpr ~SimpleNet() {
		delete layer;
		delete[] data;
	}

	// compute result
	constexpr result_type operator()(const feed_type& data) const {
		return proccess(data);
	}

	// copy data from other SimpleNet
	constexpr SimpleNet& operator=(const SimpleNet& other) {
		std::memcpy(data, other.data, data_size * sizeof(store_type));

		return *this;
	}

	// check SimpleNets are equal
	constexpr bool operator==(const SimpleNet& other) const {
		return 0 ==
			   std::memcmp(data, other.data, data_size * sizeof(store_type));
	}

	// wrapper for merge()
	SimpleNet operator+(const SimpleNet& other) const { return merge(other); }

	// wrapper for mutation()
	SimpleNet& operator+(int mut) { return mutation(mut); }

	// wrapper for mutation()
	SimpleNet& operator++() { return mutation(); }

	// wrapper for mutation()
	SimpleNet operator++(int z) {
		SimpleNet o{*this};
		mutation(z ? z : 1);
		return o;
	}

	// randomize network
	void rand() {
		std::random_device rd;
		std::uniform_real_distribution<store_type> rand;
		for (std::size_t i = 0; i < data_size; ++i) {
			data[i] = rand(rd);
		}
	}

	// merge networks by random indexes
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

	// mutate stored neurons data at random index @count@ times
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

	// compute result
	result_type proccess(const feed_type& data) const {
		auto data_copy{data};
		return layer->proccess(data_copy);
	}

   private:
	// neurons data
	store_type* data;
	// ptr to first layer
	layer_type* layer;

	// operator for restoring SimpleNet from stream
	template <typename Tchar>
	friend std::basic_istream<Tchar>& operator>>(std::basic_istream<Tchar>& s,
												 SimpleNet<Ss...>& n) {
		return s.read(reinterpret_cast<Tchar*>(n.data),
					  net::SimpleNet<Ss...>::data_size * sizeof(store_type));
	}

	// operator for saving SimpleNet to stream
	template <typename Tchar>
	friend std::basic_ostream<Tchar>& operator<<(std::basic_ostream<Tchar>& s,
												 const SimpleNet<Ss...>& n) {
		return s.write(reinterpret_cast<const Tchar*>(n.data),
					   net::SimpleNet<Ss...>::data_size * sizeof(store_type));
	}
};

// class Net store SimpleNets his score and result
template <std::size_t... Ss>
requires(sizeof...(Ss) >= 2) class Net {
   public:
	// type of used networks
	using net_type = SimpleNet<Ss...>;
	// type used for score of networks
	using score_type = store_type;
	// type of output data
	using result_type = typename net_type::result_type;
	// type of input data
	using feed_type = typename net_type::feed_type;

	// class for default comparing results and score
	// return true if first param closest to 0 then second
	template <typename T>
	struct compare_default {
		constexpr bool operator()(const T& a, const T& b) const {
			return abs(a) < abs(b);
		}
	};

	// struct stored network, his score and result
	struct tuple_type : std::tuple<score_type, net_type, result_type> {
		friend constexpr auto operator<=>(const tuple_type& a,
										  const tuple_type& b) {
			return std::get<score_type>(a) <=> std::get<score_type>(b);
		}
	};

	template <>
	struct compare_default<tuple_type> {
		constexpr bool operator()(const tuple_type& a,
								  const tuple_type& b) const {
			return compare_default<score_type>()(std::get<score_type>(a),
												 std::get<score_type>(b));
		}
	};

	// size of input data
	constexpr static auto in_size = net_type::in_size;
	// size of output data
	constexpr static auto out_size = net_type::out_size;

	// compute required size and allocate nets
	// to_use_ is number of nets used for generating new generation
	// immutable_ is number of nets NOT used for generating new generation
	//                  but not overrided at generating new generation
	constexpr Net(std::size_t to_use_, std::size_t immutable_ = 0)
		: to_use(to_use_),
		  immutable(immutable_),
		  nets_size(to_use_ + immutable_ + (to_use * (to_use >> 1))) {
		if (to_use < 2) {
			throw std::invalid_argument{"net::Net to_use should be >=2"};
		}
		nets.resize(nets_size);
	}

	// check Nets is equal
	constexpr bool operator==(const Net& other) {
		for (std::size_t i = 0; i < nets_size; ++i) {
			auto n1 = std::get<net_type>(nets[i]);
			auto n2 = std::get<net_type>(other.nets[i]);
			if (n1 != n2) {
				return false;
			}
		}
		return true;
	}

	// randomize all nets
	constexpr Net& rand() {
		for (auto& n : nets) {
			auto& nn = std::get<net_type>(n);
			nn.rand();
		}
		return *this;
	}

	// compute result
	constexpr Net& feed(const feed_type& data) {
		/* TODO: add multithreading */
		for (auto& n : nets) {
			auto& nn = std::get<net_type>(n);
			auto& res = std::get<result_type>(n);

			res = nn(data);
		}
		return *this;
	}

	// count score for each network
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

	// generate new generation using mutations and select best score by Compare
	// class
	template <template <typename> typename Compare = compare_default>
	constexpr Net& next(int mutation = 2,
						Compare<tuple_type> comp = Compare<tuple_type>()) {
		std::sort(std::begin(nets), std::end(nets), comp);

		std::size_t child_id = to_use + immutable;

		for (std::size_t i = 0; i < to_use; ++i) {
			for (std::size_t j = i + 1; j < to_use; ++j) {
				auto& child = std::get<net_type>(nets[child_id]);
				auto& parent1 = std::get<net_type>(nets[i]);
				auto& parent2 = std::get<net_type>(nets[j]);

				child = parent1 + parent2 + mutation;

				++child_id;
			}
		}

		return *this;
	}

	// return avg score
	constexpr score_type score() const {
		score_type o{};

		for (auto& n : nets) {
			o += std::get<score_type>(n);
		}

		return o / nets_size;
	}

	// return best score selected by Compare class
	template <template <typename> typename Compare = compare_default>
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

	// reset all scores
	constexpr Net& reset_score() {
		/* TODO: add multithreading */
		for (auto& n : nets) {
			std::get<score_type>(n) = score_type{};
		}
		return *this;
	}

	// return avg result
	constexpr result_type result() const {
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

	// return result of best net
	template <template <typename> typename Compare = compare_default>
	constexpr result_type best_result(Compare<score_type> comp = {}) const {
		result_type res = std::get<result_type>(nets[0]);
		score_type score = std::get<score_type>(nets[0]);
		for (auto& n : nets) {
			score_type c_score = std::get<score_type>(n);
			if (comp(c_score, score)) {
				score = c_score;
				res = std::get<result_type>(n);
			}
		}

		return res;
	}

   private:
	const std::size_t to_use;
	const std::size_t nets_size;
	const std::size_t immutable;

	// networks
	std::vector<tuple_type> nets;
};

}  // namespace net

// vim: set ts=4 sw=4 :
