#pragma once

#include "Net.hh"

namespace net {
template <std::size_t... Ss>
class Trainer {
	using net_type = Net<Ss...>;
	using result_type = typename net_type::result_type;

   public:
	Trainer(std::size_t to_use_, float target_error_)
		: to_use(to_use_),
		  nets_size(to_use * to_use),
		  target_error(target_error_) {
		for (auto& n : nets) {
			std::get<net_type>(n).rand();
		}
	}

	net_type getBest() const { return std::get<net_type>(nets.at(0)); }

   private:
	std::vector<std::tuple<float, net_type, result_type>> nets;

	const float target_error;
	const std::size_t to_use;
	const std::size_t nets_size;
};
}  // namespace net

// vim: set ts=4 sw=4 :
