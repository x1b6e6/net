#pragma once

#include <thread>

struct TimeLimit {
	template <typename T>
	TimeLimit(std::chrono::duration<T> limit) {
		auto begin = std::chrono::high_resolution_clock::now();
		th = std::jthread([limit, begin](std::stop_token token) {
			auto now = std::chrono::high_resolution_clock::now();
			std::this_thread::sleep_for(limit - (now - begin));
			if (not token.stop_requested())
				std::abort();
		});
	}
	~TimeLimit() { cancel(); }

	void cancel() { th.request_stop(); }

   private:
	std::jthread th;
};

// vim: set ts=4 sw=4 :
