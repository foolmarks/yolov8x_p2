#ifndef LTTNG_SESSION_UTILS_H
#define LTTNG_SESSION_UTILS_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <string>

#include <glib-2.0/glib.h>
#include <lttng/lttng.h>

namespace utils {
class LttngSession {
	public:
		LttngSession(const char *session_name, const pid_t pid);
		~LttngSession();

		static std::string create_session_url(const char *session_name);

		// Forbid others methods to create an instance
		LttngSession() = delete;

		LttngSession(const LttngSession&) = delete;
		LttngSession& operator=(const LttngSession&) = delete;

		LttngSession(LttngSession&&) = delete;
		LttngSession& operator=(LttngSession&&) = delete;
	private:
		struct lttng_handle * _handle = nullptr;
};
} // namespace utils

#endif // LTTNG_SESSION_UTILS_H