#ifndef _FLAGS_H
#define _FLAGS_H

namespace hph {

enum Flags {
	FLOAT = 1 << 2,
	TBB = 1 << 3,
	OPENCL = 1 << 4,
    SSE = 1 << 5,
    AVX = 1 << 6,
	AVX512 = 1 << 7
};

} // namespace mds

#endif // _FLAGS_H
