#ifndef annmodule_h
#define annmodule_h

#include <VX/vx.h>

extern "C" {
    VX_API_ENTRY void     VX_API_CALL annGetTensorDimensions(vx_size dimInput[4], vx_size dimOutput[4]);
    VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor input,  vx_tensor output, const char * options);
};

#endif
