#include "annmodule.h"
#include <vx_ext_amd.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <inttypes.h>
#include <chrono>
#include <unistd.h>

#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d", status, __LINE__); return -1; } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

static vx_status copyTensor(vx_tensor tensor, std::string fileName, vx_enum usage = VX_WRITE_ONLY)
{
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    vx_size itemsize = sizeof(float);
    if(data_type == VX_TYPE_UINT8 || data_type == VX_TYPE_INT8) {
        itemsize = sizeof(vx_uint8);
    }
    else if(data_type == VX_TYPE_UINT16 || data_type == VX_TYPE_INT16 || data_type == VX_TYPE_FLOAT16) {
        itemsize = sizeof(vx_uint16);
    }
    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_map_id map_id;
    float * ptr;
    vx_status status = vxMapTensorPatch(tensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for " << fileName << std::endl;
        return -1;
    }
    FILE * fp = fopen(fileName.c_str(), usage == VX_WRITE_ONLY ? "rb" : "wb");
    if(!fp) {
        std::cerr << "ERROR: unable to open: " << fileName << std::endl;
        return -1;
    }
    if(usage == VX_WRITE_ONLY) {
        vx_size n = fread(ptr, itemsize, count, fp);
        if(n != count) {
            std::cerr << "ERROR: expected char[" << count*itemsize << "], but got char[" << n*itemsize << "] in " << fileName << std::endl;
            return -1;
        }
    }
    else {
        fwrite(ptr, itemsize, count, fp);
    }
    fclose(fp);
    status = vxUnmapTensorPatch(tensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << fileName << std::endl;
        return -1;
    }
    return 0;
}

int main(int argc , char ** argv)
{
    // get module configuration
    vx_size dimInput[4] = { 0 }, dimOutput[4] = { 0 };
    annGetTensorDimensions(dimInput, dimOutput);
    printf("OK: annGetTensorDimensions() => [input %ldx%ldx%ldx%ld] [output %ldx%ldx%ldx%ld]\n", dimInput[0], dimInput[1], dimInput[2], dimInput[3], dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);

    // create context, input, output, and graph
    vxRegisterLogCallback(NULL, log_callback, vx_false_e);
    vx_context context = vxCreateContext();
    if(vxGetStatus((vx_reference)context)) {
        printf("ERROR: vxCreateContext() failed\n");
        return -1;
    }
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    vx_tensor input = vxCreateTensor(context, 4, dimInput, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)input)) {
        printf("ERROR: vxCreateTensor(input,4,{%ld,%ld,%ld,%ld}) failed\n", dimInput[0], dimInput[1], dimInput[2], dimInput[3]);
        return -1;
    }
    vx_tensor output = vxCreateTensor(context, 4, dimOutput, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)output)) {
        printf("ERROR: vxCreateTensor(output,4,{%ld,%ld,%ld,%ld},VX_TYPE_FLOAT32,0) failed\n", dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);
        return -1;
    }

    // build graph from the module
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();
    vx_graph graph = annCreateGraph(context, input, output, argc > 1 ? argv[1] : nullptr);
    t1 = clockCounter();
    if(vxGetStatus((vx_reference)graph)) {
        printf("ERROR: annCreateGraph(...,%s) failed\n", argv[1]);
        return -1;
    }
    printf("OK: annCreateGraph() took %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    if(argc > 2) {
        if(copyTensor(input, argv[2], VX_WRITE_ONLY) < 0) {
            return -1;
        }
        printf("OK: read %ldx%ldx%ldx%ld tensor from %s\n", dimInput[3], dimInput[2], dimInput[1], dimInput[0], argv[2]);
    }

    t0 = clockCounter();
    vx_status status = vxProcessGraph(graph);
    t1 = clockCounter();
    if(status != VX_SUCCESS) {
        printf("ERROR: vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    printf("OK: vxProcessGraph() took %.3f msec (1st iteration)\n", (float)(t1-t0)*1000.0f/(float)freq);

    if(argc > 3) {
        if(copyTensor(output, argv[3], VX_READ_ONLY) < 0) {
            return -1;
        }
        printf("OK: wrote %ldx%ldx%ldx%ld tensor into %s\n", dimOutput[3], dimOutput[2], dimOutput[1], dimOutput[0], argv[3]);
    }
    t0 = clockCounter();
    int N = 100;
    for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph);
        if(status != VX_SUCCESS)
            break;
    }
    t1 = clockCounter();
    printf("OK: vxProcessGraph() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);

    // release resources
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseTensor(&input));
    ERROR_CHECK_STATUS(vxReleaseTensor(&output));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    printf("OK: successful\n");

    return 0;
}
