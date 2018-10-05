#include "annmodule.h"

#include <vx_ext_amd.h>
#include <VX/vx_khr_nn.h>
#include <vx_amd_nn.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return nullptr; } }
#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return nullptr; } }

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

VX_API_ENTRY void VX_API_CALL annGetTensorDimensions(vx_size dimInput[4], vx_size dimOutput[4])
{
    dimInput[0] = 299;
    dimInput[1] = 299;
    dimInput[2] = 3;
    dimInput[3] = 1;
    dimOutput[0] = 1;
    dimOutput[1] = 1;
    dimOutput[2] = 1000;
    dimOutput[3] = 1;
}

VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor data, vx_tensor prob, const char * dataFolder_)
{
    // load neural network extension kernels
    ERROR_CHECK_STATUS(vxLoadKernels(context,"vx_nn"));

    // create graph
    vx_graph graph = vxCreateGraph(context); 
    ERROR_CHECK_OBJECT(graph);

    // get dataFolder option
    std::string dataFolder = dataFolder_ ? dataFolder_ : ".", fileName;

    ////
    // initialize the graph
    // conv1_3x3_s2 Layer
    vx_size conv1_3x3_s2_dims[4] = { 149, 149, 32, 1 };
    vx_tensor conv1_3x3_s2;
    conv1_3x3_s2 = vxCreateVirtualTensor(graph,4, conv1_3x3_s2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2);
    vx_size conv1_3x3_s2_W_dims[4] = { 3, 3, 3, 32 };
    vx_tensor conv1_3x3_s2_W;
    conv1_3x3_s2_W = vxCreateTensor(context,4, conv1_3x3_s2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_W, dataFolder + "/weights/conv1_3x3_s2.f32"));
    vx_nn_convolution_params_t conv1_3x3_s2_params;
    conv1_3x3_s2_params.padding_x = 0;
    conv1_3x3_s2_params.padding_y = 0;
    conv1_3x3_s2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv1_3x3_s2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv1_3x3_s2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv1_3x3_s2_params.dilation_x = 0;
    conv1_3x3_s2_params.dilation_y = 0;
    vx_node conv1_3x3_s2_node;
    conv1_3x3_s2_node = vxConvolutionLayer(graph, data, conv1_3x3_s2_W, NULL, &conv1_3x3_s2_params, sizeof(conv1_3x3_s2_params ), conv1_3x3_s2);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_3x3_s2_node));

    // conv1_3x3_s2_bn Layer
    vx_size conv1_3x3_s2_scale_dims[4] = { 149, 149, 32, 1 };
    vx_tensor conv1_3x3_s2_scale;
    conv1_3x3_s2_scale = vxCreateVirtualTensor(graph,4, conv1_3x3_s2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_scale);
    vx_size conv1_3x3_s2_bn_W_dims[1] = { 32 };
    vx_float32 conv1_3x3_s2_bn_eps = 0.001;
    vx_tensor conv1_3x3_s2_bn_W;
    conv1_3x3_s2_bn_W = vxCreateTensor(context,1, conv1_3x3_s2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_bn_W, dataFolder + "/weights/conv1_3x3_s2_bn.f32"));
    vx_size conv1_3x3_s2_bn_B_dims[1] = { 32 };
    vx_tensor conv1_3x3_s2_bn_B;
    conv1_3x3_s2_bn_B = vxCreateTensor(context,1, conv1_3x3_s2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_bn_B, dataFolder + "/bias/conv1_3x3_s2_bn.f32"));
    vx_size conv1_3x3_s2_scale_W_dims[1] = { 32 };
    vx_tensor conv1_3x3_s2_scale_W;
    conv1_3x3_s2_scale_W = vxCreateTensor(context,1, conv1_3x3_s2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_scale_W, dataFolder + "/weights/conv1_3x3_s2_scale.f32"));
    vx_size conv1_3x3_s2_scale_B_dims[1] = { 32 };
    vx_tensor conv1_3x3_s2_scale_B;
    conv1_3x3_s2_scale_B = vxCreateTensor(context,1, conv1_3x3_s2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_scale_B, dataFolder + "/bias/conv1_3x3_s2_scale.f32"));
    vx_node conv1_3x3_s2_bn_node;
    conv1_3x3_s2_bn_node = vxBatchNormalizationLayer(graph, conv1_3x3_s2, conv1_3x3_s2_bn_W, conv1_3x3_s2_bn_B, conv1_3x3_s2_scale_W, conv1_3x3_s2_scale_B, conv1_3x3_s2_bn_eps, conv1_3x3_s2_scale);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_3x3_s2_bn_node));

    // conv1_3x3_s2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv1_3x3_relu Layer
    vx_size conv1_3x3_relu_dims[4] = { 149, 149, 32, 1 };
    vx_tensor conv1_3x3_relu;
    conv1_3x3_relu = vxCreateVirtualTensor(graph,4, conv1_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_3x3_relu);
    vx_enum conv1_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv1_3x3_relu_param_a = 0;
    vx_float32 conv1_3x3_relu_param_b = 0;
    vx_node conv1_3x3_relu_node;
    conv1_3x3_relu_node = vxActivationLayer(graph, conv1_3x3_s2_scale, conv1_3x3_relu_mode, conv1_3x3_relu_param_a, conv1_3x3_relu_param_b, conv1_3x3_relu);
    ERROR_CHECK_OBJECT(conv1_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_3x3_relu_node));

    // conv2_3x3_s1 Layer
    vx_size conv2_3x3_s1_dims[4] = { 147, 147, 32, 1 };
    vx_tensor conv2_3x3_s1;
    conv2_3x3_s1 = vxCreateVirtualTensor(graph,4, conv2_3x3_s1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1);
    vx_size conv2_3x3_s1_W_dims[4] = { 3, 3, 32, 32 };
    vx_tensor conv2_3x3_s1_W;
    conv2_3x3_s1_W = vxCreateTensor(context,4, conv2_3x3_s1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_W, dataFolder + "/weights/conv2_3x3_s1.f32"));
    vx_nn_convolution_params_t conv2_3x3_s1_params;
    conv2_3x3_s1_params.padding_x = 0;
    conv2_3x3_s1_params.padding_y = 0;
    conv2_3x3_s1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv2_3x3_s1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv2_3x3_s1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv2_3x3_s1_params.dilation_x = 0;
    conv2_3x3_s1_params.dilation_y = 0;
    vx_node conv2_3x3_s1_node;
    conv2_3x3_s1_node = vxConvolutionLayer(graph, conv1_3x3_relu, conv2_3x3_s1_W, NULL, &conv2_3x3_s1_params, sizeof(conv2_3x3_s1_params ), conv2_3x3_s1);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_3x3_s1_node));

    // conv2_3x3_s1_bn Layer
    vx_size conv2_3x3_s1_scale_dims[4] = { 147, 147, 32, 1 };
    vx_tensor conv2_3x3_s1_scale;
    conv2_3x3_s1_scale = vxCreateVirtualTensor(graph,4, conv2_3x3_s1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_scale);
    vx_size conv2_3x3_s1_bn_W_dims[1] = { 32 };
    vx_float32 conv2_3x3_s1_bn_eps = 0.001;
    vx_tensor conv2_3x3_s1_bn_W;
    conv2_3x3_s1_bn_W = vxCreateTensor(context,1, conv2_3x3_s1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_bn_W, dataFolder + "/weights/conv2_3x3_s1_bn.f32"));
    vx_size conv2_3x3_s1_bn_B_dims[1] = { 32 };
    vx_tensor conv2_3x3_s1_bn_B;
    conv2_3x3_s1_bn_B = vxCreateTensor(context,1, conv2_3x3_s1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_bn_B, dataFolder + "/bias/conv2_3x3_s1_bn.f32"));
    vx_size conv2_3x3_s1_scale_W_dims[1] = { 32 };
    vx_tensor conv2_3x3_s1_scale_W;
    conv2_3x3_s1_scale_W = vxCreateTensor(context,1, conv2_3x3_s1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_scale_W, dataFolder + "/weights/conv2_3x3_s1_scale.f32"));
    vx_size conv2_3x3_s1_scale_B_dims[1] = { 32 };
    vx_tensor conv2_3x3_s1_scale_B;
    conv2_3x3_s1_scale_B = vxCreateTensor(context,1, conv2_3x3_s1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_scale_B, dataFolder + "/bias/conv2_3x3_s1_scale.f32"));
    vx_node conv2_3x3_s1_bn_node;
    conv2_3x3_s1_bn_node = vxBatchNormalizationLayer(graph, conv2_3x3_s1, conv2_3x3_s1_bn_W, conv2_3x3_s1_bn_B, conv2_3x3_s1_scale_W, conv2_3x3_s1_scale_B, conv2_3x3_s1_bn_eps, conv2_3x3_s1_scale);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_3x3_s1_bn_node));

    // conv2_3x3_s1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv2_3x3_relu Layer
    vx_size conv2_3x3_relu_dims[4] = { 147, 147, 32, 1 };
    vx_tensor conv2_3x3_relu;
    conv2_3x3_relu = vxCreateVirtualTensor(graph,4, conv2_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_3x3_relu);
    vx_enum conv2_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv2_3x3_relu_param_a = 0;
    vx_float32 conv2_3x3_relu_param_b = 0;
    vx_node conv2_3x3_relu_node;
    conv2_3x3_relu_node = vxActivationLayer(graph, conv2_3x3_s1_scale, conv2_3x3_relu_mode, conv2_3x3_relu_param_a, conv2_3x3_relu_param_b, conv2_3x3_relu);
    ERROR_CHECK_OBJECT(conv2_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_3x3_relu_node));

    // conv3_3x3_s1 Layer
    vx_size conv3_3x3_s1_dims[4] = { 147, 147, 64, 1 };
    vx_tensor conv3_3x3_s1;
    conv3_3x3_s1 = vxCreateVirtualTensor(graph,4, conv3_3x3_s1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1);
    vx_size conv3_3x3_s1_W_dims[4] = { 3, 3, 32, 64 };
    vx_tensor conv3_3x3_s1_W;
    conv3_3x3_s1_W = vxCreateTensor(context,4, conv3_3x3_s1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_W, dataFolder + "/weights/conv3_3x3_s1.f32"));
    vx_nn_convolution_params_t conv3_3x3_s1_params;
    conv3_3x3_s1_params.padding_x = 1;
    conv3_3x3_s1_params.padding_y = 1;
    conv3_3x3_s1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv3_3x3_s1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv3_3x3_s1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv3_3x3_s1_params.dilation_x = 0;
    conv3_3x3_s1_params.dilation_y = 0;
    vx_node conv3_3x3_s1_node;
    conv3_3x3_s1_node = vxConvolutionLayer(graph, conv2_3x3_relu, conv3_3x3_s1_W, NULL, &conv3_3x3_s1_params, sizeof(conv3_3x3_s1_params ), conv3_3x3_s1);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_3x3_s1_node));

    // conv3_3x3_s1_bn Layer
    vx_size conv3_3x3_s1_scale_dims[4] = { 147, 147, 64, 1 };
    vx_tensor conv3_3x3_s1_scale;
    conv3_3x3_s1_scale = vxCreateVirtualTensor(graph,4, conv3_3x3_s1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_scale);
    vx_size conv3_3x3_s1_bn_W_dims[1] = { 64 };
    vx_float32 conv3_3x3_s1_bn_eps = 0.001;
    vx_tensor conv3_3x3_s1_bn_W;
    conv3_3x3_s1_bn_W = vxCreateTensor(context,1, conv3_3x3_s1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_bn_W, dataFolder + "/weights/conv3_3x3_s1_bn.f32"));
    vx_size conv3_3x3_s1_bn_B_dims[1] = { 64 };
    vx_tensor conv3_3x3_s1_bn_B;
    conv3_3x3_s1_bn_B = vxCreateTensor(context,1, conv3_3x3_s1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_bn_B, dataFolder + "/bias/conv3_3x3_s1_bn.f32"));
    vx_size conv3_3x3_s1_scale_W_dims[1] = { 64 };
    vx_tensor conv3_3x3_s1_scale_W;
    conv3_3x3_s1_scale_W = vxCreateTensor(context,1, conv3_3x3_s1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_scale_W, dataFolder + "/weights/conv3_3x3_s1_scale.f32"));
    vx_size conv3_3x3_s1_scale_B_dims[1] = { 64 };
    vx_tensor conv3_3x3_s1_scale_B;
    conv3_3x3_s1_scale_B = vxCreateTensor(context,1, conv3_3x3_s1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_scale_B, dataFolder + "/bias/conv3_3x3_s1_scale.f32"));
    vx_node conv3_3x3_s1_bn_node;
    conv3_3x3_s1_bn_node = vxBatchNormalizationLayer(graph, conv3_3x3_s1, conv3_3x3_s1_bn_W, conv3_3x3_s1_bn_B, conv3_3x3_s1_scale_W, conv3_3x3_s1_scale_B, conv3_3x3_s1_bn_eps, conv3_3x3_s1_scale);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_3x3_s1_bn_node));

    // conv3_3x3_s1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv3_3x3_relu Layer
    vx_size conv3_3x3_relu_dims[4] = { 147, 147, 64, 1 };
    vx_tensor conv3_3x3_relu;
    conv3_3x3_relu = vxCreateVirtualTensor(graph,4, conv3_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_3x3_relu);
    vx_enum conv3_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv3_3x3_relu_param_a = 0;
    vx_float32 conv3_3x3_relu_param_b = 0;
    vx_node conv3_3x3_relu_node;
    conv3_3x3_relu_node = vxActivationLayer(graph, conv3_3x3_s1_scale, conv3_3x3_relu_mode, conv3_3x3_relu_param_a, conv3_3x3_relu_param_b, conv3_3x3_relu);
    ERROR_CHECK_OBJECT(conv3_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_3x3_relu_node));

    // pool1_3x3_s2 Layer
    vx_size pool1_3x3_s2_dims[4] = { 73, 73, 64, 1 };
    vx_tensor pool1_3x3_s2;
    pool1_3x3_s2 = vxCreateVirtualTensor(graph,4, pool1_3x3_s2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool1_3x3_s2);
    vx_enum pool1_3x3_s2_type = VX_NN_POOLING_MAX;
    vx_size pool1_3x3_s2_kernel_w = 3;
    vx_size pool1_3x3_s2_kernel_h = 3;
    vx_size pool1_3x3_s2_pad_w = 0;
    vx_size pool1_3x3_s2_pad_h = 0;
    vx_enum pool1_3x3_s2_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool1_3x3_s2_node;
    pool1_3x3_s2_node = vxPoolingLayer(graph, conv3_3x3_relu, pool1_3x3_s2_type, pool1_3x3_s2_kernel_w, pool1_3x3_s2_kernel_h, pool1_3x3_s2_pad_w, pool1_3x3_s2_pad_h, pool1_3x3_s2_roundPolicy, pool1_3x3_s2 );
    ERROR_CHECK_OBJECT(pool1_3x3_s2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool1_3x3_s2_node));

    // conv4_3x3_reduce Layer
    vx_size conv4_3x3_reduce_dims[4] = { 73, 73, 80, 1 };
    vx_tensor conv4_3x3_reduce;
    conv4_3x3_reduce = vxCreateVirtualTensor(graph,4, conv4_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce);
    vx_size conv4_3x3_reduce_W_dims[4] = { 1, 1, 64, 80 };
    vx_tensor conv4_3x3_reduce_W;
    conv4_3x3_reduce_W = vxCreateTensor(context,4, conv4_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_reduce_W, dataFolder + "/weights/conv4_3x3_reduce.f32"));
    vx_nn_convolution_params_t conv4_3x3_reduce_params;
    conv4_3x3_reduce_params.padding_x = 0;
    conv4_3x3_reduce_params.padding_y = 0;
    conv4_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv4_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv4_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv4_3x3_reduce_params.dilation_x = 0;
    conv4_3x3_reduce_params.dilation_y = 0;
    vx_node conv4_3x3_reduce_node;
    conv4_3x3_reduce_node = vxConvolutionLayer(graph, pool1_3x3_s2, conv4_3x3_reduce_W, NULL, &conv4_3x3_reduce_params, sizeof(conv4_3x3_reduce_params ), conv4_3x3_reduce);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_3x3_reduce_node));

    // conv4_3x3_reduce_bn Layer
    vx_size conv4_3x3_reduce_scale_dims[4] = { 73, 73, 80, 1 };
    vx_tensor conv4_3x3_reduce_scale;
    conv4_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, conv4_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_scale);
    vx_size conv4_3x3_reduce_bn_W_dims[1] = { 80 };
    vx_float32 conv4_3x3_reduce_bn_eps = 0.001;
    vx_tensor conv4_3x3_reduce_bn_W;
    conv4_3x3_reduce_bn_W = vxCreateTensor(context,1, conv4_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_reduce_bn_W, dataFolder + "/weights/conv4_3x3_reduce_bn.f32"));
    vx_size conv4_3x3_reduce_bn_B_dims[1] = { 80 };
    vx_tensor conv4_3x3_reduce_bn_B;
    conv4_3x3_reduce_bn_B = vxCreateTensor(context,1, conv4_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_reduce_bn_B, dataFolder + "/bias/conv4_3x3_reduce_bn.f32"));
    vx_size conv4_3x3_reduce_scale_W_dims[1] = { 80 };
    vx_tensor conv4_3x3_reduce_scale_W;
    conv4_3x3_reduce_scale_W = vxCreateTensor(context,1, conv4_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_reduce_scale_W, dataFolder + "/weights/conv4_3x3_reduce_scale.f32"));
    vx_size conv4_3x3_reduce_scale_B_dims[1] = { 80 };
    vx_tensor conv4_3x3_reduce_scale_B;
    conv4_3x3_reduce_scale_B = vxCreateTensor(context,1, conv4_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_reduce_scale_B, dataFolder + "/bias/conv4_3x3_reduce_scale.f32"));
    vx_node conv4_3x3_reduce_bn_node;
    conv4_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, conv4_3x3_reduce, conv4_3x3_reduce_bn_W, conv4_3x3_reduce_bn_B, conv4_3x3_reduce_scale_W, conv4_3x3_reduce_scale_B, conv4_3x3_reduce_bn_eps, conv4_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(conv4_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_3x3_reduce_bn_node));

    // conv4_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv4_relu_3x3_reduce Layer
    vx_size conv4_relu_3x3_reduce_dims[4] = { 73, 73, 80, 1 };
    vx_tensor conv4_relu_3x3_reduce;
    conv4_relu_3x3_reduce = vxCreateVirtualTensor(graph,4, conv4_relu_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_relu_3x3_reduce);
    vx_enum conv4_relu_3x3_reduce_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv4_relu_3x3_reduce_param_a = 0;
    vx_float32 conv4_relu_3x3_reduce_param_b = 0;
    vx_node conv4_relu_3x3_reduce_node;
    conv4_relu_3x3_reduce_node = vxActivationLayer(graph, conv4_3x3_reduce_scale, conv4_relu_3x3_reduce_mode, conv4_relu_3x3_reduce_param_a, conv4_relu_3x3_reduce_param_b, conv4_relu_3x3_reduce);
    ERROR_CHECK_OBJECT(conv4_relu_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_relu_3x3_reduce_node));

    // conv4_3x3 Layer
    vx_size conv4_3x3_dims[4] = { 71, 71, 192, 1 };
    vx_tensor conv4_3x3;
    conv4_3x3 = vxCreateVirtualTensor(graph,4, conv4_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_3x3);
    vx_size conv4_3x3_W_dims[4] = { 3, 3, 80, 192 };
    vx_tensor conv4_3x3_W;
    conv4_3x3_W = vxCreateTensor(context,4, conv4_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_W, dataFolder + "/weights/conv4_3x3.f32"));
    vx_nn_convolution_params_t conv4_3x3_params;
    conv4_3x3_params.padding_x = 0;
    conv4_3x3_params.padding_y = 0;
    conv4_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv4_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv4_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv4_3x3_params.dilation_x = 0;
    conv4_3x3_params.dilation_y = 0;
    vx_node conv4_3x3_node;
    conv4_3x3_node = vxConvolutionLayer(graph, conv4_relu_3x3_reduce, conv4_3x3_W, NULL, &conv4_3x3_params, sizeof(conv4_3x3_params ), conv4_3x3);
    ERROR_CHECK_OBJECT(conv4_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_3x3_node));

    // conv4_3x3_bn Layer
    vx_size conv4_3x3_scale_dims[4] = { 71, 71, 192, 1 };
    vx_tensor conv4_3x3_scale;
    conv4_3x3_scale = vxCreateVirtualTensor(graph,4, conv4_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_3x3_scale);
    vx_size conv4_3x3_bn_W_dims[1] = { 192 };
    vx_float32 conv4_3x3_bn_eps = 0.001;
    vx_tensor conv4_3x3_bn_W;
    conv4_3x3_bn_W = vxCreateTensor(context,1, conv4_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_bn_W, dataFolder + "/weights/conv4_3x3_bn.f32"));
    vx_size conv4_3x3_bn_B_dims[1] = { 192 };
    vx_tensor conv4_3x3_bn_B;
    conv4_3x3_bn_B = vxCreateTensor(context,1, conv4_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_bn_B, dataFolder + "/bias/conv4_3x3_bn.f32"));
    vx_size conv4_3x3_scale_W_dims[1] = { 192 };
    vx_tensor conv4_3x3_scale_W;
    conv4_3x3_scale_W = vxCreateTensor(context,1, conv4_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_scale_W, dataFolder + "/weights/conv4_3x3_scale.f32"));
    vx_size conv4_3x3_scale_B_dims[1] = { 192 };
    vx_tensor conv4_3x3_scale_B;
    conv4_3x3_scale_B = vxCreateTensor(context,1, conv4_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv4_3x3_scale_B, dataFolder + "/bias/conv4_3x3_scale.f32"));
    vx_node conv4_3x3_bn_node;
    conv4_3x3_bn_node = vxBatchNormalizationLayer(graph, conv4_3x3, conv4_3x3_bn_W, conv4_3x3_bn_B, conv4_3x3_scale_W, conv4_3x3_scale_B, conv4_3x3_bn_eps, conv4_3x3_scale);
    ERROR_CHECK_OBJECT(conv4_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_3x3_bn_node));

    // conv4_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv4_relu_3x3 Layer
    vx_size conv4_relu_3x3_dims[4] = { 71, 71, 192, 1 };
    vx_tensor conv4_relu_3x3;
    conv4_relu_3x3 = vxCreateVirtualTensor(graph,4, conv4_relu_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_relu_3x3);
    vx_enum conv4_relu_3x3_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv4_relu_3x3_param_a = 0;
    vx_float32 conv4_relu_3x3_param_b = 0;
    vx_node conv4_relu_3x3_node;
    conv4_relu_3x3_node = vxActivationLayer(graph, conv4_3x3_scale, conv4_relu_3x3_mode, conv4_relu_3x3_param_a, conv4_relu_3x3_param_b, conv4_relu_3x3);
    ERROR_CHECK_OBJECT(conv4_relu_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_relu_3x3_node));

    // pool2_3x3_s2 Layer
    vx_size pool2_3x3_s2_dims[4] = { 35, 35, 192, 1 };
    vx_tensor pool2_3x3_s2;
    pool2_3x3_s2 = vxCreateVirtualTensor(graph,4, pool2_3x3_s2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool2_3x3_s2);
    vx_enum pool2_3x3_s2_type = VX_NN_POOLING_MAX;
    vx_size pool2_3x3_s2_kernel_w = 3;
    vx_size pool2_3x3_s2_kernel_h = 3;
    vx_size pool2_3x3_s2_pad_w = 0;
    vx_size pool2_3x3_s2_pad_h = 0;
    vx_enum pool2_3x3_s2_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool2_3x3_s2_node;
    pool2_3x3_s2_node = vxPoolingLayer(graph, conv4_relu_3x3, pool2_3x3_s2_type, pool2_3x3_s2_kernel_w, pool2_3x3_s2_kernel_h, pool2_3x3_s2_pad_w, pool2_3x3_s2_pad_h, pool2_3x3_s2_roundPolicy, pool2_3x3_s2 );
    ERROR_CHECK_OBJECT(pool2_3x3_s2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool2_3x3_s2_node));

    // pool2_3x3_s2_pool2_3x3_s2_0_split_0 Layer
    vx_size pool2_3x3_s2_pool2_3x3_s2_0_split_0_dims[4] = { 35, 35, 192, 1 };
    vx_tensor pool2_3x3_s2_pool2_3x3_s2_0_split_0;
    pool2_3x3_s2_pool2_3x3_s2_0_split_0 = vxCreateVirtualTensor(graph,4, pool2_3x3_s2_pool2_3x3_s2_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_0);
    vx_node pool2_3x3_s2_pool2_3x3_s2_0_split_0_node;
    pool2_3x3_s2_pool2_3x3_s2_0_split_0_node = vxCopyNode( graph, (vx_reference)pool2_3x3_s2, (vx_reference)pool2_3x3_s2_pool2_3x3_s2_0_split_0);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool2_3x3_s2_pool2_3x3_s2_0_split_0_node));

    // pool2_3x3_s2_pool2_3x3_s2_0_split_1 Layer
    vx_size pool2_3x3_s2_pool2_3x3_s2_0_split_1_dims[4] = { 35, 35, 192, 1 };
    vx_tensor pool2_3x3_s2_pool2_3x3_s2_0_split_1;
    pool2_3x3_s2_pool2_3x3_s2_0_split_1 = vxCreateVirtualTensor(graph,4, pool2_3x3_s2_pool2_3x3_s2_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_1);
    vx_node pool2_3x3_s2_pool2_3x3_s2_0_split_1_node;
    pool2_3x3_s2_pool2_3x3_s2_0_split_1_node = vxCopyNode( graph, (vx_reference)pool2_3x3_s2, (vx_reference)pool2_3x3_s2_pool2_3x3_s2_0_split_1);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool2_3x3_s2_pool2_3x3_s2_0_split_1_node));

    // pool2_3x3_s2_pool2_3x3_s2_0_split_2 Layer
    vx_size pool2_3x3_s2_pool2_3x3_s2_0_split_2_dims[4] = { 35, 35, 192, 1 };
    vx_tensor pool2_3x3_s2_pool2_3x3_s2_0_split_2;
    pool2_3x3_s2_pool2_3x3_s2_0_split_2 = vxCreateVirtualTensor(graph,4, pool2_3x3_s2_pool2_3x3_s2_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_2);
    vx_node pool2_3x3_s2_pool2_3x3_s2_0_split_2_node;
    pool2_3x3_s2_pool2_3x3_s2_0_split_2_node = vxCopyNode( graph, (vx_reference)pool2_3x3_s2, (vx_reference)pool2_3x3_s2_pool2_3x3_s2_0_split_2);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool2_3x3_s2_pool2_3x3_s2_0_split_2_node));

    // pool2_3x3_s2_pool2_3x3_s2_0_split_3 Layer
    vx_size pool2_3x3_s2_pool2_3x3_s2_0_split_3_dims[4] = { 35, 35, 192, 1 };
    vx_tensor pool2_3x3_s2_pool2_3x3_s2_0_split_3;
    pool2_3x3_s2_pool2_3x3_s2_0_split_3 = vxCreateVirtualTensor(graph,4, pool2_3x3_s2_pool2_3x3_s2_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_3);
    vx_node pool2_3x3_s2_pool2_3x3_s2_0_split_3_node;
    pool2_3x3_s2_pool2_3x3_s2_0_split_3_node = vxCopyNode( graph, (vx_reference)pool2_3x3_s2, (vx_reference)pool2_3x3_s2_pool2_3x3_s2_0_split_3);
    ERROR_CHECK_OBJECT(pool2_3x3_s2_pool2_3x3_s2_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool2_3x3_s2_pool2_3x3_s2_0_split_3_node));

    // inception_a1_1x1 Layer
    vx_size inception_a1_1x1_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_1x1;
    inception_a1_1x1 = vxCreateVirtualTensor(graph,4, inception_a1_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1);
    vx_size inception_a1_1x1_W_dims[4] = { 1, 1, 192, 64 };
    vx_tensor inception_a1_1x1_W;
    inception_a1_1x1_W = vxCreateTensor(context,4, inception_a1_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_W, dataFolder + "/weights/inception_a1_1x1.f32"));
    vx_nn_convolution_params_t inception_a1_1x1_params;
    inception_a1_1x1_params.padding_x = 0;
    inception_a1_1x1_params.padding_y = 0;
    inception_a1_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_1x1_params.dilation_x = 0;
    inception_a1_1x1_params.dilation_y = 0;
    vx_node inception_a1_1x1_node;
    inception_a1_1x1_node = vxConvolutionLayer(graph, pool2_3x3_s2_pool2_3x3_s2_0_split_0, inception_a1_1x1_W, NULL, &inception_a1_1x1_params, sizeof(inception_a1_1x1_params ), inception_a1_1x1);
    ERROR_CHECK_OBJECT(inception_a1_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_node));

    // inception_a1_1x1_bn Layer
    vx_size inception_a1_1x1_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_1x1_scale;
    inception_a1_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a1_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_scale);
    vx_size inception_a1_1x1_bn_W_dims[1] = { 64 };
    vx_float32 inception_a1_1x1_bn_eps = 0.001;
    vx_tensor inception_a1_1x1_bn_W;
    inception_a1_1x1_bn_W = vxCreateTensor(context,1, inception_a1_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_bn_W, dataFolder + "/weights/inception_a1_1x1_bn.f32"));
    vx_size inception_a1_1x1_bn_B_dims[1] = { 64 };
    vx_tensor inception_a1_1x1_bn_B;
    inception_a1_1x1_bn_B = vxCreateTensor(context,1, inception_a1_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_bn_B, dataFolder + "/bias/inception_a1_1x1_bn.f32"));
    vx_size inception_a1_1x1_scale_W_dims[1] = { 64 };
    vx_tensor inception_a1_1x1_scale_W;
    inception_a1_1x1_scale_W = vxCreateTensor(context,1, inception_a1_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_scale_W, dataFolder + "/weights/inception_a1_1x1_scale.f32"));
    vx_size inception_a1_1x1_scale_B_dims[1] = { 64 };
    vx_tensor inception_a1_1x1_scale_B;
    inception_a1_1x1_scale_B = vxCreateTensor(context,1, inception_a1_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_scale_B, dataFolder + "/bias/inception_a1_1x1_scale.f32"));
    vx_node inception_a1_1x1_bn_node;
    inception_a1_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a1_1x1, inception_a1_1x1_bn_W, inception_a1_1x1_bn_B, inception_a1_1x1_scale_W, inception_a1_1x1_scale_B, inception_a1_1x1_bn_eps, inception_a1_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a1_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_bn_node));

    // inception_a1_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_1x1_relu Layer
    vx_size inception_a1_1x1_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_1x1_relu;
    inception_a1_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a1_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_relu);
    vx_enum inception_a1_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_1x1_relu_param_a = 0;
    vx_float32 inception_a1_1x1_relu_param_b = 0;
    vx_node inception_a1_1x1_relu_node;
    inception_a1_1x1_relu_node = vxActivationLayer(graph, inception_a1_1x1_scale, inception_a1_1x1_relu_mode, inception_a1_1x1_relu_param_a, inception_a1_1x1_relu_param_b, inception_a1_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a1_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_relu_node));

    // inception_a1_5x5_reduce Layer
    vx_size inception_a1_5x5_reduce_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a1_5x5_reduce;
    inception_a1_5x5_reduce = vxCreateVirtualTensor(graph,4, inception_a1_5x5_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce);
    vx_size inception_a1_5x5_reduce_W_dims[4] = { 1, 1, 192, 48 };
    vx_tensor inception_a1_5x5_reduce_W;
    inception_a1_5x5_reduce_W = vxCreateTensor(context,4, inception_a1_5x5_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_reduce_W, dataFolder + "/weights/inception_a1_5x5_reduce.f32"));
    vx_nn_convolution_params_t inception_a1_5x5_reduce_params;
    inception_a1_5x5_reduce_params.padding_x = 0;
    inception_a1_5x5_reduce_params.padding_y = 0;
    inception_a1_5x5_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_5x5_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_5x5_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_5x5_reduce_params.dilation_x = 0;
    inception_a1_5x5_reduce_params.dilation_y = 0;
    vx_node inception_a1_5x5_reduce_node;
    inception_a1_5x5_reduce_node = vxConvolutionLayer(graph, pool2_3x3_s2_pool2_3x3_s2_0_split_1, inception_a1_5x5_reduce_W, NULL, &inception_a1_5x5_reduce_params, sizeof(inception_a1_5x5_reduce_params ), inception_a1_5x5_reduce);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_5x5_reduce_node));

    // inception_a1_5x5_reduce_bn Layer
    vx_size inception_a1_5x5_reduce_scale_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a1_5x5_reduce_scale;
    inception_a1_5x5_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a1_5x5_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_scale);
    vx_size inception_a1_5x5_reduce_bn_W_dims[1] = { 48 };
    vx_float32 inception_a1_5x5_reduce_bn_eps = 0.001;
    vx_tensor inception_a1_5x5_reduce_bn_W;
    inception_a1_5x5_reduce_bn_W = vxCreateTensor(context,1, inception_a1_5x5_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_reduce_bn_W, dataFolder + "/weights/inception_a1_5x5_reduce_bn.f32"));
    vx_size inception_a1_5x5_reduce_bn_B_dims[1] = { 48 };
    vx_tensor inception_a1_5x5_reduce_bn_B;
    inception_a1_5x5_reduce_bn_B = vxCreateTensor(context,1, inception_a1_5x5_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_reduce_bn_B, dataFolder + "/bias/inception_a1_5x5_reduce_bn.f32"));
    vx_size inception_a1_5x5_reduce_scale_W_dims[1] = { 48 };
    vx_tensor inception_a1_5x5_reduce_scale_W;
    inception_a1_5x5_reduce_scale_W = vxCreateTensor(context,1, inception_a1_5x5_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_reduce_scale_W, dataFolder + "/weights/inception_a1_5x5_reduce_scale.f32"));
    vx_size inception_a1_5x5_reduce_scale_B_dims[1] = { 48 };
    vx_tensor inception_a1_5x5_reduce_scale_B;
    inception_a1_5x5_reduce_scale_B = vxCreateTensor(context,1, inception_a1_5x5_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_reduce_scale_B, dataFolder + "/bias/inception_a1_5x5_reduce_scale.f32"));
    vx_node inception_a1_5x5_reduce_bn_node;
    inception_a1_5x5_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a1_5x5_reduce, inception_a1_5x5_reduce_bn_W, inception_a1_5x5_reduce_bn_B, inception_a1_5x5_reduce_scale_W, inception_a1_5x5_reduce_scale_B, inception_a1_5x5_reduce_bn_eps, inception_a1_5x5_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_5x5_reduce_bn_node));

    // inception_a1_5x5_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_5x5_reduce_relu Layer
    vx_size inception_a1_5x5_reduce_relu_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a1_5x5_reduce_relu;
    inception_a1_5x5_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a1_5x5_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_relu);
    vx_enum inception_a1_5x5_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_5x5_reduce_relu_param_a = 0;
    vx_float32 inception_a1_5x5_reduce_relu_param_b = 0;
    vx_node inception_a1_5x5_reduce_relu_node;
    inception_a1_5x5_reduce_relu_node = vxActivationLayer(graph, inception_a1_5x5_reduce_scale, inception_a1_5x5_reduce_relu_mode, inception_a1_5x5_reduce_relu_param_a, inception_a1_5x5_reduce_relu_param_b, inception_a1_5x5_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a1_5x5_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_5x5_reduce_relu_node));

    // inception_a1_5x5 Layer
    vx_size inception_a1_5x5_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_5x5;
    inception_a1_5x5 = vxCreateVirtualTensor(graph,4, inception_a1_5x5_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_5x5);
    vx_size inception_a1_5x5_W_dims[4] = { 5, 5, 48, 64 };
    vx_tensor inception_a1_5x5_W;
    inception_a1_5x5_W = vxCreateTensor(context,4, inception_a1_5x5_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_W, dataFolder + "/weights/inception_a1_5x5.f32"));
    vx_nn_convolution_params_t inception_a1_5x5_params;
    inception_a1_5x5_params.padding_x = 2;
    inception_a1_5x5_params.padding_y = 2;
    inception_a1_5x5_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_5x5_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_5x5_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_5x5_params.dilation_x = 0;
    inception_a1_5x5_params.dilation_y = 0;
    vx_node inception_a1_5x5_node;
    inception_a1_5x5_node = vxConvolutionLayer(graph, inception_a1_5x5_reduce_relu, inception_a1_5x5_W, NULL, &inception_a1_5x5_params, sizeof(inception_a1_5x5_params ), inception_a1_5x5);
    ERROR_CHECK_OBJECT(inception_a1_5x5_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_5x5_node));

    // inception_a1_5x5_bn Layer
    vx_size inception_a1_5x5_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_5x5_scale;
    inception_a1_5x5_scale = vxCreateVirtualTensor(graph,4, inception_a1_5x5_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_scale);
    vx_size inception_a1_5x5_bn_W_dims[1] = { 64 };
    vx_float32 inception_a1_5x5_bn_eps = 0.001;
    vx_tensor inception_a1_5x5_bn_W;
    inception_a1_5x5_bn_W = vxCreateTensor(context,1, inception_a1_5x5_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_bn_W, dataFolder + "/weights/inception_a1_5x5_bn.f32"));
    vx_size inception_a1_5x5_bn_B_dims[1] = { 64 };
    vx_tensor inception_a1_5x5_bn_B;
    inception_a1_5x5_bn_B = vxCreateTensor(context,1, inception_a1_5x5_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_bn_B, dataFolder + "/bias/inception_a1_5x5_bn.f32"));
    vx_size inception_a1_5x5_scale_W_dims[1] = { 64 };
    vx_tensor inception_a1_5x5_scale_W;
    inception_a1_5x5_scale_W = vxCreateTensor(context,1, inception_a1_5x5_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_scale_W, dataFolder + "/weights/inception_a1_5x5_scale.f32"));
    vx_size inception_a1_5x5_scale_B_dims[1] = { 64 };
    vx_tensor inception_a1_5x5_scale_B;
    inception_a1_5x5_scale_B = vxCreateTensor(context,1, inception_a1_5x5_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_5x5_scale_B, dataFolder + "/bias/inception_a1_5x5_scale.f32"));
    vx_node inception_a1_5x5_bn_node;
    inception_a1_5x5_bn_node = vxBatchNormalizationLayer(graph, inception_a1_5x5, inception_a1_5x5_bn_W, inception_a1_5x5_bn_B, inception_a1_5x5_scale_W, inception_a1_5x5_scale_B, inception_a1_5x5_bn_eps, inception_a1_5x5_scale);
    ERROR_CHECK_OBJECT(inception_a1_5x5_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_5x5_bn_node));

    // inception_a1_5x5_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_5x5_relu Layer
    vx_size inception_a1_5x5_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_5x5_relu;
    inception_a1_5x5_relu = vxCreateVirtualTensor(graph,4, inception_a1_5x5_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_5x5_relu);
    vx_enum inception_a1_5x5_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_5x5_relu_param_a = 0;
    vx_float32 inception_a1_5x5_relu_param_b = 0;
    vx_node inception_a1_5x5_relu_node;
    inception_a1_5x5_relu_node = vxActivationLayer(graph, inception_a1_5x5_scale, inception_a1_5x5_relu_mode, inception_a1_5x5_relu_param_a, inception_a1_5x5_relu_param_b, inception_a1_5x5_relu);
    ERROR_CHECK_OBJECT(inception_a1_5x5_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_5x5_relu_node));

    // inception_a1_3x3_reduce Layer
    vx_size inception_a1_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_reduce;
    inception_a1_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a1_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce);
    vx_size inception_a1_3x3_reduce_W_dims[4] = { 1, 1, 192, 64 };
    vx_tensor inception_a1_3x3_reduce_W;
    inception_a1_3x3_reduce_W = vxCreateTensor(context,4, inception_a1_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_W, dataFolder + "/weights/inception_a1_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_reduce_params;
    inception_a1_3x3_reduce_params.padding_x = 0;
    inception_a1_3x3_reduce_params.padding_y = 0;
    inception_a1_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_reduce_params.dilation_x = 0;
    inception_a1_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a1_3x3_reduce_node;
    inception_a1_3x3_reduce_node = vxConvolutionLayer(graph, pool2_3x3_s2_pool2_3x3_s2_0_split_2, inception_a1_3x3_reduce_W, NULL, &inception_a1_3x3_reduce_params, sizeof(inception_a1_3x3_reduce_params ), inception_a1_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_reduce_node));

    // inception_a1_3x3_reduce_bn Layer
    vx_size inception_a1_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_reduce_scale;
    inception_a1_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_scale);
    vx_size inception_a1_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a1_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_reduce_bn_W;
    inception_a1_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a1_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_bn_W, dataFolder + "/weights/inception_a1_3x3_reduce_bn.f32"));
    vx_size inception_a1_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_reduce_bn_B;
    inception_a1_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a1_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_bn_B, dataFolder + "/bias/inception_a1_3x3_reduce_bn.f32"));
    vx_size inception_a1_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_reduce_scale_W;
    inception_a1_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a1_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_scale_W, dataFolder + "/weights/inception_a1_3x3_reduce_scale.f32"));
    vx_size inception_a1_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_reduce_scale_B;
    inception_a1_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a1_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_scale_B, dataFolder + "/bias/inception_a1_3x3_reduce_scale.f32"));
    vx_node inception_a1_3x3_reduce_bn_node;
    inception_a1_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_reduce, inception_a1_3x3_reduce_bn_W, inception_a1_3x3_reduce_bn_B, inception_a1_3x3_reduce_scale_W, inception_a1_3x3_reduce_scale_B, inception_a1_3x3_reduce_bn_eps, inception_a1_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_reduce_bn_node));

    // inception_a1_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_reduce_relu Layer
    vx_size inception_a1_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_reduce_relu;
    inception_a1_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_relu);
    vx_enum inception_a1_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a1_3x3_reduce_relu_param_b = 0;
    vx_node inception_a1_3x3_reduce_relu_node;
    inception_a1_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a1_3x3_reduce_scale, inception_a1_3x3_reduce_relu_mode, inception_a1_3x3_reduce_relu_param_a, inception_a1_3x3_reduce_relu_param_b, inception_a1_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_reduce_relu_node));

    // inception_a1_3x3_1 Layer
    vx_size inception_a1_3x3_1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_1;
    inception_a1_3x3_1 = vxCreateVirtualTensor(graph,4, inception_a1_3x3_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1);
    vx_size inception_a1_3x3_1_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a1_3x3_1_W;
    inception_a1_3x3_1_W = vxCreateTensor(context,4, inception_a1_3x3_1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_1_W, dataFolder + "/weights/inception_a1_3x3_1.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_1_params;
    inception_a1_3x3_1_params.padding_x = 1;
    inception_a1_3x3_1_params.padding_y = 1;
    inception_a1_3x3_1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_1_params.dilation_x = 0;
    inception_a1_3x3_1_params.dilation_y = 0;
    vx_node inception_a1_3x3_1_node;
    inception_a1_3x3_1_node = vxConvolutionLayer(graph, inception_a1_3x3_reduce_relu, inception_a1_3x3_1_W, NULL, &inception_a1_3x3_1_params, sizeof(inception_a1_3x3_1_params ), inception_a1_3x3_1);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_1_node));

    // inception_a1_3x3_1_bn Layer
    vx_size inception_a1_3x3_1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_1_scale;
    inception_a1_3x3_1_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_scale);
    vx_size inception_a1_3x3_1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_3x3_1_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_1_bn_W;
    inception_a1_3x3_1_bn_W = vxCreateTensor(context,1, inception_a1_3x3_1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_1_bn_W, dataFolder + "/weights/inception_a1_3x3_1_bn.f32"));
    vx_size inception_a1_3x3_1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_1_bn_B;
    inception_a1_3x3_1_bn_B = vxCreateTensor(context,1, inception_a1_3x3_1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_1_bn_B, dataFolder + "/bias/inception_a1_3x3_1_bn.f32"));
    vx_size inception_a1_3x3_1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_1_scale_W;
    inception_a1_3x3_1_scale_W = vxCreateTensor(context,1, inception_a1_3x3_1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_1_scale_W, dataFolder + "/weights/inception_a1_3x3_1_scale.f32"));
    vx_size inception_a1_3x3_1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_1_scale_B;
    inception_a1_3x3_1_scale_B = vxCreateTensor(context,1, inception_a1_3x3_1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_1_scale_B, dataFolder + "/bias/inception_a1_3x3_1_scale.f32"));
    vx_node inception_a1_3x3_1_bn_node;
    inception_a1_3x3_1_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_1, inception_a1_3x3_1_bn_W, inception_a1_3x3_1_bn_B, inception_a1_3x3_1_scale_W, inception_a1_3x3_1_scale_B, inception_a1_3x3_1_bn_eps, inception_a1_3x3_1_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_1_bn_node));

    // inception_a1_3x3_1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_1_relu Layer
    vx_size inception_a1_3x3_1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_1_relu;
    inception_a1_3x3_1_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_relu);
    vx_enum inception_a1_3x3_1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_1_relu_param_a = 0;
    vx_float32 inception_a1_3x3_1_relu_param_b = 0;
    vx_node inception_a1_3x3_1_relu_node;
    inception_a1_3x3_1_relu_node = vxActivationLayer(graph, inception_a1_3x3_1_scale, inception_a1_3x3_1_relu_mode, inception_a1_3x3_1_relu_param_a, inception_a1_3x3_1_relu_param_b, inception_a1_3x3_1_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_1_relu_node));

    // inception_a1_3x3_2 Layer
    vx_size inception_a1_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_2;
    inception_a1_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2);
    vx_size inception_a1_3x3_2_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a1_3x3_2_W;
    inception_a1_3x3_2_W = vxCreateTensor(context,4, inception_a1_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_W, dataFolder + "/weights/inception_a1_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_2_params;
    inception_a1_3x3_2_params.padding_x = 1;
    inception_a1_3x3_2_params.padding_y = 1;
    inception_a1_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_2_params.dilation_x = 0;
    inception_a1_3x3_2_params.dilation_y = 0;
    vx_node inception_a1_3x3_2_node;
    inception_a1_3x3_2_node = vxConvolutionLayer(graph, inception_a1_3x3_1_relu, inception_a1_3x3_2_W, NULL, &inception_a1_3x3_2_params, sizeof(inception_a1_3x3_2_params ), inception_a1_3x3_2);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_node));

    // inception_a1_3x3_2_bn Layer
    vx_size inception_a1_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_2_scale;
    inception_a1_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_scale);
    vx_size inception_a1_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_2_bn_W;
    inception_a1_3x3_2_bn_W = vxCreateTensor(context,1, inception_a1_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_bn_W, dataFolder + "/weights/inception_a1_3x3_2_bn.f32"));
    vx_size inception_a1_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_2_bn_B;
    inception_a1_3x3_2_bn_B = vxCreateTensor(context,1, inception_a1_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_bn_B, dataFolder + "/bias/inception_a1_3x3_2_bn.f32"));
    vx_size inception_a1_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_2_scale_W;
    inception_a1_3x3_2_scale_W = vxCreateTensor(context,1, inception_a1_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_scale_W, dataFolder + "/weights/inception_a1_3x3_2_scale.f32"));
    vx_size inception_a1_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_2_scale_B;
    inception_a1_3x3_2_scale_B = vxCreateTensor(context,1, inception_a1_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_scale_B, dataFolder + "/bias/inception_a1_3x3_2_scale.f32"));
    vx_node inception_a1_3x3_2_bn_node;
    inception_a1_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_2, inception_a1_3x3_2_bn_W, inception_a1_3x3_2_bn_B, inception_a1_3x3_2_scale_W, inception_a1_3x3_2_scale_B, inception_a1_3x3_2_bn_eps, inception_a1_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_bn_node));

    // inception_a1_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_2_relu Layer
    vx_size inception_a1_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_2_relu;
    inception_a1_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_relu);
    vx_enum inception_a1_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_2_relu_param_a = 0;
    vx_float32 inception_a1_3x3_2_relu_param_b = 0;
    vx_node inception_a1_3x3_2_relu_node;
    inception_a1_3x3_2_relu_node = vxActivationLayer(graph, inception_a1_3x3_2_scale, inception_a1_3x3_2_relu_mode, inception_a1_3x3_2_relu_param_a, inception_a1_3x3_2_relu_param_b, inception_a1_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_relu_node));

    // inception_a1_pool Layer
    vx_size inception_a1_pool_dims[4] = { 35, 35, 192, 1 };
    vx_tensor inception_a1_pool;
    inception_a1_pool = vxCreateVirtualTensor(graph,4, inception_a1_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_pool);
    vx_enum inception_a1_pool_type = VX_NN_POOLING_AVG;
    vx_size inception_a1_pool_kernel_w = 3;
    vx_size inception_a1_pool_kernel_h = 3;
    vx_size inception_a1_pool_pad_w = 1;
    vx_size inception_a1_pool_pad_h = 1;
    vx_enum inception_a1_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a1_pool_node;
    inception_a1_pool_node = vxPoolingLayer(graph, pool2_3x3_s2_pool2_3x3_s2_0_split_3, inception_a1_pool_type, inception_a1_pool_kernel_w, inception_a1_pool_kernel_h, inception_a1_pool_pad_w, inception_a1_pool_pad_h, inception_a1_pool_roundPolicy, inception_a1_pool );
    ERROR_CHECK_OBJECT(inception_a1_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_pool_node));

    // inception_a1_pool_proj Layer
    vx_size inception_a1_pool_proj_dims[4] = { 35, 35, 32, 1 };
    vx_tensor inception_a1_pool_proj;
    inception_a1_pool_proj = vxCreateVirtualTensor(graph,4, inception_a1_pool_proj_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj);
    vx_size inception_a1_pool_proj_W_dims[4] = { 1, 1, 192, 32 };
    vx_tensor inception_a1_pool_proj_W;
    inception_a1_pool_proj_W = vxCreateTensor(context,4, inception_a1_pool_proj_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_pool_proj_W, dataFolder + "/weights/inception_a1_pool_proj.f32"));
    vx_nn_convolution_params_t inception_a1_pool_proj_params;
    inception_a1_pool_proj_params.padding_x = 0;
    inception_a1_pool_proj_params.padding_y = 0;
    inception_a1_pool_proj_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_pool_proj_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_pool_proj_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_pool_proj_params.dilation_x = 0;
    inception_a1_pool_proj_params.dilation_y = 0;
    vx_node inception_a1_pool_proj_node;
    inception_a1_pool_proj_node = vxConvolutionLayer(graph, inception_a1_pool, inception_a1_pool_proj_W, NULL, &inception_a1_pool_proj_params, sizeof(inception_a1_pool_proj_params ), inception_a1_pool_proj);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_pool_proj_node));

    // inception_a1_pool_proj_bn Layer
    vx_size inception_a1_pool_proj_scale_dims[4] = { 35, 35, 32, 1 };
    vx_tensor inception_a1_pool_proj_scale;
    inception_a1_pool_proj_scale = vxCreateVirtualTensor(graph,4, inception_a1_pool_proj_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_scale);
    vx_size inception_a1_pool_proj_bn_W_dims[1] = { 32 };
    vx_float32 inception_a1_pool_proj_bn_eps = 0.001;
    vx_tensor inception_a1_pool_proj_bn_W;
    inception_a1_pool_proj_bn_W = vxCreateTensor(context,1, inception_a1_pool_proj_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_pool_proj_bn_W, dataFolder + "/weights/inception_a1_pool_proj_bn.f32"));
    vx_size inception_a1_pool_proj_bn_B_dims[1] = { 32 };
    vx_tensor inception_a1_pool_proj_bn_B;
    inception_a1_pool_proj_bn_B = vxCreateTensor(context,1, inception_a1_pool_proj_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_pool_proj_bn_B, dataFolder + "/bias/inception_a1_pool_proj_bn.f32"));
    vx_size inception_a1_pool_proj_scale_W_dims[1] = { 32 };
    vx_tensor inception_a1_pool_proj_scale_W;
    inception_a1_pool_proj_scale_W = vxCreateTensor(context,1, inception_a1_pool_proj_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_pool_proj_scale_W, dataFolder + "/weights/inception_a1_pool_proj_scale.f32"));
    vx_size inception_a1_pool_proj_scale_B_dims[1] = { 32 };
    vx_tensor inception_a1_pool_proj_scale_B;
    inception_a1_pool_proj_scale_B = vxCreateTensor(context,1, inception_a1_pool_proj_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_pool_proj_scale_B, dataFolder + "/bias/inception_a1_pool_proj_scale.f32"));
    vx_node inception_a1_pool_proj_bn_node;
    inception_a1_pool_proj_bn_node = vxBatchNormalizationLayer(graph, inception_a1_pool_proj, inception_a1_pool_proj_bn_W, inception_a1_pool_proj_bn_B, inception_a1_pool_proj_scale_W, inception_a1_pool_proj_scale_B, inception_a1_pool_proj_bn_eps, inception_a1_pool_proj_scale);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_pool_proj_bn_node));

    // inception_a1_pool_proj_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_pool_proj_relu Layer
    vx_size inception_a1_pool_proj_relu_dims[4] = { 35, 35, 32, 1 };
    vx_tensor inception_a1_pool_proj_relu;
    inception_a1_pool_proj_relu = vxCreateVirtualTensor(graph,4, inception_a1_pool_proj_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_relu);
    vx_enum inception_a1_pool_proj_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_pool_proj_relu_param_a = 0;
    vx_float32 inception_a1_pool_proj_relu_param_b = 0;
    vx_node inception_a1_pool_proj_relu_node;
    inception_a1_pool_proj_relu_node = vxActivationLayer(graph, inception_a1_pool_proj_scale, inception_a1_pool_proj_relu_mode, inception_a1_pool_proj_relu_param_a, inception_a1_pool_proj_relu_param_b, inception_a1_pool_proj_relu);
    ERROR_CHECK_OBJECT(inception_a1_pool_proj_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_pool_proj_relu_node));

    // inception_a1_output Layer
    vx_size inception_a1_output_dims[4] = { 35, 35, 256, 1 };
    vx_tensor inception_a1_output;
    inception_a1_output = vxCreateVirtualTensor(graph,4, inception_a1_output_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_output);
    vx_node inception_a1_output_node;
    inception_a1_output_node = vxConcatLayer(graph, inception_a1_output, inception_a1_1x1_relu, inception_a1_5x5_relu, inception_a1_3x3_2_relu, inception_a1_pool_proj_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a1_output_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_output_node));

    // inception_a1_output_inception_a1_output_0_split_0 Layer
    vx_size inception_a1_output_inception_a1_output_0_split_0_dims[4] = { 35, 35, 256, 1 };
    vx_tensor inception_a1_output_inception_a1_output_0_split_0;
    inception_a1_output_inception_a1_output_0_split_0 = vxCreateVirtualTensor(graph,4, inception_a1_output_inception_a1_output_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_0);
    vx_node inception_a1_output_inception_a1_output_0_split_0_node;
    inception_a1_output_inception_a1_output_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_a1_output, (vx_reference)inception_a1_output_inception_a1_output_0_split_0);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_output_inception_a1_output_0_split_0_node));

    // inception_a1_output_inception_a1_output_0_split_1 Layer
    vx_size inception_a1_output_inception_a1_output_0_split_1_dims[4] = { 35, 35, 256, 1 };
    vx_tensor inception_a1_output_inception_a1_output_0_split_1;
    inception_a1_output_inception_a1_output_0_split_1 = vxCreateVirtualTensor(graph,4, inception_a1_output_inception_a1_output_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_1);
    vx_node inception_a1_output_inception_a1_output_0_split_1_node;
    inception_a1_output_inception_a1_output_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_a1_output, (vx_reference)inception_a1_output_inception_a1_output_0_split_1);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_output_inception_a1_output_0_split_1_node));

    // inception_a1_output_inception_a1_output_0_split_2 Layer
    vx_size inception_a1_output_inception_a1_output_0_split_2_dims[4] = { 35, 35, 256, 1 };
    vx_tensor inception_a1_output_inception_a1_output_0_split_2;
    inception_a1_output_inception_a1_output_0_split_2 = vxCreateVirtualTensor(graph,4, inception_a1_output_inception_a1_output_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_2);
    vx_node inception_a1_output_inception_a1_output_0_split_2_node;
    inception_a1_output_inception_a1_output_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_a1_output, (vx_reference)inception_a1_output_inception_a1_output_0_split_2);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_output_inception_a1_output_0_split_2_node));

    // inception_a1_output_inception_a1_output_0_split_3 Layer
    vx_size inception_a1_output_inception_a1_output_0_split_3_dims[4] = { 35, 35, 256, 1 };
    vx_tensor inception_a1_output_inception_a1_output_0_split_3;
    inception_a1_output_inception_a1_output_0_split_3 = vxCreateVirtualTensor(graph,4, inception_a1_output_inception_a1_output_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_3);
    vx_node inception_a1_output_inception_a1_output_0_split_3_node;
    inception_a1_output_inception_a1_output_0_split_3_node = vxCopyNode( graph, (vx_reference)inception_a1_output, (vx_reference)inception_a1_output_inception_a1_output_0_split_3);
    ERROR_CHECK_OBJECT(inception_a1_output_inception_a1_output_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_output_inception_a1_output_0_split_3_node));

    // inception_a2_1x1 Layer
    vx_size inception_a2_1x1_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_1x1;
    inception_a2_1x1 = vxCreateVirtualTensor(graph,4, inception_a2_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1);
    vx_size inception_a2_1x1_W_dims[4] = { 1, 1, 256, 64 };
    vx_tensor inception_a2_1x1_W;
    inception_a2_1x1_W = vxCreateTensor(context,4, inception_a2_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_W, dataFolder + "/weights/inception_a2_1x1.f32"));
    vx_nn_convolution_params_t inception_a2_1x1_params;
    inception_a2_1x1_params.padding_x = 0;
    inception_a2_1x1_params.padding_y = 0;
    inception_a2_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_1x1_params.dilation_x = 0;
    inception_a2_1x1_params.dilation_y = 0;
    vx_node inception_a2_1x1_node;
    inception_a2_1x1_node = vxConvolutionLayer(graph, inception_a1_output_inception_a1_output_0_split_0, inception_a2_1x1_W, NULL, &inception_a2_1x1_params, sizeof(inception_a2_1x1_params ), inception_a2_1x1);
    ERROR_CHECK_OBJECT(inception_a2_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_node));

    // inception_a2_1x1_bn Layer
    vx_size inception_a2_1x1_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_1x1_scale;
    inception_a2_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a2_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_scale);
    vx_size inception_a2_1x1_bn_W_dims[1] = { 64 };
    vx_float32 inception_a2_1x1_bn_eps = 0.001;
    vx_tensor inception_a2_1x1_bn_W;
    inception_a2_1x1_bn_W = vxCreateTensor(context,1, inception_a2_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_bn_W, dataFolder + "/weights/inception_a2_1x1_bn.f32"));
    vx_size inception_a2_1x1_bn_B_dims[1] = { 64 };
    vx_tensor inception_a2_1x1_bn_B;
    inception_a2_1x1_bn_B = vxCreateTensor(context,1, inception_a2_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_bn_B, dataFolder + "/bias/inception_a2_1x1_bn.f32"));
    vx_size inception_a2_1x1_scale_W_dims[1] = { 64 };
    vx_tensor inception_a2_1x1_scale_W;
    inception_a2_1x1_scale_W = vxCreateTensor(context,1, inception_a2_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_scale_W, dataFolder + "/weights/inception_a2_1x1_scale.f32"));
    vx_size inception_a2_1x1_scale_B_dims[1] = { 64 };
    vx_tensor inception_a2_1x1_scale_B;
    inception_a2_1x1_scale_B = vxCreateTensor(context,1, inception_a2_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_scale_B, dataFolder + "/bias/inception_a2_1x1_scale.f32"));
    vx_node inception_a2_1x1_bn_node;
    inception_a2_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a2_1x1, inception_a2_1x1_bn_W, inception_a2_1x1_bn_B, inception_a2_1x1_scale_W, inception_a2_1x1_scale_B, inception_a2_1x1_bn_eps, inception_a2_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a2_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_bn_node));

    // inception_a2_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_1x1_relu Layer
    vx_size inception_a2_1x1_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_1x1_relu;
    inception_a2_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a2_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_relu);
    vx_enum inception_a2_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_1x1_relu_param_a = 0;
    vx_float32 inception_a2_1x1_relu_param_b = 0;
    vx_node inception_a2_1x1_relu_node;
    inception_a2_1x1_relu_node = vxActivationLayer(graph, inception_a2_1x1_scale, inception_a2_1x1_relu_mode, inception_a2_1x1_relu_param_a, inception_a2_1x1_relu_param_b, inception_a2_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a2_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_relu_node));

    // inception_a2_5x5_reduce Layer
    vx_size inception_a2_5x5_reduce_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a2_5x5_reduce;
    inception_a2_5x5_reduce = vxCreateVirtualTensor(graph,4, inception_a2_5x5_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce);
    vx_size inception_a2_5x5_reduce_W_dims[4] = { 1, 1, 256, 48 };
    vx_tensor inception_a2_5x5_reduce_W;
    inception_a2_5x5_reduce_W = vxCreateTensor(context,4, inception_a2_5x5_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_reduce_W, dataFolder + "/weights/inception_a2_5x5_reduce.f32"));
    vx_nn_convolution_params_t inception_a2_5x5_reduce_params;
    inception_a2_5x5_reduce_params.padding_x = 0;
    inception_a2_5x5_reduce_params.padding_y = 0;
    inception_a2_5x5_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_5x5_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_5x5_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_5x5_reduce_params.dilation_x = 0;
    inception_a2_5x5_reduce_params.dilation_y = 0;
    vx_node inception_a2_5x5_reduce_node;
    inception_a2_5x5_reduce_node = vxConvolutionLayer(graph, inception_a1_output_inception_a1_output_0_split_1, inception_a2_5x5_reduce_W, NULL, &inception_a2_5x5_reduce_params, sizeof(inception_a2_5x5_reduce_params ), inception_a2_5x5_reduce);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_5x5_reduce_node));

    // inception_a2_5x5_reduce_bn Layer
    vx_size inception_a2_5x5_reduce_scale_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a2_5x5_reduce_scale;
    inception_a2_5x5_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a2_5x5_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_scale);
    vx_size inception_a2_5x5_reduce_bn_W_dims[1] = { 48 };
    vx_float32 inception_a2_5x5_reduce_bn_eps = 0.001;
    vx_tensor inception_a2_5x5_reduce_bn_W;
    inception_a2_5x5_reduce_bn_W = vxCreateTensor(context,1, inception_a2_5x5_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_reduce_bn_W, dataFolder + "/weights/inception_a2_5x5_reduce_bn.f32"));
    vx_size inception_a2_5x5_reduce_bn_B_dims[1] = { 48 };
    vx_tensor inception_a2_5x5_reduce_bn_B;
    inception_a2_5x5_reduce_bn_B = vxCreateTensor(context,1, inception_a2_5x5_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_reduce_bn_B, dataFolder + "/bias/inception_a2_5x5_reduce_bn.f32"));
    vx_size inception_a2_5x5_reduce_scale_W_dims[1] = { 48 };
    vx_tensor inception_a2_5x5_reduce_scale_W;
    inception_a2_5x5_reduce_scale_W = vxCreateTensor(context,1, inception_a2_5x5_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_reduce_scale_W, dataFolder + "/weights/inception_a2_5x5_reduce_scale.f32"));
    vx_size inception_a2_5x5_reduce_scale_B_dims[1] = { 48 };
    vx_tensor inception_a2_5x5_reduce_scale_B;
    inception_a2_5x5_reduce_scale_B = vxCreateTensor(context,1, inception_a2_5x5_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_reduce_scale_B, dataFolder + "/bias/inception_a2_5x5_reduce_scale.f32"));
    vx_node inception_a2_5x5_reduce_bn_node;
    inception_a2_5x5_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a2_5x5_reduce, inception_a2_5x5_reduce_bn_W, inception_a2_5x5_reduce_bn_B, inception_a2_5x5_reduce_scale_W, inception_a2_5x5_reduce_scale_B, inception_a2_5x5_reduce_bn_eps, inception_a2_5x5_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_5x5_reduce_bn_node));

    // inception_a2_5x5_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_5x5_reduce_relu Layer
    vx_size inception_a2_5x5_reduce_relu_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a2_5x5_reduce_relu;
    inception_a2_5x5_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a2_5x5_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_relu);
    vx_enum inception_a2_5x5_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_5x5_reduce_relu_param_a = 0;
    vx_float32 inception_a2_5x5_reduce_relu_param_b = 0;
    vx_node inception_a2_5x5_reduce_relu_node;
    inception_a2_5x5_reduce_relu_node = vxActivationLayer(graph, inception_a2_5x5_reduce_scale, inception_a2_5x5_reduce_relu_mode, inception_a2_5x5_reduce_relu_param_a, inception_a2_5x5_reduce_relu_param_b, inception_a2_5x5_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a2_5x5_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_5x5_reduce_relu_node));

    // inception_a2_5x5 Layer
    vx_size inception_a2_5x5_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_5x5;
    inception_a2_5x5 = vxCreateVirtualTensor(graph,4, inception_a2_5x5_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_5x5);
    vx_size inception_a2_5x5_W_dims[4] = { 5, 5, 48, 64 };
    vx_tensor inception_a2_5x5_W;
    inception_a2_5x5_W = vxCreateTensor(context,4, inception_a2_5x5_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_W, dataFolder + "/weights/inception_a2_5x5.f32"));
    vx_nn_convolution_params_t inception_a2_5x5_params;
    inception_a2_5x5_params.padding_x = 2;
    inception_a2_5x5_params.padding_y = 2;
    inception_a2_5x5_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_5x5_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_5x5_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_5x5_params.dilation_x = 0;
    inception_a2_5x5_params.dilation_y = 0;
    vx_node inception_a2_5x5_node;
    inception_a2_5x5_node = vxConvolutionLayer(graph, inception_a2_5x5_reduce_relu, inception_a2_5x5_W, NULL, &inception_a2_5x5_params, sizeof(inception_a2_5x5_params ), inception_a2_5x5);
    ERROR_CHECK_OBJECT(inception_a2_5x5_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_5x5_node));

    // inception_a2_5x5_bn Layer
    vx_size inception_a2_5x5_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_5x5_scale;
    inception_a2_5x5_scale = vxCreateVirtualTensor(graph,4, inception_a2_5x5_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_scale);
    vx_size inception_a2_5x5_bn_W_dims[1] = { 64 };
    vx_float32 inception_a2_5x5_bn_eps = 0.001;
    vx_tensor inception_a2_5x5_bn_W;
    inception_a2_5x5_bn_W = vxCreateTensor(context,1, inception_a2_5x5_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_bn_W, dataFolder + "/weights/inception_a2_5x5_bn.f32"));
    vx_size inception_a2_5x5_bn_B_dims[1] = { 64 };
    vx_tensor inception_a2_5x5_bn_B;
    inception_a2_5x5_bn_B = vxCreateTensor(context,1, inception_a2_5x5_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_bn_B, dataFolder + "/bias/inception_a2_5x5_bn.f32"));
    vx_size inception_a2_5x5_scale_W_dims[1] = { 64 };
    vx_tensor inception_a2_5x5_scale_W;
    inception_a2_5x5_scale_W = vxCreateTensor(context,1, inception_a2_5x5_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_scale_W, dataFolder + "/weights/inception_a2_5x5_scale.f32"));
    vx_size inception_a2_5x5_scale_B_dims[1] = { 64 };
    vx_tensor inception_a2_5x5_scale_B;
    inception_a2_5x5_scale_B = vxCreateTensor(context,1, inception_a2_5x5_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_5x5_scale_B, dataFolder + "/bias/inception_a2_5x5_scale.f32"));
    vx_node inception_a2_5x5_bn_node;
    inception_a2_5x5_bn_node = vxBatchNormalizationLayer(graph, inception_a2_5x5, inception_a2_5x5_bn_W, inception_a2_5x5_bn_B, inception_a2_5x5_scale_W, inception_a2_5x5_scale_B, inception_a2_5x5_bn_eps, inception_a2_5x5_scale);
    ERROR_CHECK_OBJECT(inception_a2_5x5_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_5x5_bn_node));

    // inception_a2_5x5_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_5x5_relu Layer
    vx_size inception_a2_5x5_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_5x5_relu;
    inception_a2_5x5_relu = vxCreateVirtualTensor(graph,4, inception_a2_5x5_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_5x5_relu);
    vx_enum inception_a2_5x5_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_5x5_relu_param_a = 0;
    vx_float32 inception_a2_5x5_relu_param_b = 0;
    vx_node inception_a2_5x5_relu_node;
    inception_a2_5x5_relu_node = vxActivationLayer(graph, inception_a2_5x5_scale, inception_a2_5x5_relu_mode, inception_a2_5x5_relu_param_a, inception_a2_5x5_relu_param_b, inception_a2_5x5_relu);
    ERROR_CHECK_OBJECT(inception_a2_5x5_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_5x5_relu_node));

    // inception_a2_3x3_reduce Layer
    vx_size inception_a2_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_reduce;
    inception_a2_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a2_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce);
    vx_size inception_a2_3x3_reduce_W_dims[4] = { 1, 1, 256, 64 };
    vx_tensor inception_a2_3x3_reduce_W;
    inception_a2_3x3_reduce_W = vxCreateTensor(context,4, inception_a2_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_W, dataFolder + "/weights/inception_a2_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_reduce_params;
    inception_a2_3x3_reduce_params.padding_x = 0;
    inception_a2_3x3_reduce_params.padding_y = 0;
    inception_a2_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_reduce_params.dilation_x = 0;
    inception_a2_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a2_3x3_reduce_node;
    inception_a2_3x3_reduce_node = vxConvolutionLayer(graph, inception_a1_output_inception_a1_output_0_split_2, inception_a2_3x3_reduce_W, NULL, &inception_a2_3x3_reduce_params, sizeof(inception_a2_3x3_reduce_params ), inception_a2_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_reduce_node));

    // inception_a2_3x3_reduce_bn Layer
    vx_size inception_a2_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_reduce_scale;
    inception_a2_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_scale);
    vx_size inception_a2_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a2_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_reduce_bn_W;
    inception_a2_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a2_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_bn_W, dataFolder + "/weights/inception_a2_3x3_reduce_bn.f32"));
    vx_size inception_a2_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_reduce_bn_B;
    inception_a2_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a2_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_bn_B, dataFolder + "/bias/inception_a2_3x3_reduce_bn.f32"));
    vx_size inception_a2_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_reduce_scale_W;
    inception_a2_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a2_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_scale_W, dataFolder + "/weights/inception_a2_3x3_reduce_scale.f32"));
    vx_size inception_a2_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_reduce_scale_B;
    inception_a2_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a2_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_scale_B, dataFolder + "/bias/inception_a2_3x3_reduce_scale.f32"));
    vx_node inception_a2_3x3_reduce_bn_node;
    inception_a2_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_reduce, inception_a2_3x3_reduce_bn_W, inception_a2_3x3_reduce_bn_B, inception_a2_3x3_reduce_scale_W, inception_a2_3x3_reduce_scale_B, inception_a2_3x3_reduce_bn_eps, inception_a2_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_reduce_bn_node));

    // inception_a2_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_reduce_relu Layer
    vx_size inception_a2_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_reduce_relu;
    inception_a2_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_relu);
    vx_enum inception_a2_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a2_3x3_reduce_relu_param_b = 0;
    vx_node inception_a2_3x3_reduce_relu_node;
    inception_a2_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a2_3x3_reduce_scale, inception_a2_3x3_reduce_relu_mode, inception_a2_3x3_reduce_relu_param_a, inception_a2_3x3_reduce_relu_param_b, inception_a2_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_reduce_relu_node));

    // inception_a2_3x3_1 Layer
    vx_size inception_a2_3x3_1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_1;
    inception_a2_3x3_1 = vxCreateVirtualTensor(graph,4, inception_a2_3x3_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1);
    vx_size inception_a2_3x3_1_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a2_3x3_1_W;
    inception_a2_3x3_1_W = vxCreateTensor(context,4, inception_a2_3x3_1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_1_W, dataFolder + "/weights/inception_a2_3x3_1.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_1_params;
    inception_a2_3x3_1_params.padding_x = 1;
    inception_a2_3x3_1_params.padding_y = 1;
    inception_a2_3x3_1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_1_params.dilation_x = 0;
    inception_a2_3x3_1_params.dilation_y = 0;
    vx_node inception_a2_3x3_1_node;
    inception_a2_3x3_1_node = vxConvolutionLayer(graph, inception_a2_3x3_reduce_relu, inception_a2_3x3_1_W, NULL, &inception_a2_3x3_1_params, sizeof(inception_a2_3x3_1_params ), inception_a2_3x3_1);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_1_node));

    // inception_a2_3x3_1_bn Layer
    vx_size inception_a2_3x3_1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_1_scale;
    inception_a2_3x3_1_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_scale);
    vx_size inception_a2_3x3_1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_3x3_1_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_1_bn_W;
    inception_a2_3x3_1_bn_W = vxCreateTensor(context,1, inception_a2_3x3_1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_1_bn_W, dataFolder + "/weights/inception_a2_3x3_1_bn.f32"));
    vx_size inception_a2_3x3_1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_1_bn_B;
    inception_a2_3x3_1_bn_B = vxCreateTensor(context,1, inception_a2_3x3_1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_1_bn_B, dataFolder + "/bias/inception_a2_3x3_1_bn.f32"));
    vx_size inception_a2_3x3_1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_1_scale_W;
    inception_a2_3x3_1_scale_W = vxCreateTensor(context,1, inception_a2_3x3_1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_1_scale_W, dataFolder + "/weights/inception_a2_3x3_1_scale.f32"));
    vx_size inception_a2_3x3_1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_1_scale_B;
    inception_a2_3x3_1_scale_B = vxCreateTensor(context,1, inception_a2_3x3_1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_1_scale_B, dataFolder + "/bias/inception_a2_3x3_1_scale.f32"));
    vx_node inception_a2_3x3_1_bn_node;
    inception_a2_3x3_1_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_1, inception_a2_3x3_1_bn_W, inception_a2_3x3_1_bn_B, inception_a2_3x3_1_scale_W, inception_a2_3x3_1_scale_B, inception_a2_3x3_1_bn_eps, inception_a2_3x3_1_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_1_bn_node));

    // inception_a2_3x3_1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_1_relu Layer
    vx_size inception_a2_3x3_1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_1_relu;
    inception_a2_3x3_1_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_relu);
    vx_enum inception_a2_3x3_1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_1_relu_param_a = 0;
    vx_float32 inception_a2_3x3_1_relu_param_b = 0;
    vx_node inception_a2_3x3_1_relu_node;
    inception_a2_3x3_1_relu_node = vxActivationLayer(graph, inception_a2_3x3_1_scale, inception_a2_3x3_1_relu_mode, inception_a2_3x3_1_relu_param_a, inception_a2_3x3_1_relu_param_b, inception_a2_3x3_1_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_1_relu_node));

    // inception_a2_3x3_2 Layer
    vx_size inception_a2_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_2;
    inception_a2_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2);
    vx_size inception_a2_3x3_2_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a2_3x3_2_W;
    inception_a2_3x3_2_W = vxCreateTensor(context,4, inception_a2_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_W, dataFolder + "/weights/inception_a2_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_2_params;
    inception_a2_3x3_2_params.padding_x = 1;
    inception_a2_3x3_2_params.padding_y = 1;
    inception_a2_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_2_params.dilation_x = 0;
    inception_a2_3x3_2_params.dilation_y = 0;
    vx_node inception_a2_3x3_2_node;
    inception_a2_3x3_2_node = vxConvolutionLayer(graph, inception_a2_3x3_1_relu, inception_a2_3x3_2_W, NULL, &inception_a2_3x3_2_params, sizeof(inception_a2_3x3_2_params ), inception_a2_3x3_2);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_node));

    // inception_a2_3x3_2_bn Layer
    vx_size inception_a2_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_2_scale;
    inception_a2_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_scale);
    vx_size inception_a2_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_2_bn_W;
    inception_a2_3x3_2_bn_W = vxCreateTensor(context,1, inception_a2_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_bn_W, dataFolder + "/weights/inception_a2_3x3_2_bn.f32"));
    vx_size inception_a2_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_2_bn_B;
    inception_a2_3x3_2_bn_B = vxCreateTensor(context,1, inception_a2_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_bn_B, dataFolder + "/bias/inception_a2_3x3_2_bn.f32"));
    vx_size inception_a2_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_2_scale_W;
    inception_a2_3x3_2_scale_W = vxCreateTensor(context,1, inception_a2_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_scale_W, dataFolder + "/weights/inception_a2_3x3_2_scale.f32"));
    vx_size inception_a2_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_2_scale_B;
    inception_a2_3x3_2_scale_B = vxCreateTensor(context,1, inception_a2_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_scale_B, dataFolder + "/bias/inception_a2_3x3_2_scale.f32"));
    vx_node inception_a2_3x3_2_bn_node;
    inception_a2_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_2, inception_a2_3x3_2_bn_W, inception_a2_3x3_2_bn_B, inception_a2_3x3_2_scale_W, inception_a2_3x3_2_scale_B, inception_a2_3x3_2_bn_eps, inception_a2_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_bn_node));

    // inception_a2_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_2_relu Layer
    vx_size inception_a2_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_2_relu;
    inception_a2_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_relu);
    vx_enum inception_a2_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_2_relu_param_a = 0;
    vx_float32 inception_a2_3x3_2_relu_param_b = 0;
    vx_node inception_a2_3x3_2_relu_node;
    inception_a2_3x3_2_relu_node = vxActivationLayer(graph, inception_a2_3x3_2_scale, inception_a2_3x3_2_relu_mode, inception_a2_3x3_2_relu_param_a, inception_a2_3x3_2_relu_param_b, inception_a2_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_relu_node));

    // inception_a2_pool Layer
    vx_size inception_a2_pool_dims[4] = { 35, 35, 256, 1 };
    vx_tensor inception_a2_pool;
    inception_a2_pool = vxCreateVirtualTensor(graph,4, inception_a2_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_pool);
    vx_enum inception_a2_pool_type = VX_NN_POOLING_AVG;
    vx_size inception_a2_pool_kernel_w = 3;
    vx_size inception_a2_pool_kernel_h = 3;
    vx_size inception_a2_pool_pad_w = 1;
    vx_size inception_a2_pool_pad_h = 1;
    vx_enum inception_a2_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a2_pool_node;
    inception_a2_pool_node = vxPoolingLayer(graph, inception_a1_output_inception_a1_output_0_split_3, inception_a2_pool_type, inception_a2_pool_kernel_w, inception_a2_pool_kernel_h, inception_a2_pool_pad_w, inception_a2_pool_pad_h, inception_a2_pool_roundPolicy, inception_a2_pool );
    ERROR_CHECK_OBJECT(inception_a2_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_pool_node));

    // inception_a2_pool_proj Layer
    vx_size inception_a2_pool_proj_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_pool_proj;
    inception_a2_pool_proj = vxCreateVirtualTensor(graph,4, inception_a2_pool_proj_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj);
    vx_size inception_a2_pool_proj_W_dims[4] = { 1, 1, 256, 64 };
    vx_tensor inception_a2_pool_proj_W;
    inception_a2_pool_proj_W = vxCreateTensor(context,4, inception_a2_pool_proj_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_pool_proj_W, dataFolder + "/weights/inception_a2_pool_proj.f32"));
    vx_nn_convolution_params_t inception_a2_pool_proj_params;
    inception_a2_pool_proj_params.padding_x = 0;
    inception_a2_pool_proj_params.padding_y = 0;
    inception_a2_pool_proj_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_pool_proj_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_pool_proj_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_pool_proj_params.dilation_x = 0;
    inception_a2_pool_proj_params.dilation_y = 0;
    vx_node inception_a2_pool_proj_node;
    inception_a2_pool_proj_node = vxConvolutionLayer(graph, inception_a2_pool, inception_a2_pool_proj_W, NULL, &inception_a2_pool_proj_params, sizeof(inception_a2_pool_proj_params ), inception_a2_pool_proj);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_pool_proj_node));

    // inception_a2_pool_proj_bn Layer
    vx_size inception_a2_pool_proj_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_pool_proj_scale;
    inception_a2_pool_proj_scale = vxCreateVirtualTensor(graph,4, inception_a2_pool_proj_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_scale);
    vx_size inception_a2_pool_proj_bn_W_dims[1] = { 64 };
    vx_float32 inception_a2_pool_proj_bn_eps = 0.001;
    vx_tensor inception_a2_pool_proj_bn_W;
    inception_a2_pool_proj_bn_W = vxCreateTensor(context,1, inception_a2_pool_proj_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_pool_proj_bn_W, dataFolder + "/weights/inception_a2_pool_proj_bn.f32"));
    vx_size inception_a2_pool_proj_bn_B_dims[1] = { 64 };
    vx_tensor inception_a2_pool_proj_bn_B;
    inception_a2_pool_proj_bn_B = vxCreateTensor(context,1, inception_a2_pool_proj_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_pool_proj_bn_B, dataFolder + "/bias/inception_a2_pool_proj_bn.f32"));
    vx_size inception_a2_pool_proj_scale_W_dims[1] = { 64 };
    vx_tensor inception_a2_pool_proj_scale_W;
    inception_a2_pool_proj_scale_W = vxCreateTensor(context,1, inception_a2_pool_proj_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_pool_proj_scale_W, dataFolder + "/weights/inception_a2_pool_proj_scale.f32"));
    vx_size inception_a2_pool_proj_scale_B_dims[1] = { 64 };
    vx_tensor inception_a2_pool_proj_scale_B;
    inception_a2_pool_proj_scale_B = vxCreateTensor(context,1, inception_a2_pool_proj_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_pool_proj_scale_B, dataFolder + "/bias/inception_a2_pool_proj_scale.f32"));
    vx_node inception_a2_pool_proj_bn_node;
    inception_a2_pool_proj_bn_node = vxBatchNormalizationLayer(graph, inception_a2_pool_proj, inception_a2_pool_proj_bn_W, inception_a2_pool_proj_bn_B, inception_a2_pool_proj_scale_W, inception_a2_pool_proj_scale_B, inception_a2_pool_proj_bn_eps, inception_a2_pool_proj_scale);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_pool_proj_bn_node));

    // inception_a2_pool_proj_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_pool_proj_relu Layer
    vx_size inception_a2_pool_proj_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_pool_proj_relu;
    inception_a2_pool_proj_relu = vxCreateVirtualTensor(graph,4, inception_a2_pool_proj_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_relu);
    vx_enum inception_a2_pool_proj_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_pool_proj_relu_param_a = 0;
    vx_float32 inception_a2_pool_proj_relu_param_b = 0;
    vx_node inception_a2_pool_proj_relu_node;
    inception_a2_pool_proj_relu_node = vxActivationLayer(graph, inception_a2_pool_proj_scale, inception_a2_pool_proj_relu_mode, inception_a2_pool_proj_relu_param_a, inception_a2_pool_proj_relu_param_b, inception_a2_pool_proj_relu);
    ERROR_CHECK_OBJECT(inception_a2_pool_proj_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_pool_proj_relu_node));

    // inception_a2_output Layer
    vx_size inception_a2_output_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a2_output;
    inception_a2_output = vxCreateVirtualTensor(graph,4, inception_a2_output_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_output);
    vx_node inception_a2_output_node;
    inception_a2_output_node = vxConcatLayer(graph, inception_a2_output, inception_a2_1x1_relu, inception_a2_5x5_relu, inception_a2_3x3_2_relu, inception_a2_pool_proj_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a2_output_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_output_node));

    // inception_a2_output_inception_a2_output_0_split_0 Layer
    vx_size inception_a2_output_inception_a2_output_0_split_0_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a2_output_inception_a2_output_0_split_0;
    inception_a2_output_inception_a2_output_0_split_0 = vxCreateVirtualTensor(graph,4, inception_a2_output_inception_a2_output_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_0);
    vx_node inception_a2_output_inception_a2_output_0_split_0_node;
    inception_a2_output_inception_a2_output_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_a2_output, (vx_reference)inception_a2_output_inception_a2_output_0_split_0);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_output_inception_a2_output_0_split_0_node));

    // inception_a2_output_inception_a2_output_0_split_1 Layer
    vx_size inception_a2_output_inception_a2_output_0_split_1_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a2_output_inception_a2_output_0_split_1;
    inception_a2_output_inception_a2_output_0_split_1 = vxCreateVirtualTensor(graph,4, inception_a2_output_inception_a2_output_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_1);
    vx_node inception_a2_output_inception_a2_output_0_split_1_node;
    inception_a2_output_inception_a2_output_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_a2_output, (vx_reference)inception_a2_output_inception_a2_output_0_split_1);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_output_inception_a2_output_0_split_1_node));

    // inception_a2_output_inception_a2_output_0_split_2 Layer
    vx_size inception_a2_output_inception_a2_output_0_split_2_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a2_output_inception_a2_output_0_split_2;
    inception_a2_output_inception_a2_output_0_split_2 = vxCreateVirtualTensor(graph,4, inception_a2_output_inception_a2_output_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_2);
    vx_node inception_a2_output_inception_a2_output_0_split_2_node;
    inception_a2_output_inception_a2_output_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_a2_output, (vx_reference)inception_a2_output_inception_a2_output_0_split_2);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_output_inception_a2_output_0_split_2_node));

    // inception_a2_output_inception_a2_output_0_split_3 Layer
    vx_size inception_a2_output_inception_a2_output_0_split_3_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a2_output_inception_a2_output_0_split_3;
    inception_a2_output_inception_a2_output_0_split_3 = vxCreateVirtualTensor(graph,4, inception_a2_output_inception_a2_output_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_3);
    vx_node inception_a2_output_inception_a2_output_0_split_3_node;
    inception_a2_output_inception_a2_output_0_split_3_node = vxCopyNode( graph, (vx_reference)inception_a2_output, (vx_reference)inception_a2_output_inception_a2_output_0_split_3);
    ERROR_CHECK_OBJECT(inception_a2_output_inception_a2_output_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_output_inception_a2_output_0_split_3_node));

    // inception_a3_1x1 Layer
    vx_size inception_a3_1x1_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_1x1;
    inception_a3_1x1 = vxCreateVirtualTensor(graph,4, inception_a3_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1);
    vx_size inception_a3_1x1_W_dims[4] = { 1, 1, 288, 64 };
    vx_tensor inception_a3_1x1_W;
    inception_a3_1x1_W = vxCreateTensor(context,4, inception_a3_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_W, dataFolder + "/weights/inception_a3_1x1.f32"));
    vx_nn_convolution_params_t inception_a3_1x1_params;
    inception_a3_1x1_params.padding_x = 0;
    inception_a3_1x1_params.padding_y = 0;
    inception_a3_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_1x1_params.dilation_x = 0;
    inception_a3_1x1_params.dilation_y = 0;
    vx_node inception_a3_1x1_node;
    inception_a3_1x1_node = vxConvolutionLayer(graph, inception_a2_output_inception_a2_output_0_split_0, inception_a3_1x1_W, NULL, &inception_a3_1x1_params, sizeof(inception_a3_1x1_params ), inception_a3_1x1);
    ERROR_CHECK_OBJECT(inception_a3_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_node));

    // inception_a3_1x1_bn Layer
    vx_size inception_a3_1x1_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_1x1_scale;
    inception_a3_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a3_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_scale);
    vx_size inception_a3_1x1_bn_W_dims[1] = { 64 };
    vx_float32 inception_a3_1x1_bn_eps = 0.001;
    vx_tensor inception_a3_1x1_bn_W;
    inception_a3_1x1_bn_W = vxCreateTensor(context,1, inception_a3_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_bn_W, dataFolder + "/weights/inception_a3_1x1_bn.f32"));
    vx_size inception_a3_1x1_bn_B_dims[1] = { 64 };
    vx_tensor inception_a3_1x1_bn_B;
    inception_a3_1x1_bn_B = vxCreateTensor(context,1, inception_a3_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_bn_B, dataFolder + "/bias/inception_a3_1x1_bn.f32"));
    vx_size inception_a3_1x1_scale_W_dims[1] = { 64 };
    vx_tensor inception_a3_1x1_scale_W;
    inception_a3_1x1_scale_W = vxCreateTensor(context,1, inception_a3_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_scale_W, dataFolder + "/weights/inception_a3_1x1_scale.f32"));
    vx_size inception_a3_1x1_scale_B_dims[1] = { 64 };
    vx_tensor inception_a3_1x1_scale_B;
    inception_a3_1x1_scale_B = vxCreateTensor(context,1, inception_a3_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_scale_B, dataFolder + "/bias/inception_a3_1x1_scale.f32"));
    vx_node inception_a3_1x1_bn_node;
    inception_a3_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a3_1x1, inception_a3_1x1_bn_W, inception_a3_1x1_bn_B, inception_a3_1x1_scale_W, inception_a3_1x1_scale_B, inception_a3_1x1_bn_eps, inception_a3_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a3_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_bn_node));

    // inception_a3_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_1x1_relu Layer
    vx_size inception_a3_1x1_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_1x1_relu;
    inception_a3_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a3_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_relu);
    vx_enum inception_a3_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_1x1_relu_param_a = 0;
    vx_float32 inception_a3_1x1_relu_param_b = 0;
    vx_node inception_a3_1x1_relu_node;
    inception_a3_1x1_relu_node = vxActivationLayer(graph, inception_a3_1x1_scale, inception_a3_1x1_relu_mode, inception_a3_1x1_relu_param_a, inception_a3_1x1_relu_param_b, inception_a3_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a3_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_relu_node));

    // inception_a3_5x5_reduce Layer
    vx_size inception_a3_5x5_reduce_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a3_5x5_reduce;
    inception_a3_5x5_reduce = vxCreateVirtualTensor(graph,4, inception_a3_5x5_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce);
    vx_size inception_a3_5x5_reduce_W_dims[4] = { 1, 1, 288, 48 };
    vx_tensor inception_a3_5x5_reduce_W;
    inception_a3_5x5_reduce_W = vxCreateTensor(context,4, inception_a3_5x5_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_reduce_W, dataFolder + "/weights/inception_a3_5x5_reduce.f32"));
    vx_nn_convolution_params_t inception_a3_5x5_reduce_params;
    inception_a3_5x5_reduce_params.padding_x = 0;
    inception_a3_5x5_reduce_params.padding_y = 0;
    inception_a3_5x5_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_5x5_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_5x5_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_5x5_reduce_params.dilation_x = 0;
    inception_a3_5x5_reduce_params.dilation_y = 0;
    vx_node inception_a3_5x5_reduce_node;
    inception_a3_5x5_reduce_node = vxConvolutionLayer(graph, inception_a2_output_inception_a2_output_0_split_1, inception_a3_5x5_reduce_W, NULL, &inception_a3_5x5_reduce_params, sizeof(inception_a3_5x5_reduce_params ), inception_a3_5x5_reduce);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_5x5_reduce_node));

    // inception_a3_5x5_reduce_bn Layer
    vx_size inception_a3_5x5_reduce_scale_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a3_5x5_reduce_scale;
    inception_a3_5x5_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a3_5x5_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_scale);
    vx_size inception_a3_5x5_reduce_bn_W_dims[1] = { 48 };
    vx_float32 inception_a3_5x5_reduce_bn_eps = 0.001;
    vx_tensor inception_a3_5x5_reduce_bn_W;
    inception_a3_5x5_reduce_bn_W = vxCreateTensor(context,1, inception_a3_5x5_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_reduce_bn_W, dataFolder + "/weights/inception_a3_5x5_reduce_bn.f32"));
    vx_size inception_a3_5x5_reduce_bn_B_dims[1] = { 48 };
    vx_tensor inception_a3_5x5_reduce_bn_B;
    inception_a3_5x5_reduce_bn_B = vxCreateTensor(context,1, inception_a3_5x5_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_reduce_bn_B, dataFolder + "/bias/inception_a3_5x5_reduce_bn.f32"));
    vx_size inception_a3_5x5_reduce_scale_W_dims[1] = { 48 };
    vx_tensor inception_a3_5x5_reduce_scale_W;
    inception_a3_5x5_reduce_scale_W = vxCreateTensor(context,1, inception_a3_5x5_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_reduce_scale_W, dataFolder + "/weights/inception_a3_5x5_reduce_scale.f32"));
    vx_size inception_a3_5x5_reduce_scale_B_dims[1] = { 48 };
    vx_tensor inception_a3_5x5_reduce_scale_B;
    inception_a3_5x5_reduce_scale_B = vxCreateTensor(context,1, inception_a3_5x5_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_reduce_scale_B, dataFolder + "/bias/inception_a3_5x5_reduce_scale.f32"));
    vx_node inception_a3_5x5_reduce_bn_node;
    inception_a3_5x5_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a3_5x5_reduce, inception_a3_5x5_reduce_bn_W, inception_a3_5x5_reduce_bn_B, inception_a3_5x5_reduce_scale_W, inception_a3_5x5_reduce_scale_B, inception_a3_5x5_reduce_bn_eps, inception_a3_5x5_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_5x5_reduce_bn_node));

    // inception_a3_5x5_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_5x5_reduce_relu Layer
    vx_size inception_a3_5x5_reduce_relu_dims[4] = { 35, 35, 48, 1 };
    vx_tensor inception_a3_5x5_reduce_relu;
    inception_a3_5x5_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a3_5x5_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_relu);
    vx_enum inception_a3_5x5_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_5x5_reduce_relu_param_a = 0;
    vx_float32 inception_a3_5x5_reduce_relu_param_b = 0;
    vx_node inception_a3_5x5_reduce_relu_node;
    inception_a3_5x5_reduce_relu_node = vxActivationLayer(graph, inception_a3_5x5_reduce_scale, inception_a3_5x5_reduce_relu_mode, inception_a3_5x5_reduce_relu_param_a, inception_a3_5x5_reduce_relu_param_b, inception_a3_5x5_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a3_5x5_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_5x5_reduce_relu_node));

    // inception_a3_5x5 Layer
    vx_size inception_a3_5x5_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_5x5;
    inception_a3_5x5 = vxCreateVirtualTensor(graph,4, inception_a3_5x5_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_5x5);
    vx_size inception_a3_5x5_W_dims[4] = { 5, 5, 48, 64 };
    vx_tensor inception_a3_5x5_W;
    inception_a3_5x5_W = vxCreateTensor(context,4, inception_a3_5x5_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_W, dataFolder + "/weights/inception_a3_5x5.f32"));
    vx_nn_convolution_params_t inception_a3_5x5_params;
    inception_a3_5x5_params.padding_x = 2;
    inception_a3_5x5_params.padding_y = 2;
    inception_a3_5x5_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_5x5_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_5x5_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_5x5_params.dilation_x = 0;
    inception_a3_5x5_params.dilation_y = 0;
    vx_node inception_a3_5x5_node;
    inception_a3_5x5_node = vxConvolutionLayer(graph, inception_a3_5x5_reduce_relu, inception_a3_5x5_W, NULL, &inception_a3_5x5_params, sizeof(inception_a3_5x5_params ), inception_a3_5x5);
    ERROR_CHECK_OBJECT(inception_a3_5x5_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_5x5_node));

    // inception_a3_5x5_bn Layer
    vx_size inception_a3_5x5_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_5x5_scale;
    inception_a3_5x5_scale = vxCreateVirtualTensor(graph,4, inception_a3_5x5_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_scale);
    vx_size inception_a3_5x5_bn_W_dims[1] = { 64 };
    vx_float32 inception_a3_5x5_bn_eps = 0.001;
    vx_tensor inception_a3_5x5_bn_W;
    inception_a3_5x5_bn_W = vxCreateTensor(context,1, inception_a3_5x5_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_bn_W, dataFolder + "/weights/inception_a3_5x5_bn.f32"));
    vx_size inception_a3_5x5_bn_B_dims[1] = { 64 };
    vx_tensor inception_a3_5x5_bn_B;
    inception_a3_5x5_bn_B = vxCreateTensor(context,1, inception_a3_5x5_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_bn_B, dataFolder + "/bias/inception_a3_5x5_bn.f32"));
    vx_size inception_a3_5x5_scale_W_dims[1] = { 64 };
    vx_tensor inception_a3_5x5_scale_W;
    inception_a3_5x5_scale_W = vxCreateTensor(context,1, inception_a3_5x5_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_scale_W, dataFolder + "/weights/inception_a3_5x5_scale.f32"));
    vx_size inception_a3_5x5_scale_B_dims[1] = { 64 };
    vx_tensor inception_a3_5x5_scale_B;
    inception_a3_5x5_scale_B = vxCreateTensor(context,1, inception_a3_5x5_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_5x5_scale_B, dataFolder + "/bias/inception_a3_5x5_scale.f32"));
    vx_node inception_a3_5x5_bn_node;
    inception_a3_5x5_bn_node = vxBatchNormalizationLayer(graph, inception_a3_5x5, inception_a3_5x5_bn_W, inception_a3_5x5_bn_B, inception_a3_5x5_scale_W, inception_a3_5x5_scale_B, inception_a3_5x5_bn_eps, inception_a3_5x5_scale);
    ERROR_CHECK_OBJECT(inception_a3_5x5_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_5x5_bn_node));

    // inception_a3_5x5_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_5x5_relu Layer
    vx_size inception_a3_5x5_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_5x5_relu;
    inception_a3_5x5_relu = vxCreateVirtualTensor(graph,4, inception_a3_5x5_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_5x5_relu);
    vx_enum inception_a3_5x5_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_5x5_relu_param_a = 0;
    vx_float32 inception_a3_5x5_relu_param_b = 0;
    vx_node inception_a3_5x5_relu_node;
    inception_a3_5x5_relu_node = vxActivationLayer(graph, inception_a3_5x5_scale, inception_a3_5x5_relu_mode, inception_a3_5x5_relu_param_a, inception_a3_5x5_relu_param_b, inception_a3_5x5_relu);
    ERROR_CHECK_OBJECT(inception_a3_5x5_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_5x5_relu_node));

    // inception_a3_3x3_reduce Layer
    vx_size inception_a3_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_reduce;
    inception_a3_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a3_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce);
    vx_size inception_a3_3x3_reduce_W_dims[4] = { 1, 1, 288, 64 };
    vx_tensor inception_a3_3x3_reduce_W;
    inception_a3_3x3_reduce_W = vxCreateTensor(context,4, inception_a3_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_W, dataFolder + "/weights/inception_a3_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_reduce_params;
    inception_a3_3x3_reduce_params.padding_x = 0;
    inception_a3_3x3_reduce_params.padding_y = 0;
    inception_a3_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_reduce_params.dilation_x = 0;
    inception_a3_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a3_3x3_reduce_node;
    inception_a3_3x3_reduce_node = vxConvolutionLayer(graph, inception_a2_output_inception_a2_output_0_split_2, inception_a3_3x3_reduce_W, NULL, &inception_a3_3x3_reduce_params, sizeof(inception_a3_3x3_reduce_params ), inception_a3_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_reduce_node));

    // inception_a3_3x3_reduce_bn Layer
    vx_size inception_a3_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_reduce_scale;
    inception_a3_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_scale);
    vx_size inception_a3_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a3_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_reduce_bn_W;
    inception_a3_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a3_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_bn_W, dataFolder + "/weights/inception_a3_3x3_reduce_bn.f32"));
    vx_size inception_a3_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_reduce_bn_B;
    inception_a3_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a3_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_bn_B, dataFolder + "/bias/inception_a3_3x3_reduce_bn.f32"));
    vx_size inception_a3_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_reduce_scale_W;
    inception_a3_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a3_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_scale_W, dataFolder + "/weights/inception_a3_3x3_reduce_scale.f32"));
    vx_size inception_a3_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_reduce_scale_B;
    inception_a3_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a3_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_scale_B, dataFolder + "/bias/inception_a3_3x3_reduce_scale.f32"));
    vx_node inception_a3_3x3_reduce_bn_node;
    inception_a3_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_reduce, inception_a3_3x3_reduce_bn_W, inception_a3_3x3_reduce_bn_B, inception_a3_3x3_reduce_scale_W, inception_a3_3x3_reduce_scale_B, inception_a3_3x3_reduce_bn_eps, inception_a3_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_reduce_bn_node));

    // inception_a3_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_reduce_relu Layer
    vx_size inception_a3_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_reduce_relu;
    inception_a3_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_relu);
    vx_enum inception_a3_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a3_3x3_reduce_relu_param_b = 0;
    vx_node inception_a3_3x3_reduce_relu_node;
    inception_a3_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a3_3x3_reduce_scale, inception_a3_3x3_reduce_relu_mode, inception_a3_3x3_reduce_relu_param_a, inception_a3_3x3_reduce_relu_param_b, inception_a3_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_reduce_relu_node));

    // inception_a3_3x3_1 Layer
    vx_size inception_a3_3x3_1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_1;
    inception_a3_3x3_1 = vxCreateVirtualTensor(graph,4, inception_a3_3x3_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1);
    vx_size inception_a3_3x3_1_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a3_3x3_1_W;
    inception_a3_3x3_1_W = vxCreateTensor(context,4, inception_a3_3x3_1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_1_W, dataFolder + "/weights/inception_a3_3x3_1.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_1_params;
    inception_a3_3x3_1_params.padding_x = 1;
    inception_a3_3x3_1_params.padding_y = 1;
    inception_a3_3x3_1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_1_params.dilation_x = 0;
    inception_a3_3x3_1_params.dilation_y = 0;
    vx_node inception_a3_3x3_1_node;
    inception_a3_3x3_1_node = vxConvolutionLayer(graph, inception_a3_3x3_reduce_relu, inception_a3_3x3_1_W, NULL, &inception_a3_3x3_1_params, sizeof(inception_a3_3x3_1_params ), inception_a3_3x3_1);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_1_node));

    // inception_a3_3x3_1_bn Layer
    vx_size inception_a3_3x3_1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_1_scale;
    inception_a3_3x3_1_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_scale);
    vx_size inception_a3_3x3_1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_3x3_1_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_1_bn_W;
    inception_a3_3x3_1_bn_W = vxCreateTensor(context,1, inception_a3_3x3_1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_1_bn_W, dataFolder + "/weights/inception_a3_3x3_1_bn.f32"));
    vx_size inception_a3_3x3_1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_1_bn_B;
    inception_a3_3x3_1_bn_B = vxCreateTensor(context,1, inception_a3_3x3_1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_1_bn_B, dataFolder + "/bias/inception_a3_3x3_1_bn.f32"));
    vx_size inception_a3_3x3_1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_1_scale_W;
    inception_a3_3x3_1_scale_W = vxCreateTensor(context,1, inception_a3_3x3_1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_1_scale_W, dataFolder + "/weights/inception_a3_3x3_1_scale.f32"));
    vx_size inception_a3_3x3_1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_1_scale_B;
    inception_a3_3x3_1_scale_B = vxCreateTensor(context,1, inception_a3_3x3_1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_1_scale_B, dataFolder + "/bias/inception_a3_3x3_1_scale.f32"));
    vx_node inception_a3_3x3_1_bn_node;
    inception_a3_3x3_1_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_1, inception_a3_3x3_1_bn_W, inception_a3_3x3_1_bn_B, inception_a3_3x3_1_scale_W, inception_a3_3x3_1_scale_B, inception_a3_3x3_1_bn_eps, inception_a3_3x3_1_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_1_bn_node));

    // inception_a3_3x3_1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_1_relu Layer
    vx_size inception_a3_3x3_1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_1_relu;
    inception_a3_3x3_1_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_relu);
    vx_enum inception_a3_3x3_1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_1_relu_param_a = 0;
    vx_float32 inception_a3_3x3_1_relu_param_b = 0;
    vx_node inception_a3_3x3_1_relu_node;
    inception_a3_3x3_1_relu_node = vxActivationLayer(graph, inception_a3_3x3_1_scale, inception_a3_3x3_1_relu_mode, inception_a3_3x3_1_relu_param_a, inception_a3_3x3_1_relu_param_b, inception_a3_3x3_1_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_1_relu_node));

    // inception_a3_3x3_2 Layer
    vx_size inception_a3_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_2;
    inception_a3_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2);
    vx_size inception_a3_3x3_2_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a3_3x3_2_W;
    inception_a3_3x3_2_W = vxCreateTensor(context,4, inception_a3_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_W, dataFolder + "/weights/inception_a3_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_2_params;
    inception_a3_3x3_2_params.padding_x = 1;
    inception_a3_3x3_2_params.padding_y = 1;
    inception_a3_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_2_params.dilation_x = 0;
    inception_a3_3x3_2_params.dilation_y = 0;
    vx_node inception_a3_3x3_2_node;
    inception_a3_3x3_2_node = vxConvolutionLayer(graph, inception_a3_3x3_1_relu, inception_a3_3x3_2_W, NULL, &inception_a3_3x3_2_params, sizeof(inception_a3_3x3_2_params ), inception_a3_3x3_2);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_node));

    // inception_a3_3x3_2_bn Layer
    vx_size inception_a3_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_2_scale;
    inception_a3_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_scale);
    vx_size inception_a3_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_2_bn_W;
    inception_a3_3x3_2_bn_W = vxCreateTensor(context,1, inception_a3_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_bn_W, dataFolder + "/weights/inception_a3_3x3_2_bn.f32"));
    vx_size inception_a3_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_2_bn_B;
    inception_a3_3x3_2_bn_B = vxCreateTensor(context,1, inception_a3_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_bn_B, dataFolder + "/bias/inception_a3_3x3_2_bn.f32"));
    vx_size inception_a3_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_2_scale_W;
    inception_a3_3x3_2_scale_W = vxCreateTensor(context,1, inception_a3_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_scale_W, dataFolder + "/weights/inception_a3_3x3_2_scale.f32"));
    vx_size inception_a3_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_2_scale_B;
    inception_a3_3x3_2_scale_B = vxCreateTensor(context,1, inception_a3_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_scale_B, dataFolder + "/bias/inception_a3_3x3_2_scale.f32"));
    vx_node inception_a3_3x3_2_bn_node;
    inception_a3_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_2, inception_a3_3x3_2_bn_W, inception_a3_3x3_2_bn_B, inception_a3_3x3_2_scale_W, inception_a3_3x3_2_scale_B, inception_a3_3x3_2_bn_eps, inception_a3_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_bn_node));

    // inception_a3_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_2_relu Layer
    vx_size inception_a3_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_2_relu;
    inception_a3_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_relu);
    vx_enum inception_a3_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_2_relu_param_a = 0;
    vx_float32 inception_a3_3x3_2_relu_param_b = 0;
    vx_node inception_a3_3x3_2_relu_node;
    inception_a3_3x3_2_relu_node = vxActivationLayer(graph, inception_a3_3x3_2_scale, inception_a3_3x3_2_relu_mode, inception_a3_3x3_2_relu_param_a, inception_a3_3x3_2_relu_param_b, inception_a3_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_relu_node));

    // inception_a3_pool Layer
    vx_size inception_a3_pool_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a3_pool;
    inception_a3_pool = vxCreateVirtualTensor(graph,4, inception_a3_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_pool);
    vx_enum inception_a3_pool_type = VX_NN_POOLING_AVG;
    vx_size inception_a3_pool_kernel_w = 3;
    vx_size inception_a3_pool_kernel_h = 3;
    vx_size inception_a3_pool_pad_w = 1;
    vx_size inception_a3_pool_pad_h = 1;
    vx_enum inception_a3_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a3_pool_node;
    inception_a3_pool_node = vxPoolingLayer(graph, inception_a2_output_inception_a2_output_0_split_3, inception_a3_pool_type, inception_a3_pool_kernel_w, inception_a3_pool_kernel_h, inception_a3_pool_pad_w, inception_a3_pool_pad_h, inception_a3_pool_roundPolicy, inception_a3_pool );
    ERROR_CHECK_OBJECT(inception_a3_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_pool_node));

    // inception_a3_pool_proj Layer
    vx_size inception_a3_pool_proj_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_pool_proj;
    inception_a3_pool_proj = vxCreateVirtualTensor(graph,4, inception_a3_pool_proj_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj);
    vx_size inception_a3_pool_proj_W_dims[4] = { 1, 1, 288, 64 };
    vx_tensor inception_a3_pool_proj_W;
    inception_a3_pool_proj_W = vxCreateTensor(context,4, inception_a3_pool_proj_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_pool_proj_W, dataFolder + "/weights/inception_a3_pool_proj.f32"));
    vx_nn_convolution_params_t inception_a3_pool_proj_params;
    inception_a3_pool_proj_params.padding_x = 0;
    inception_a3_pool_proj_params.padding_y = 0;
    inception_a3_pool_proj_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_pool_proj_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_pool_proj_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_pool_proj_params.dilation_x = 0;
    inception_a3_pool_proj_params.dilation_y = 0;
    vx_node inception_a3_pool_proj_node;
    inception_a3_pool_proj_node = vxConvolutionLayer(graph, inception_a3_pool, inception_a3_pool_proj_W, NULL, &inception_a3_pool_proj_params, sizeof(inception_a3_pool_proj_params ), inception_a3_pool_proj);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_pool_proj_node));

    // inception_a3_pool_proj_bn Layer
    vx_size inception_a3_pool_proj_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_pool_proj_scale;
    inception_a3_pool_proj_scale = vxCreateVirtualTensor(graph,4, inception_a3_pool_proj_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_scale);
    vx_size inception_a3_pool_proj_bn_W_dims[1] = { 64 };
    vx_float32 inception_a3_pool_proj_bn_eps = 0.001;
    vx_tensor inception_a3_pool_proj_bn_W;
    inception_a3_pool_proj_bn_W = vxCreateTensor(context,1, inception_a3_pool_proj_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_pool_proj_bn_W, dataFolder + "/weights/inception_a3_pool_proj_bn.f32"));
    vx_size inception_a3_pool_proj_bn_B_dims[1] = { 64 };
    vx_tensor inception_a3_pool_proj_bn_B;
    inception_a3_pool_proj_bn_B = vxCreateTensor(context,1, inception_a3_pool_proj_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_pool_proj_bn_B, dataFolder + "/bias/inception_a3_pool_proj_bn.f32"));
    vx_size inception_a3_pool_proj_scale_W_dims[1] = { 64 };
    vx_tensor inception_a3_pool_proj_scale_W;
    inception_a3_pool_proj_scale_W = vxCreateTensor(context,1, inception_a3_pool_proj_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_pool_proj_scale_W, dataFolder + "/weights/inception_a3_pool_proj_scale.f32"));
    vx_size inception_a3_pool_proj_scale_B_dims[1] = { 64 };
    vx_tensor inception_a3_pool_proj_scale_B;
    inception_a3_pool_proj_scale_B = vxCreateTensor(context,1, inception_a3_pool_proj_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_pool_proj_scale_B, dataFolder + "/bias/inception_a3_pool_proj_scale.f32"));
    vx_node inception_a3_pool_proj_bn_node;
    inception_a3_pool_proj_bn_node = vxBatchNormalizationLayer(graph, inception_a3_pool_proj, inception_a3_pool_proj_bn_W, inception_a3_pool_proj_bn_B, inception_a3_pool_proj_scale_W, inception_a3_pool_proj_scale_B, inception_a3_pool_proj_bn_eps, inception_a3_pool_proj_scale);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_pool_proj_bn_node));

    // inception_a3_pool_proj_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_pool_proj_relu Layer
    vx_size inception_a3_pool_proj_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_pool_proj_relu;
    inception_a3_pool_proj_relu = vxCreateVirtualTensor(graph,4, inception_a3_pool_proj_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_relu);
    vx_enum inception_a3_pool_proj_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_pool_proj_relu_param_a = 0;
    vx_float32 inception_a3_pool_proj_relu_param_b = 0;
    vx_node inception_a3_pool_proj_relu_node;
    inception_a3_pool_proj_relu_node = vxActivationLayer(graph, inception_a3_pool_proj_scale, inception_a3_pool_proj_relu_mode, inception_a3_pool_proj_relu_param_a, inception_a3_pool_proj_relu_param_b, inception_a3_pool_proj_relu);
    ERROR_CHECK_OBJECT(inception_a3_pool_proj_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_pool_proj_relu_node));

    // inception_a3_output Layer
    vx_size inception_a3_output_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a3_output;
    inception_a3_output = vxCreateVirtualTensor(graph,4, inception_a3_output_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_output);
    vx_node inception_a3_output_node;
    inception_a3_output_node = vxConcatLayer(graph, inception_a3_output, inception_a3_1x1_relu, inception_a3_5x5_relu, inception_a3_3x3_2_relu, inception_a3_pool_proj_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a3_output_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_output_node));

    // inception_a3_output_inception_a3_output_0_split_0 Layer
    vx_size inception_a3_output_inception_a3_output_0_split_0_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a3_output_inception_a3_output_0_split_0;
    inception_a3_output_inception_a3_output_0_split_0 = vxCreateVirtualTensor(graph,4, inception_a3_output_inception_a3_output_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_output_inception_a3_output_0_split_0);
    vx_node inception_a3_output_inception_a3_output_0_split_0_node;
    inception_a3_output_inception_a3_output_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_a3_output, (vx_reference)inception_a3_output_inception_a3_output_0_split_0);
    ERROR_CHECK_OBJECT(inception_a3_output_inception_a3_output_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_output_inception_a3_output_0_split_0_node));

    // inception_a3_output_inception_a3_output_0_split_1 Layer
    vx_size inception_a3_output_inception_a3_output_0_split_1_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a3_output_inception_a3_output_0_split_1;
    inception_a3_output_inception_a3_output_0_split_1 = vxCreateVirtualTensor(graph,4, inception_a3_output_inception_a3_output_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_output_inception_a3_output_0_split_1);
    vx_node inception_a3_output_inception_a3_output_0_split_1_node;
    inception_a3_output_inception_a3_output_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_a3_output, (vx_reference)inception_a3_output_inception_a3_output_0_split_1);
    ERROR_CHECK_OBJECT(inception_a3_output_inception_a3_output_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_output_inception_a3_output_0_split_1_node));

    // inception_a3_output_inception_a3_output_0_split_2 Layer
    vx_size inception_a3_output_inception_a3_output_0_split_2_dims[4] = { 35, 35, 288, 1 };
    vx_tensor inception_a3_output_inception_a3_output_0_split_2;
    inception_a3_output_inception_a3_output_0_split_2 = vxCreateVirtualTensor(graph,4, inception_a3_output_inception_a3_output_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_output_inception_a3_output_0_split_2);
    vx_node inception_a3_output_inception_a3_output_0_split_2_node;
    inception_a3_output_inception_a3_output_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_a3_output, (vx_reference)inception_a3_output_inception_a3_output_0_split_2);
    ERROR_CHECK_OBJECT(inception_a3_output_inception_a3_output_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_output_inception_a3_output_0_split_2_node));

    // reduction_a_3x3 Layer
    vx_size reduction_a_3x3_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_3x3;
    reduction_a_3x3 = vxCreateVirtualTensor(graph,4, reduction_a_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3);
    vx_size reduction_a_3x3_W_dims[4] = { 3, 3, 288, 384 };
    vx_tensor reduction_a_3x3_W;
    reduction_a_3x3_W = vxCreateTensor(context,4, reduction_a_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_W, dataFolder + "/weights/reduction_a_3x3.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_params;
    reduction_a_3x3_params.padding_x = 0;
    reduction_a_3x3_params.padding_y = 0;
    reduction_a_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_params.dilation_x = 0;
    reduction_a_3x3_params.dilation_y = 0;
    vx_node reduction_a_3x3_node;
    reduction_a_3x3_node = vxConvolutionLayer(graph, inception_a3_output_inception_a3_output_0_split_0, reduction_a_3x3_W, NULL, &reduction_a_3x3_params, sizeof(reduction_a_3x3_params ), reduction_a_3x3);
    ERROR_CHECK_OBJECT(reduction_a_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_node));

    // reduction_a_3x3_bn Layer
    vx_size reduction_a_3x3_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_3x3_scale;
    reduction_a_3x3_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_scale);
    vx_size reduction_a_3x3_bn_W_dims[1] = { 384 };
    vx_float32 reduction_a_3x3_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_bn_W;
    reduction_a_3x3_bn_W = vxCreateTensor(context,1, reduction_a_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_bn_W, dataFolder + "/weights/reduction_a_3x3_bn.f32"));
    vx_size reduction_a_3x3_bn_B_dims[1] = { 384 };
    vx_tensor reduction_a_3x3_bn_B;
    reduction_a_3x3_bn_B = vxCreateTensor(context,1, reduction_a_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_bn_B, dataFolder + "/bias/reduction_a_3x3_bn.f32"));
    vx_size reduction_a_3x3_scale_W_dims[1] = { 384 };
    vx_tensor reduction_a_3x3_scale_W;
    reduction_a_3x3_scale_W = vxCreateTensor(context,1, reduction_a_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_scale_W, dataFolder + "/weights/reduction_a_3x3_scale.f32"));
    vx_size reduction_a_3x3_scale_B_dims[1] = { 384 };
    vx_tensor reduction_a_3x3_scale_B;
    reduction_a_3x3_scale_B = vxCreateTensor(context,1, reduction_a_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_scale_B, dataFolder + "/bias/reduction_a_3x3_scale.f32"));
    vx_node reduction_a_3x3_bn_node;
    reduction_a_3x3_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3, reduction_a_3x3_bn_W, reduction_a_3x3_bn_B, reduction_a_3x3_scale_W, reduction_a_3x3_scale_B, reduction_a_3x3_bn_eps, reduction_a_3x3_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_bn_node));

    // reduction_a_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_relu Layer
    vx_size reduction_a_3x3_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_3x3_relu;
    reduction_a_3x3_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_relu);
    vx_enum reduction_a_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_relu_param_a = 0;
    vx_float32 reduction_a_3x3_relu_param_b = 0;
    vx_node reduction_a_3x3_relu_node;
    reduction_a_3x3_relu_node = vxActivationLayer(graph, reduction_a_3x3_scale, reduction_a_3x3_relu_mode, reduction_a_3x3_relu_param_a, reduction_a_3x3_relu_param_b, reduction_a_3x3_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_relu_node));

    // reduction_a_3x3_2_reduce Layer
    vx_size reduction_a_3x3_2_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor reduction_a_3x3_2_reduce;
    reduction_a_3x3_2_reduce = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce);
    vx_size reduction_a_3x3_2_reduce_W_dims[4] = { 1, 1, 288, 64 };
    vx_tensor reduction_a_3x3_2_reduce_W;
    reduction_a_3x3_2_reduce_W = vxCreateTensor(context,4, reduction_a_3x3_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_W, dataFolder + "/weights/reduction_a_3x3_2_reduce.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_2_reduce_params;
    reduction_a_3x3_2_reduce_params.padding_x = 0;
    reduction_a_3x3_2_reduce_params.padding_y = 0;
    reduction_a_3x3_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_2_reduce_params.dilation_x = 0;
    reduction_a_3x3_2_reduce_params.dilation_y = 0;
    vx_node reduction_a_3x3_2_reduce_node;
    reduction_a_3x3_2_reduce_node = vxConvolutionLayer(graph, inception_a3_output_inception_a3_output_0_split_1, reduction_a_3x3_2_reduce_W, NULL, &reduction_a_3x3_2_reduce_params, sizeof(reduction_a_3x3_2_reduce_params ), reduction_a_3x3_2_reduce);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_reduce_node));

    // reduction_a_3x3_2_reduce_bn Layer
    vx_size reduction_a_3x3_2_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor reduction_a_3x3_2_reduce_scale;
    reduction_a_3x3_2_reduce_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_scale);
    vx_size reduction_a_3x3_2_reduce_bn_W_dims[1] = { 64 };
    vx_float32 reduction_a_3x3_2_reduce_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_2_reduce_bn_W;
    reduction_a_3x3_2_reduce_bn_W = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_bn_W, dataFolder + "/weights/reduction_a_3x3_2_reduce_bn.f32"));
    vx_size reduction_a_3x3_2_reduce_bn_B_dims[1] = { 64 };
    vx_tensor reduction_a_3x3_2_reduce_bn_B;
    reduction_a_3x3_2_reduce_bn_B = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_bn_B, dataFolder + "/bias/reduction_a_3x3_2_reduce_bn.f32"));
    vx_size reduction_a_3x3_2_reduce_scale_W_dims[1] = { 64 };
    vx_tensor reduction_a_3x3_2_reduce_scale_W;
    reduction_a_3x3_2_reduce_scale_W = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_scale_W, dataFolder + "/weights/reduction_a_3x3_2_reduce_scale.f32"));
    vx_size reduction_a_3x3_2_reduce_scale_B_dims[1] = { 64 };
    vx_tensor reduction_a_3x3_2_reduce_scale_B;
    reduction_a_3x3_2_reduce_scale_B = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_scale_B, dataFolder + "/bias/reduction_a_3x3_2_reduce_scale.f32"));
    vx_node reduction_a_3x3_2_reduce_bn_node;
    reduction_a_3x3_2_reduce_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3_2_reduce, reduction_a_3x3_2_reduce_bn_W, reduction_a_3x3_2_reduce_bn_B, reduction_a_3x3_2_reduce_scale_W, reduction_a_3x3_2_reduce_scale_B, reduction_a_3x3_2_reduce_bn_eps, reduction_a_3x3_2_reduce_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_reduce_bn_node));

    // reduction_a_3x3_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_2_reduce_relu Layer
    vx_size reduction_a_3x3_2_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor reduction_a_3x3_2_reduce_relu;
    reduction_a_3x3_2_reduce_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_relu);
    vx_enum reduction_a_3x3_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_2_reduce_relu_param_a = 0;
    vx_float32 reduction_a_3x3_2_reduce_relu_param_b = 0;
    vx_node reduction_a_3x3_2_reduce_relu_node;
    reduction_a_3x3_2_reduce_relu_node = vxActivationLayer(graph, reduction_a_3x3_2_reduce_scale, reduction_a_3x3_2_reduce_relu_mode, reduction_a_3x3_2_reduce_relu_param_a, reduction_a_3x3_2_reduce_relu_param_b, reduction_a_3x3_2_reduce_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_reduce_relu_node));

    // reduction_a_3x3_2 Layer
    vx_size reduction_a_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor reduction_a_3x3_2;
    reduction_a_3x3_2 = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2);
    vx_size reduction_a_3x3_2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor reduction_a_3x3_2_W;
    reduction_a_3x3_2_W = vxCreateTensor(context,4, reduction_a_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_W, dataFolder + "/weights/reduction_a_3x3_2.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_2_params;
    reduction_a_3x3_2_params.padding_x = 1;
    reduction_a_3x3_2_params.padding_y = 1;
    reduction_a_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_2_params.dilation_x = 0;
    reduction_a_3x3_2_params.dilation_y = 0;
    vx_node reduction_a_3x3_2_node;
    reduction_a_3x3_2_node = vxConvolutionLayer(graph, reduction_a_3x3_2_reduce_relu, reduction_a_3x3_2_W, NULL, &reduction_a_3x3_2_params, sizeof(reduction_a_3x3_2_params ), reduction_a_3x3_2);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_node));

    // reduction_a_3x3_2_bn Layer
    vx_size reduction_a_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor reduction_a_3x3_2_scale;
    reduction_a_3x3_2_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_scale);
    vx_size reduction_a_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 reduction_a_3x3_2_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_2_bn_W;
    reduction_a_3x3_2_bn_W = vxCreateTensor(context,1, reduction_a_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_bn_W, dataFolder + "/weights/reduction_a_3x3_2_bn.f32"));
    vx_size reduction_a_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor reduction_a_3x3_2_bn_B;
    reduction_a_3x3_2_bn_B = vxCreateTensor(context,1, reduction_a_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_bn_B, dataFolder + "/bias/reduction_a_3x3_2_bn.f32"));
    vx_size reduction_a_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor reduction_a_3x3_2_scale_W;
    reduction_a_3x3_2_scale_W = vxCreateTensor(context,1, reduction_a_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_scale_W, dataFolder + "/weights/reduction_a_3x3_2_scale.f32"));
    vx_size reduction_a_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor reduction_a_3x3_2_scale_B;
    reduction_a_3x3_2_scale_B = vxCreateTensor(context,1, reduction_a_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_scale_B, dataFolder + "/bias/reduction_a_3x3_2_scale.f32"));
    vx_node reduction_a_3x3_2_bn_node;
    reduction_a_3x3_2_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3_2, reduction_a_3x3_2_bn_W, reduction_a_3x3_2_bn_B, reduction_a_3x3_2_scale_W, reduction_a_3x3_2_scale_B, reduction_a_3x3_2_bn_eps, reduction_a_3x3_2_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_bn_node));

    // reduction_a_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_2_relu Layer
    vx_size reduction_a_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor reduction_a_3x3_2_relu;
    reduction_a_3x3_2_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_relu);
    vx_enum reduction_a_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_2_relu_param_a = 0;
    vx_float32 reduction_a_3x3_2_relu_param_b = 0;
    vx_node reduction_a_3x3_2_relu_node;
    reduction_a_3x3_2_relu_node = vxActivationLayer(graph, reduction_a_3x3_2_scale, reduction_a_3x3_2_relu_mode, reduction_a_3x3_2_relu_param_a, reduction_a_3x3_2_relu_param_b, reduction_a_3x3_2_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_relu_node));

    // reduction_a_3x3_3 Layer
    vx_size reduction_a_3x3_3_dims[4] = { 17, 17, 96, 1 };
    vx_tensor reduction_a_3x3_3;
    reduction_a_3x3_3 = vxCreateVirtualTensor(graph,4, reduction_a_3x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3);
    vx_size reduction_a_3x3_3_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor reduction_a_3x3_3_W;
    reduction_a_3x3_3_W = vxCreateTensor(context,4, reduction_a_3x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_W, dataFolder + "/weights/reduction_a_3x3_3.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_3_params;
    reduction_a_3x3_3_params.padding_x = 0;
    reduction_a_3x3_3_params.padding_y = 0;
    reduction_a_3x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_3_params.dilation_x = 0;
    reduction_a_3x3_3_params.dilation_y = 0;
    vx_node reduction_a_3x3_3_node;
    reduction_a_3x3_3_node = vxConvolutionLayer(graph, reduction_a_3x3_2_relu, reduction_a_3x3_3_W, NULL, &reduction_a_3x3_3_params, sizeof(reduction_a_3x3_3_params ), reduction_a_3x3_3);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_3_node));

    // reduction_a_3x3_3_bn Layer
    vx_size reduction_a_3x3_3_scale_dims[4] = { 17, 17, 96, 1 };
    vx_tensor reduction_a_3x3_3_scale;
    reduction_a_3x3_3_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_scale);
    vx_size reduction_a_3x3_3_bn_W_dims[1] = { 96 };
    vx_float32 reduction_a_3x3_3_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_3_bn_W;
    reduction_a_3x3_3_bn_W = vxCreateTensor(context,1, reduction_a_3x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_bn_W, dataFolder + "/weights/reduction_a_3x3_3_bn.f32"));
    vx_size reduction_a_3x3_3_bn_B_dims[1] = { 96 };
    vx_tensor reduction_a_3x3_3_bn_B;
    reduction_a_3x3_3_bn_B = vxCreateTensor(context,1, reduction_a_3x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_bn_B, dataFolder + "/bias/reduction_a_3x3_3_bn.f32"));
    vx_size reduction_a_3x3_3_scale_W_dims[1] = { 96 };
    vx_tensor reduction_a_3x3_3_scale_W;
    reduction_a_3x3_3_scale_W = vxCreateTensor(context,1, reduction_a_3x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_scale_W, dataFolder + "/weights/reduction_a_3x3_3_scale.f32"));
    vx_size reduction_a_3x3_3_scale_B_dims[1] = { 96 };
    vx_tensor reduction_a_3x3_3_scale_B;
    reduction_a_3x3_3_scale_B = vxCreateTensor(context,1, reduction_a_3x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_scale_B, dataFolder + "/bias/reduction_a_3x3_3_scale.f32"));
    vx_node reduction_a_3x3_3_bn_node;
    reduction_a_3x3_3_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3_3, reduction_a_3x3_3_bn_W, reduction_a_3x3_3_bn_B, reduction_a_3x3_3_scale_W, reduction_a_3x3_3_scale_B, reduction_a_3x3_3_bn_eps, reduction_a_3x3_3_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_3_bn_node));

    // reduction_a_3x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_3_relu Layer
    vx_size reduction_a_3x3_3_relu_dims[4] = { 17, 17, 96, 1 };
    vx_tensor reduction_a_3x3_3_relu;
    reduction_a_3x3_3_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_relu);
    vx_enum reduction_a_3x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_3_relu_param_a = 0;
    vx_float32 reduction_a_3x3_3_relu_param_b = 0;
    vx_node reduction_a_3x3_3_relu_node;
    reduction_a_3x3_3_relu_node = vxActivationLayer(graph, reduction_a_3x3_3_scale, reduction_a_3x3_3_relu_mode, reduction_a_3x3_3_relu_param_a, reduction_a_3x3_3_relu_param_b, reduction_a_3x3_3_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_3_relu_node));

    // reduction_a_pool Layer
    vx_size reduction_a_pool_dims[4] = { 17, 17, 288, 1 };
    vx_tensor reduction_a_pool;
    reduction_a_pool = vxCreateVirtualTensor(graph,4, reduction_a_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_pool);
    vx_enum reduction_a_pool_type = VX_NN_POOLING_MAX;
    vx_size reduction_a_pool_kernel_w = 3;
    vx_size reduction_a_pool_kernel_h = 3;
    vx_size reduction_a_pool_pad_w = 0;
    vx_size reduction_a_pool_pad_h = 0;
    vx_enum reduction_a_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node reduction_a_pool_node;
    reduction_a_pool_node = vxPoolingLayer(graph, inception_a3_output_inception_a3_output_0_split_2, reduction_a_pool_type, reduction_a_pool_kernel_w, reduction_a_pool_kernel_h, reduction_a_pool_pad_w, reduction_a_pool_pad_h, reduction_a_pool_roundPolicy, reduction_a_pool );
    ERROR_CHECK_OBJECT(reduction_a_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_pool_node));

    // reduction_a_concat Layer
    vx_size reduction_a_concat_dims[4] = { 17, 17, 768, 1 };
    vx_tensor reduction_a_concat;
    reduction_a_concat = vxCreateVirtualTensor(graph,4, reduction_a_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_concat);
    vx_node reduction_a_concat_node;
    reduction_a_concat_node = vxConcatLayer(graph, reduction_a_concat, reduction_a_3x3_relu, reduction_a_3x3_3_relu, reduction_a_pool, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(reduction_a_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_concat_node));

    // reduction_a_concat_reduction_a_concat_0_split_0 Layer
    vx_size reduction_a_concat_reduction_a_concat_0_split_0_dims[4] = { 17, 17, 768, 1 };
    vx_tensor reduction_a_concat_reduction_a_concat_0_split_0;
    reduction_a_concat_reduction_a_concat_0_split_0 = vxCreateVirtualTensor(graph,4, reduction_a_concat_reduction_a_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_0);
    vx_node reduction_a_concat_reduction_a_concat_0_split_0_node;
    reduction_a_concat_reduction_a_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)reduction_a_concat, (vx_reference)reduction_a_concat_reduction_a_concat_0_split_0);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_concat_reduction_a_concat_0_split_0_node));

    // reduction_a_concat_reduction_a_concat_0_split_1 Layer
    vx_size reduction_a_concat_reduction_a_concat_0_split_1_dims[4] = { 17, 17, 768, 1 };
    vx_tensor reduction_a_concat_reduction_a_concat_0_split_1;
    reduction_a_concat_reduction_a_concat_0_split_1 = vxCreateVirtualTensor(graph,4, reduction_a_concat_reduction_a_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_1);
    vx_node reduction_a_concat_reduction_a_concat_0_split_1_node;
    reduction_a_concat_reduction_a_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)reduction_a_concat, (vx_reference)reduction_a_concat_reduction_a_concat_0_split_1);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_concat_reduction_a_concat_0_split_1_node));

    // reduction_a_concat_reduction_a_concat_0_split_2 Layer
    vx_size reduction_a_concat_reduction_a_concat_0_split_2_dims[4] = { 17, 17, 768, 1 };
    vx_tensor reduction_a_concat_reduction_a_concat_0_split_2;
    reduction_a_concat_reduction_a_concat_0_split_2 = vxCreateVirtualTensor(graph,4, reduction_a_concat_reduction_a_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_2);
    vx_node reduction_a_concat_reduction_a_concat_0_split_2_node;
    reduction_a_concat_reduction_a_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)reduction_a_concat, (vx_reference)reduction_a_concat_reduction_a_concat_0_split_2);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_concat_reduction_a_concat_0_split_2_node));

    // reduction_a_concat_reduction_a_concat_0_split_3 Layer
    vx_size reduction_a_concat_reduction_a_concat_0_split_3_dims[4] = { 17, 17, 768, 1 };
    vx_tensor reduction_a_concat_reduction_a_concat_0_split_3;
    reduction_a_concat_reduction_a_concat_0_split_3 = vxCreateVirtualTensor(graph,4, reduction_a_concat_reduction_a_concat_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_3);
    vx_node reduction_a_concat_reduction_a_concat_0_split_3_node;
    reduction_a_concat_reduction_a_concat_0_split_3_node = vxCopyNode( graph, (vx_reference)reduction_a_concat, (vx_reference)reduction_a_concat_reduction_a_concat_0_split_3);
    ERROR_CHECK_OBJECT(reduction_a_concat_reduction_a_concat_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_concat_reduction_a_concat_0_split_3_node));

    // inception_b1_1x1_2 Layer
    vx_size inception_b1_1x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x1_2;
    inception_b1_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b1_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2);
    vx_size inception_b1_1x1_2_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b1_1x1_2_W;
    inception_b1_1x1_2_W = vxCreateTensor(context,4, inception_b1_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_W, dataFolder + "/weights/inception_b1_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b1_1x1_2_params;
    inception_b1_1x1_2_params.padding_x = 0;
    inception_b1_1x1_2_params.padding_y = 0;
    inception_b1_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x1_2_params.dilation_x = 0;
    inception_b1_1x1_2_params.dilation_y = 0;
    vx_node inception_b1_1x1_2_node;
    inception_b1_1x1_2_node = vxConvolutionLayer(graph, reduction_a_concat_reduction_a_concat_0_split_0, inception_b1_1x1_2_W, NULL, &inception_b1_1x1_2_params, sizeof(inception_b1_1x1_2_params ), inception_b1_1x1_2);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_2_node));

    // inception_b1_1x1_2_bn Layer
    vx_size inception_b1_1x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x1_2_scale;
    inception_b1_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_scale);
    vx_size inception_b1_1x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b1_1x1_2_bn_W;
    inception_b1_1x1_2_bn_W = vxCreateTensor(context,1, inception_b1_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_bn_W, dataFolder + "/weights/inception_b1_1x1_2_bn.f32"));
    vx_size inception_b1_1x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x1_2_bn_B;
    inception_b1_1x1_2_bn_B = vxCreateTensor(context,1, inception_b1_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_bn_B, dataFolder + "/bias/inception_b1_1x1_2_bn.f32"));
    vx_size inception_b1_1x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_1x1_2_scale_W;
    inception_b1_1x1_2_scale_W = vxCreateTensor(context,1, inception_b1_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_scale_W, dataFolder + "/weights/inception_b1_1x1_2_scale.f32"));
    vx_size inception_b1_1x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x1_2_scale_B;
    inception_b1_1x1_2_scale_B = vxCreateTensor(context,1, inception_b1_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_scale_B, dataFolder + "/bias/inception_b1_1x1_2_scale.f32"));
    vx_node inception_b1_1x1_2_bn_node;
    inception_b1_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x1_2, inception_b1_1x1_2_bn_W, inception_b1_1x1_2_bn_B, inception_b1_1x1_2_scale_W, inception_b1_1x1_2_scale_B, inception_b1_1x1_2_bn_eps, inception_b1_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_2_bn_node));

    // inception_b1_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x1_2_relu Layer
    vx_size inception_b1_1x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x1_2_relu;
    inception_b1_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_relu);
    vx_enum inception_b1_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x1_2_relu_param_a = 0;
    vx_float32 inception_b1_1x1_2_relu_param_b = 0;
    vx_node inception_b1_1x1_2_relu_node;
    inception_b1_1x1_2_relu_node = vxActivationLayer(graph, inception_b1_1x1_2_scale, inception_b1_1x1_2_relu_mode, inception_b1_1x1_2_relu_param_a, inception_b1_1x1_2_relu_param_b, inception_b1_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_2_relu_node));

    // inception_b1_1x7_reduce Layer
    vx_size inception_b1_1x7_reduce_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_reduce;
    inception_b1_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b1_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce);
    vx_size inception_b1_1x7_reduce_W_dims[4] = { 1, 1, 768, 128 };
    vx_tensor inception_b1_1x7_reduce_W;
    inception_b1_1x7_reduce_W = vxCreateTensor(context,4, inception_b1_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_W, dataFolder + "/weights/inception_b1_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_reduce_params;
    inception_b1_1x7_reduce_params.padding_x = 0;
    inception_b1_1x7_reduce_params.padding_y = 0;
    inception_b1_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_reduce_params.dilation_x = 0;
    inception_b1_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b1_1x7_reduce_node;
    inception_b1_1x7_reduce_node = vxConvolutionLayer(graph, reduction_a_concat_reduction_a_concat_0_split_1, inception_b1_1x7_reduce_W, NULL, &inception_b1_1x7_reduce_params, sizeof(inception_b1_1x7_reduce_params ), inception_b1_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_reduce_node));

    // inception_b1_1x7_reduce_bn Layer
    vx_size inception_b1_1x7_reduce_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_reduce_scale;
    inception_b1_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_scale);
    vx_size inception_b1_1x7_reduce_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_reduce_bn_W;
    inception_b1_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b1_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_bn_W, dataFolder + "/weights/inception_b1_1x7_reduce_bn.f32"));
    vx_size inception_b1_1x7_reduce_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_reduce_bn_B;
    inception_b1_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b1_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_bn_B, dataFolder + "/bias/inception_b1_1x7_reduce_bn.f32"));
    vx_size inception_b1_1x7_reduce_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_reduce_scale_W;
    inception_b1_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b1_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_scale_W, dataFolder + "/weights/inception_b1_1x7_reduce_scale.f32"));
    vx_size inception_b1_1x7_reduce_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_reduce_scale_B;
    inception_b1_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b1_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_scale_B, dataFolder + "/bias/inception_b1_1x7_reduce_scale.f32"));
    vx_node inception_b1_1x7_reduce_bn_node;
    inception_b1_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7_reduce, inception_b1_1x7_reduce_bn_W, inception_b1_1x7_reduce_bn_B, inception_b1_1x7_reduce_scale_W, inception_b1_1x7_reduce_scale_B, inception_b1_1x7_reduce_bn_eps, inception_b1_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_reduce_bn_node));

    // inception_b1_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_reduce_relu Layer
    vx_size inception_b1_1x7_reduce_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_reduce_relu;
    inception_b1_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_relu);
    vx_enum inception_b1_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b1_1x7_reduce_relu_param_b = 0;
    vx_node inception_b1_1x7_reduce_relu_node;
    inception_b1_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b1_1x7_reduce_scale, inception_b1_1x7_reduce_relu_mode, inception_b1_1x7_reduce_relu_param_a, inception_b1_1x7_reduce_relu_param_b, inception_b1_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_reduce_relu_node));

    // inception_b1_1x7 Layer
    vx_size inception_b1_1x7_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7;
    inception_b1_1x7 = vxCreateVirtualTensor(graph,4, inception_b1_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7);
    vx_size inception_b1_1x7_W_dims[4] = { 7, 1, 128, 128 };
    vx_tensor inception_b1_1x7_W;
    inception_b1_1x7_W = vxCreateTensor(context,4, inception_b1_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_W, dataFolder + "/weights/inception_b1_1x7.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_params;
    inception_b1_1x7_params.padding_x = 3;
    inception_b1_1x7_params.padding_y = 0;
    inception_b1_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_params.dilation_x = 0;
    inception_b1_1x7_params.dilation_y = 0;
    vx_node inception_b1_1x7_node;
    inception_b1_1x7_node = vxConvolutionLayer(graph, inception_b1_1x7_reduce_relu, inception_b1_1x7_W, NULL, &inception_b1_1x7_params, sizeof(inception_b1_1x7_params ), inception_b1_1x7);
    ERROR_CHECK_OBJECT(inception_b1_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_node));

    // inception_b1_1x7_bn Layer
    vx_size inception_b1_1x7_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_scale;
    inception_b1_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_scale);
    vx_size inception_b1_1x7_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_1x7_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_bn_W;
    inception_b1_1x7_bn_W = vxCreateTensor(context,1, inception_b1_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_bn_W, dataFolder + "/weights/inception_b1_1x7_bn.f32"));
    vx_size inception_b1_1x7_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_bn_B;
    inception_b1_1x7_bn_B = vxCreateTensor(context,1, inception_b1_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_bn_B, dataFolder + "/bias/inception_b1_1x7_bn.f32"));
    vx_size inception_b1_1x7_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_scale_W;
    inception_b1_1x7_scale_W = vxCreateTensor(context,1, inception_b1_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_scale_W, dataFolder + "/weights/inception_b1_1x7_scale.f32"));
    vx_size inception_b1_1x7_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_scale_B;
    inception_b1_1x7_scale_B = vxCreateTensor(context,1, inception_b1_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_scale_B, dataFolder + "/bias/inception_b1_1x7_scale.f32"));
    vx_node inception_b1_1x7_bn_node;
    inception_b1_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7, inception_b1_1x7_bn_W, inception_b1_1x7_bn_B, inception_b1_1x7_scale_W, inception_b1_1x7_scale_B, inception_b1_1x7_bn_eps, inception_b1_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_bn_node));

    // inception_b1_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_relu Layer
    vx_size inception_b1_1x7_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_relu;
    inception_b1_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_relu);
    vx_enum inception_b1_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_relu_param_a = 0;
    vx_float32 inception_b1_1x7_relu_param_b = 0;
    vx_node inception_b1_1x7_relu_node;
    inception_b1_1x7_relu_node = vxActivationLayer(graph, inception_b1_1x7_scale, inception_b1_1x7_relu_mode, inception_b1_1x7_relu_param_a, inception_b1_1x7_relu_param_b, inception_b1_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_relu_node));

    // inception_b1_7x1 Layer
    vx_size inception_b1_7x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1;
    inception_b1_7x1 = vxCreateVirtualTensor(graph,4, inception_b1_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1);
    vx_size inception_b1_7x1_W_dims[4] = { 1, 7, 128, 192 };
    vx_tensor inception_b1_7x1_W;
    inception_b1_7x1_W = vxCreateTensor(context,4, inception_b1_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_W, dataFolder + "/weights/inception_b1_7x1.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_params;
    inception_b1_7x1_params.padding_x = 0;
    inception_b1_7x1_params.padding_y = 3;
    inception_b1_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_params.dilation_x = 0;
    inception_b1_7x1_params.dilation_y = 0;
    vx_node inception_b1_7x1_node;
    inception_b1_7x1_node = vxConvolutionLayer(graph, inception_b1_1x7_relu, inception_b1_7x1_W, NULL, &inception_b1_7x1_params, sizeof(inception_b1_7x1_params ), inception_b1_7x1);
    ERROR_CHECK_OBJECT(inception_b1_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_node));

    // inception_b1_7x1_bn Layer
    vx_size inception_b1_7x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_scale;
    inception_b1_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_scale);
    vx_size inception_b1_7x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_7x1_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_bn_W;
    inception_b1_7x1_bn_W = vxCreateTensor(context,1, inception_b1_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_bn_W, dataFolder + "/weights/inception_b1_7x1_bn.f32"));
    vx_size inception_b1_7x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_bn_B;
    inception_b1_7x1_bn_B = vxCreateTensor(context,1, inception_b1_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_bn_B, dataFolder + "/bias/inception_b1_7x1_bn.f32"));
    vx_size inception_b1_7x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_scale_W;
    inception_b1_7x1_scale_W = vxCreateTensor(context,1, inception_b1_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_scale_W, dataFolder + "/weights/inception_b1_7x1_scale.f32"));
    vx_size inception_b1_7x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_scale_B;
    inception_b1_7x1_scale_B = vxCreateTensor(context,1, inception_b1_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_scale_B, dataFolder + "/bias/inception_b1_7x1_scale.f32"));
    vx_node inception_b1_7x1_bn_node;
    inception_b1_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1, inception_b1_7x1_bn_W, inception_b1_7x1_bn_B, inception_b1_7x1_scale_W, inception_b1_7x1_scale_B, inception_b1_7x1_bn_eps, inception_b1_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_bn_node));

    // inception_b1_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_relu Layer
    vx_size inception_b1_7x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_relu;
    inception_b1_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_relu);
    vx_enum inception_b1_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_relu_param_a = 0;
    vx_float32 inception_b1_7x1_relu_param_b = 0;
    vx_node inception_b1_7x1_relu_node;
    inception_b1_7x1_relu_node = vxActivationLayer(graph, inception_b1_7x1_scale, inception_b1_7x1_relu_mode, inception_b1_7x1_relu_param_a, inception_b1_7x1_relu_param_b, inception_b1_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_relu_node));

    // inception_b1_7x1_reduce Layer
    vx_size inception_b1_7x1_reduce_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_reduce;
    inception_b1_7x1_reduce = vxCreateVirtualTensor(graph,4, inception_b1_7x1_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce);
    vx_size inception_b1_7x1_reduce_W_dims[4] = { 1, 1, 768, 128 };
    vx_tensor inception_b1_7x1_reduce_W;
    inception_b1_7x1_reduce_W = vxCreateTensor(context,4, inception_b1_7x1_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_reduce_W, dataFolder + "/weights/inception_b1_7x1_reduce.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_reduce_params;
    inception_b1_7x1_reduce_params.padding_x = 0;
    inception_b1_7x1_reduce_params.padding_y = 0;
    inception_b1_7x1_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_reduce_params.dilation_x = 0;
    inception_b1_7x1_reduce_params.dilation_y = 0;
    vx_node inception_b1_7x1_reduce_node;
    inception_b1_7x1_reduce_node = vxConvolutionLayer(graph, reduction_a_concat_reduction_a_concat_0_split_2, inception_b1_7x1_reduce_W, NULL, &inception_b1_7x1_reduce_params, sizeof(inception_b1_7x1_reduce_params ), inception_b1_7x1_reduce);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_reduce_node));

    // inception_b1_7x1_reduce_bn Layer
    vx_size inception_b1_7x1_reduce_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_reduce_scale;
    inception_b1_7x1_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_scale);
    vx_size inception_b1_7x1_reduce_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_7x1_reduce_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_reduce_bn_W;
    inception_b1_7x1_reduce_bn_W = vxCreateTensor(context,1, inception_b1_7x1_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_reduce_bn_W, dataFolder + "/weights/inception_b1_7x1_reduce_bn.f32"));
    vx_size inception_b1_7x1_reduce_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_reduce_bn_B;
    inception_b1_7x1_reduce_bn_B = vxCreateTensor(context,1, inception_b1_7x1_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_reduce_bn_B, dataFolder + "/bias/inception_b1_7x1_reduce_bn.f32"));
    vx_size inception_b1_7x1_reduce_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_reduce_scale_W;
    inception_b1_7x1_reduce_scale_W = vxCreateTensor(context,1, inception_b1_7x1_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_reduce_scale_W, dataFolder + "/weights/inception_b1_7x1_reduce_scale.f32"));
    vx_size inception_b1_7x1_reduce_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_reduce_scale_B;
    inception_b1_7x1_reduce_scale_B = vxCreateTensor(context,1, inception_b1_7x1_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_reduce_scale_B, dataFolder + "/bias/inception_b1_7x1_reduce_scale.f32"));
    vx_node inception_b1_7x1_reduce_bn_node;
    inception_b1_7x1_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1_reduce, inception_b1_7x1_reduce_bn_W, inception_b1_7x1_reduce_bn_B, inception_b1_7x1_reduce_scale_W, inception_b1_7x1_reduce_scale_B, inception_b1_7x1_reduce_bn_eps, inception_b1_7x1_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_reduce_bn_node));

    // inception_b1_7x1_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_reduce_relu Layer
    vx_size inception_b1_7x1_reduce_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_reduce_relu;
    inception_b1_7x1_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_relu);
    vx_enum inception_b1_7x1_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_reduce_relu_param_a = 0;
    vx_float32 inception_b1_7x1_reduce_relu_param_b = 0;
    vx_node inception_b1_7x1_reduce_relu_node;
    inception_b1_7x1_reduce_relu_node = vxActivationLayer(graph, inception_b1_7x1_reduce_scale, inception_b1_7x1_reduce_relu_mode, inception_b1_7x1_reduce_relu_param_a, inception_b1_7x1_reduce_relu_param_b, inception_b1_7x1_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_reduce_relu_node));

    // inception_b1_7x1_2 Layer
    vx_size inception_b1_7x1_2_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_2;
    inception_b1_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2);
    vx_size inception_b1_7x1_2_W_dims[4] = { 1, 7, 128, 128 };
    vx_tensor inception_b1_7x1_2_W;
    inception_b1_7x1_2_W = vxCreateTensor(context,4, inception_b1_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_W, dataFolder + "/weights/inception_b1_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_2_params;
    inception_b1_7x1_2_params.padding_x = 0;
    inception_b1_7x1_2_params.padding_y = 3;
    inception_b1_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_2_params.dilation_x = 0;
    inception_b1_7x1_2_params.dilation_y = 0;
    vx_node inception_b1_7x1_2_node;
    inception_b1_7x1_2_node = vxConvolutionLayer(graph, inception_b1_7x1_reduce_relu, inception_b1_7x1_2_W, NULL, &inception_b1_7x1_2_params, sizeof(inception_b1_7x1_2_params ), inception_b1_7x1_2);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_node));

    // inception_b1_7x1_2_bn Layer
    vx_size inception_b1_7x1_2_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_2_scale;
    inception_b1_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_scale);
    vx_size inception_b1_7x1_2_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_2_bn_W;
    inception_b1_7x1_2_bn_W = vxCreateTensor(context,1, inception_b1_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_bn_W, dataFolder + "/weights/inception_b1_7x1_2_bn.f32"));
    vx_size inception_b1_7x1_2_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_2_bn_B;
    inception_b1_7x1_2_bn_B = vxCreateTensor(context,1, inception_b1_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_bn_B, dataFolder + "/bias/inception_b1_7x1_2_bn.f32"));
    vx_size inception_b1_7x1_2_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_2_scale_W;
    inception_b1_7x1_2_scale_W = vxCreateTensor(context,1, inception_b1_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_scale_W, dataFolder + "/weights/inception_b1_7x1_2_scale.f32"));
    vx_size inception_b1_7x1_2_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_2_scale_B;
    inception_b1_7x1_2_scale_B = vxCreateTensor(context,1, inception_b1_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_scale_B, dataFolder + "/bias/inception_b1_7x1_2_scale.f32"));
    vx_node inception_b1_7x1_2_bn_node;
    inception_b1_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1_2, inception_b1_7x1_2_bn_W, inception_b1_7x1_2_bn_B, inception_b1_7x1_2_scale_W, inception_b1_7x1_2_scale_B, inception_b1_7x1_2_bn_eps, inception_b1_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_bn_node));

    // inception_b1_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_2_relu Layer
    vx_size inception_b1_7x1_2_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_2_relu;
    inception_b1_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_relu);
    vx_enum inception_b1_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_2_relu_param_a = 0;
    vx_float32 inception_b1_7x1_2_relu_param_b = 0;
    vx_node inception_b1_7x1_2_relu_node;
    inception_b1_7x1_2_relu_node = vxActivationLayer(graph, inception_b1_7x1_2_scale, inception_b1_7x1_2_relu_mode, inception_b1_7x1_2_relu_param_a, inception_b1_7x1_2_relu_param_b, inception_b1_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_relu_node));

    // inception_b1_1x7_2 Layer
    vx_size inception_b1_1x7_2_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_2;
    inception_b1_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b1_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2);
    vx_size inception_b1_1x7_2_W_dims[4] = { 7, 1, 128, 128 };
    vx_tensor inception_b1_1x7_2_W;
    inception_b1_1x7_2_W = vxCreateTensor(context,4, inception_b1_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_W, dataFolder + "/weights/inception_b1_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_2_params;
    inception_b1_1x7_2_params.padding_x = 3;
    inception_b1_1x7_2_params.padding_y = 0;
    inception_b1_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_2_params.dilation_x = 0;
    inception_b1_1x7_2_params.dilation_y = 0;
    vx_node inception_b1_1x7_2_node;
    inception_b1_1x7_2_node = vxConvolutionLayer(graph, inception_b1_7x1_2_relu, inception_b1_1x7_2_W, NULL, &inception_b1_1x7_2_params, sizeof(inception_b1_1x7_2_params ), inception_b1_1x7_2);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_2_node));

    // inception_b1_1x7_2_bn Layer
    vx_size inception_b1_1x7_2_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_2_scale;
    inception_b1_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_scale);
    vx_size inception_b1_1x7_2_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_2_bn_W;
    inception_b1_1x7_2_bn_W = vxCreateTensor(context,1, inception_b1_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_bn_W, dataFolder + "/weights/inception_b1_1x7_2_bn.f32"));
    vx_size inception_b1_1x7_2_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_2_bn_B;
    inception_b1_1x7_2_bn_B = vxCreateTensor(context,1, inception_b1_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_bn_B, dataFolder + "/bias/inception_b1_1x7_2_bn.f32"));
    vx_size inception_b1_1x7_2_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_2_scale_W;
    inception_b1_1x7_2_scale_W = vxCreateTensor(context,1, inception_b1_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_scale_W, dataFolder + "/weights/inception_b1_1x7_2_scale.f32"));
    vx_size inception_b1_1x7_2_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x7_2_scale_B;
    inception_b1_1x7_2_scale_B = vxCreateTensor(context,1, inception_b1_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_scale_B, dataFolder + "/bias/inception_b1_1x7_2_scale.f32"));
    vx_node inception_b1_1x7_2_bn_node;
    inception_b1_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7_2, inception_b1_1x7_2_bn_W, inception_b1_1x7_2_bn_B, inception_b1_1x7_2_scale_W, inception_b1_1x7_2_scale_B, inception_b1_1x7_2_bn_eps, inception_b1_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_2_bn_node));

    // inception_b1_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_2_relu Layer
    vx_size inception_b1_1x7_2_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x7_2_relu;
    inception_b1_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_relu);
    vx_enum inception_b1_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_2_relu_param_a = 0;
    vx_float32 inception_b1_1x7_2_relu_param_b = 0;
    vx_node inception_b1_1x7_2_relu_node;
    inception_b1_1x7_2_relu_node = vxActivationLayer(graph, inception_b1_1x7_2_scale, inception_b1_1x7_2_relu_mode, inception_b1_1x7_2_relu_param_a, inception_b1_1x7_2_relu_param_b, inception_b1_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_2_relu_node));

    // inception_b1_7x1_3 Layer
    vx_size inception_b1_7x1_3_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_3;
    inception_b1_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b1_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3);
    vx_size inception_b1_7x1_3_W_dims[4] = { 1, 7, 128, 128 };
    vx_tensor inception_b1_7x1_3_W;
    inception_b1_7x1_3_W = vxCreateTensor(context,4, inception_b1_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_W, dataFolder + "/weights/inception_b1_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_3_params;
    inception_b1_7x1_3_params.padding_x = 0;
    inception_b1_7x1_3_params.padding_y = 3;
    inception_b1_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_3_params.dilation_x = 0;
    inception_b1_7x1_3_params.dilation_y = 0;
    vx_node inception_b1_7x1_3_node;
    inception_b1_7x1_3_node = vxConvolutionLayer(graph, inception_b1_1x7_2_relu, inception_b1_7x1_3_W, NULL, &inception_b1_7x1_3_params, sizeof(inception_b1_7x1_3_params ), inception_b1_7x1_3);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_3_node));

    // inception_b1_7x1_3_bn Layer
    vx_size inception_b1_7x1_3_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_3_scale;
    inception_b1_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_scale);
    vx_size inception_b1_7x1_3_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_3_bn_W;
    inception_b1_7x1_3_bn_W = vxCreateTensor(context,1, inception_b1_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_bn_W, dataFolder + "/weights/inception_b1_7x1_3_bn.f32"));
    vx_size inception_b1_7x1_3_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_3_bn_B;
    inception_b1_7x1_3_bn_B = vxCreateTensor(context,1, inception_b1_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_bn_B, dataFolder + "/bias/inception_b1_7x1_3_bn.f32"));
    vx_size inception_b1_7x1_3_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_3_scale_W;
    inception_b1_7x1_3_scale_W = vxCreateTensor(context,1, inception_b1_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_scale_W, dataFolder + "/weights/inception_b1_7x1_3_scale.f32"));
    vx_size inception_b1_7x1_3_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_7x1_3_scale_B;
    inception_b1_7x1_3_scale_B = vxCreateTensor(context,1, inception_b1_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_scale_B, dataFolder + "/bias/inception_b1_7x1_3_scale.f32"));
    vx_node inception_b1_7x1_3_bn_node;
    inception_b1_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1_3, inception_b1_7x1_3_bn_W, inception_b1_7x1_3_bn_B, inception_b1_7x1_3_scale_W, inception_b1_7x1_3_scale_B, inception_b1_7x1_3_bn_eps, inception_b1_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_3_bn_node));

    // inception_b1_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_3_relu Layer
    vx_size inception_b1_7x1_3_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_7x1_3_relu;
    inception_b1_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_relu);
    vx_enum inception_b1_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_3_relu_param_a = 0;
    vx_float32 inception_b1_7x1_3_relu_param_b = 0;
    vx_node inception_b1_7x1_3_relu_node;
    inception_b1_7x1_3_relu_node = vxActivationLayer(graph, inception_b1_7x1_3_scale, inception_b1_7x1_3_relu_mode, inception_b1_7x1_3_relu_param_a, inception_b1_7x1_3_relu_param_b, inception_b1_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_3_relu_node));

    // inception_b1_1x7_3 Layer
    vx_size inception_b1_1x7_3_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x7_3;
    inception_b1_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b1_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3);
    vx_size inception_b1_1x7_3_W_dims[4] = { 7, 1, 128, 192 };
    vx_tensor inception_b1_1x7_3_W;
    inception_b1_1x7_3_W = vxCreateTensor(context,4, inception_b1_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_W, dataFolder + "/weights/inception_b1_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_3_params;
    inception_b1_1x7_3_params.padding_x = 3;
    inception_b1_1x7_3_params.padding_y = 0;
    inception_b1_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_3_params.dilation_x = 0;
    inception_b1_1x7_3_params.dilation_y = 0;
    vx_node inception_b1_1x7_3_node;
    inception_b1_1x7_3_node = vxConvolutionLayer(graph, inception_b1_7x1_3_relu, inception_b1_1x7_3_W, NULL, &inception_b1_1x7_3_params, sizeof(inception_b1_1x7_3_params ), inception_b1_1x7_3);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_3_node));

    // inception_b1_1x7_3_bn Layer
    vx_size inception_b1_1x7_3_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x7_3_scale;
    inception_b1_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_scale);
    vx_size inception_b1_1x7_3_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_3_bn_W;
    inception_b1_1x7_3_bn_W = vxCreateTensor(context,1, inception_b1_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_bn_W, dataFolder + "/weights/inception_b1_1x7_3_bn.f32"));
    vx_size inception_b1_1x7_3_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x7_3_bn_B;
    inception_b1_1x7_3_bn_B = vxCreateTensor(context,1, inception_b1_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_bn_B, dataFolder + "/bias/inception_b1_1x7_3_bn.f32"));
    vx_size inception_b1_1x7_3_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_1x7_3_scale_W;
    inception_b1_1x7_3_scale_W = vxCreateTensor(context,1, inception_b1_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_scale_W, dataFolder + "/weights/inception_b1_1x7_3_scale.f32"));
    vx_size inception_b1_1x7_3_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x7_3_scale_B;
    inception_b1_1x7_3_scale_B = vxCreateTensor(context,1, inception_b1_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_scale_B, dataFolder + "/bias/inception_b1_1x7_3_scale.f32"));
    vx_node inception_b1_1x7_3_bn_node;
    inception_b1_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7_3, inception_b1_1x7_3_bn_W, inception_b1_1x7_3_bn_B, inception_b1_1x7_3_scale_W, inception_b1_1x7_3_scale_B, inception_b1_1x7_3_bn_eps, inception_b1_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_3_bn_node));

    // inception_b1_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_3_relu Layer
    vx_size inception_b1_1x7_3_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x7_3_relu;
    inception_b1_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_relu);
    vx_enum inception_b1_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_3_relu_param_a = 0;
    vx_float32 inception_b1_1x7_3_relu_param_b = 0;
    vx_node inception_b1_1x7_3_relu_node;
    inception_b1_1x7_3_relu_node = vxActivationLayer(graph, inception_b1_1x7_3_scale, inception_b1_1x7_3_relu_mode, inception_b1_1x7_3_relu_param_a, inception_b1_1x7_3_relu_param_b, inception_b1_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_3_relu_node));

    // inception_b1_pool_ave Layer
    vx_size inception_b1_pool_ave_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b1_pool_ave;
    inception_b1_pool_ave = vxCreateVirtualTensor(graph,4, inception_b1_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_pool_ave);
    vx_enum inception_b1_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b1_pool_ave_kernel_w = 3;
    vx_size inception_b1_pool_ave_kernel_h = 3;
    vx_size inception_b1_pool_ave_pad_w = 1;
    vx_size inception_b1_pool_ave_pad_h = 1;
    vx_enum inception_b1_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b1_pool_ave_node;
    inception_b1_pool_ave_node = vxPoolingLayer(graph, reduction_a_concat_reduction_a_concat_0_split_3, inception_b1_pool_ave_type, inception_b1_pool_ave_kernel_w, inception_b1_pool_ave_kernel_h, inception_b1_pool_ave_pad_w, inception_b1_pool_ave_pad_h, inception_b1_pool_ave_roundPolicy, inception_b1_pool_ave );
    ERROR_CHECK_OBJECT(inception_b1_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_pool_ave_node));

    // inception_b1_1x1 Layer
    vx_size inception_b1_1x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x1;
    inception_b1_1x1 = vxCreateVirtualTensor(graph,4, inception_b1_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1);
    vx_size inception_b1_1x1_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b1_1x1_W;
    inception_b1_1x1_W = vxCreateTensor(context,4, inception_b1_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_W, dataFolder + "/weights/inception_b1_1x1.f32"));
    vx_nn_convolution_params_t inception_b1_1x1_params;
    inception_b1_1x1_params.padding_x = 0;
    inception_b1_1x1_params.padding_y = 0;
    inception_b1_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x1_params.dilation_x = 0;
    inception_b1_1x1_params.dilation_y = 0;
    vx_node inception_b1_1x1_node;
    inception_b1_1x1_node = vxConvolutionLayer(graph, inception_b1_pool_ave, inception_b1_1x1_W, NULL, &inception_b1_1x1_params, sizeof(inception_b1_1x1_params ), inception_b1_1x1);
    ERROR_CHECK_OBJECT(inception_b1_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_node));

    // inception_b1_1x1_bn Layer
    vx_size inception_b1_1x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x1_scale;
    inception_b1_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_scale);
    vx_size inception_b1_1x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_1x1_bn_eps = 0.001;
    vx_tensor inception_b1_1x1_bn_W;
    inception_b1_1x1_bn_W = vxCreateTensor(context,1, inception_b1_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_bn_W, dataFolder + "/weights/inception_b1_1x1_bn.f32"));
    vx_size inception_b1_1x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x1_bn_B;
    inception_b1_1x1_bn_B = vxCreateTensor(context,1, inception_b1_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_bn_B, dataFolder + "/bias/inception_b1_1x1_bn.f32"));
    vx_size inception_b1_1x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_1x1_scale_W;
    inception_b1_1x1_scale_W = vxCreateTensor(context,1, inception_b1_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_scale_W, dataFolder + "/weights/inception_b1_1x1_scale.f32"));
    vx_size inception_b1_1x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x1_scale_B;
    inception_b1_1x1_scale_B = vxCreateTensor(context,1, inception_b1_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_scale_B, dataFolder + "/bias/inception_b1_1x1_scale.f32"));
    vx_node inception_b1_1x1_bn_node;
    inception_b1_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x1, inception_b1_1x1_bn_W, inception_b1_1x1_bn_B, inception_b1_1x1_scale_W, inception_b1_1x1_scale_B, inception_b1_1x1_bn_eps, inception_b1_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_bn_node));

    // inception_b1_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x1_relu Layer
    vx_size inception_b1_1x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x1_relu;
    inception_b1_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_relu);
    vx_enum inception_b1_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x1_relu_param_a = 0;
    vx_float32 inception_b1_1x1_relu_param_b = 0;
    vx_node inception_b1_1x1_relu_node;
    inception_b1_1x1_relu_node = vxActivationLayer(graph, inception_b1_1x1_scale, inception_b1_1x1_relu_mode, inception_b1_1x1_relu_param_a, inception_b1_1x1_relu_param_b, inception_b1_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_relu_node));

    // inception_b1_concat Layer
    vx_size inception_b1_concat_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b1_concat;
    inception_b1_concat = vxCreateVirtualTensor(graph,4, inception_b1_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_concat);
    vx_node inception_b1_concat_node;
    inception_b1_concat_node = vxConcatLayer(graph, inception_b1_concat, inception_b1_1x1_2_relu, inception_b1_7x1_relu, inception_b1_1x7_3_relu, inception_b1_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b1_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_concat_node));

    // inception_b1_concat_inception_b1_concat_0_split_0 Layer
    vx_size inception_b1_concat_inception_b1_concat_0_split_0_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b1_concat_inception_b1_concat_0_split_0;
    inception_b1_concat_inception_b1_concat_0_split_0 = vxCreateVirtualTensor(graph,4, inception_b1_concat_inception_b1_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_0);
    vx_node inception_b1_concat_inception_b1_concat_0_split_0_node;
    inception_b1_concat_inception_b1_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_b1_concat, (vx_reference)inception_b1_concat_inception_b1_concat_0_split_0);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_concat_inception_b1_concat_0_split_0_node));

    // inception_b1_concat_inception_b1_concat_0_split_1 Layer
    vx_size inception_b1_concat_inception_b1_concat_0_split_1_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b1_concat_inception_b1_concat_0_split_1;
    inception_b1_concat_inception_b1_concat_0_split_1 = vxCreateVirtualTensor(graph,4, inception_b1_concat_inception_b1_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_1);
    vx_node inception_b1_concat_inception_b1_concat_0_split_1_node;
    inception_b1_concat_inception_b1_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_b1_concat, (vx_reference)inception_b1_concat_inception_b1_concat_0_split_1);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_concat_inception_b1_concat_0_split_1_node));

    // inception_b1_concat_inception_b1_concat_0_split_2 Layer
    vx_size inception_b1_concat_inception_b1_concat_0_split_2_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b1_concat_inception_b1_concat_0_split_2;
    inception_b1_concat_inception_b1_concat_0_split_2 = vxCreateVirtualTensor(graph,4, inception_b1_concat_inception_b1_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_2);
    vx_node inception_b1_concat_inception_b1_concat_0_split_2_node;
    inception_b1_concat_inception_b1_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_b1_concat, (vx_reference)inception_b1_concat_inception_b1_concat_0_split_2);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_concat_inception_b1_concat_0_split_2_node));

    // inception_b1_concat_inception_b1_concat_0_split_3 Layer
    vx_size inception_b1_concat_inception_b1_concat_0_split_3_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b1_concat_inception_b1_concat_0_split_3;
    inception_b1_concat_inception_b1_concat_0_split_3 = vxCreateVirtualTensor(graph,4, inception_b1_concat_inception_b1_concat_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_3);
    vx_node inception_b1_concat_inception_b1_concat_0_split_3_node;
    inception_b1_concat_inception_b1_concat_0_split_3_node = vxCopyNode( graph, (vx_reference)inception_b1_concat, (vx_reference)inception_b1_concat_inception_b1_concat_0_split_3);
    ERROR_CHECK_OBJECT(inception_b1_concat_inception_b1_concat_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_concat_inception_b1_concat_0_split_3_node));

    // inception_b2_1x1_2 Layer
    vx_size inception_b2_1x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x1_2;
    inception_b2_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b2_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2);
    vx_size inception_b2_1x1_2_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b2_1x1_2_W;
    inception_b2_1x1_2_W = vxCreateTensor(context,4, inception_b2_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_W, dataFolder + "/weights/inception_b2_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b2_1x1_2_params;
    inception_b2_1x1_2_params.padding_x = 0;
    inception_b2_1x1_2_params.padding_y = 0;
    inception_b2_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x1_2_params.dilation_x = 0;
    inception_b2_1x1_2_params.dilation_y = 0;
    vx_node inception_b2_1x1_2_node;
    inception_b2_1x1_2_node = vxConvolutionLayer(graph, inception_b1_concat_inception_b1_concat_0_split_0, inception_b2_1x1_2_W, NULL, &inception_b2_1x1_2_params, sizeof(inception_b2_1x1_2_params ), inception_b2_1x1_2);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_2_node));

    // inception_b2_1x1_2_bn Layer
    vx_size inception_b2_1x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x1_2_scale;
    inception_b2_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_scale);
    vx_size inception_b2_1x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b2_1x1_2_bn_W;
    inception_b2_1x1_2_bn_W = vxCreateTensor(context,1, inception_b2_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_bn_W, dataFolder + "/weights/inception_b2_1x1_2_bn.f32"));
    vx_size inception_b2_1x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x1_2_bn_B;
    inception_b2_1x1_2_bn_B = vxCreateTensor(context,1, inception_b2_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_bn_B, dataFolder + "/bias/inception_b2_1x1_2_bn.f32"));
    vx_size inception_b2_1x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_1x1_2_scale_W;
    inception_b2_1x1_2_scale_W = vxCreateTensor(context,1, inception_b2_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_scale_W, dataFolder + "/weights/inception_b2_1x1_2_scale.f32"));
    vx_size inception_b2_1x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x1_2_scale_B;
    inception_b2_1x1_2_scale_B = vxCreateTensor(context,1, inception_b2_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_scale_B, dataFolder + "/bias/inception_b2_1x1_2_scale.f32"));
    vx_node inception_b2_1x1_2_bn_node;
    inception_b2_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x1_2, inception_b2_1x1_2_bn_W, inception_b2_1x1_2_bn_B, inception_b2_1x1_2_scale_W, inception_b2_1x1_2_scale_B, inception_b2_1x1_2_bn_eps, inception_b2_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_2_bn_node));

    // inception_b2_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x1_2_relu Layer
    vx_size inception_b2_1x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x1_2_relu;
    inception_b2_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_relu);
    vx_enum inception_b2_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x1_2_relu_param_a = 0;
    vx_float32 inception_b2_1x1_2_relu_param_b = 0;
    vx_node inception_b2_1x1_2_relu_node;
    inception_b2_1x1_2_relu_node = vxActivationLayer(graph, inception_b2_1x1_2_scale, inception_b2_1x1_2_relu_mode, inception_b2_1x1_2_relu_param_a, inception_b2_1x1_2_relu_param_b, inception_b2_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_2_relu_node));

    // inception_b2_1x7_reduce Layer
    vx_size inception_b2_1x7_reduce_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_reduce;
    inception_b2_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b2_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce);
    vx_size inception_b2_1x7_reduce_W_dims[4] = { 1, 1, 768, 160 };
    vx_tensor inception_b2_1x7_reduce_W;
    inception_b2_1x7_reduce_W = vxCreateTensor(context,4, inception_b2_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_W, dataFolder + "/weights/inception_b2_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_reduce_params;
    inception_b2_1x7_reduce_params.padding_x = 0;
    inception_b2_1x7_reduce_params.padding_y = 0;
    inception_b2_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_reduce_params.dilation_x = 0;
    inception_b2_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b2_1x7_reduce_node;
    inception_b2_1x7_reduce_node = vxConvolutionLayer(graph, inception_b1_concat_inception_b1_concat_0_split_1, inception_b2_1x7_reduce_W, NULL, &inception_b2_1x7_reduce_params, sizeof(inception_b2_1x7_reduce_params ), inception_b2_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_reduce_node));

    // inception_b2_1x7_reduce_bn Layer
    vx_size inception_b2_1x7_reduce_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_reduce_scale;
    inception_b2_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_scale);
    vx_size inception_b2_1x7_reduce_bn_W_dims[1] = { 160 };
    vx_float32 inception_b2_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_reduce_bn_W;
    inception_b2_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b2_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_bn_W, dataFolder + "/weights/inception_b2_1x7_reduce_bn.f32"));
    vx_size inception_b2_1x7_reduce_bn_B_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_reduce_bn_B;
    inception_b2_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b2_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_bn_B, dataFolder + "/bias/inception_b2_1x7_reduce_bn.f32"));
    vx_size inception_b2_1x7_reduce_scale_W_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_reduce_scale_W;
    inception_b2_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b2_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_scale_W, dataFolder + "/weights/inception_b2_1x7_reduce_scale.f32"));
    vx_size inception_b2_1x7_reduce_scale_B_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_reduce_scale_B;
    inception_b2_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b2_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_scale_B, dataFolder + "/bias/inception_b2_1x7_reduce_scale.f32"));
    vx_node inception_b2_1x7_reduce_bn_node;
    inception_b2_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7_reduce, inception_b2_1x7_reduce_bn_W, inception_b2_1x7_reduce_bn_B, inception_b2_1x7_reduce_scale_W, inception_b2_1x7_reduce_scale_B, inception_b2_1x7_reduce_bn_eps, inception_b2_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_reduce_bn_node));

    // inception_b2_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_reduce_relu Layer
    vx_size inception_b2_1x7_reduce_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_reduce_relu;
    inception_b2_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_relu);
    vx_enum inception_b2_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b2_1x7_reduce_relu_param_b = 0;
    vx_node inception_b2_1x7_reduce_relu_node;
    inception_b2_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b2_1x7_reduce_scale, inception_b2_1x7_reduce_relu_mode, inception_b2_1x7_reduce_relu_param_a, inception_b2_1x7_reduce_relu_param_b, inception_b2_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_reduce_relu_node));

    // inception_b2_1x7 Layer
    vx_size inception_b2_1x7_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7;
    inception_b2_1x7 = vxCreateVirtualTensor(graph,4, inception_b2_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7);
    vx_size inception_b2_1x7_W_dims[4] = { 7, 1, 160, 160 };
    vx_tensor inception_b2_1x7_W;
    inception_b2_1x7_W = vxCreateTensor(context,4, inception_b2_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_W, dataFolder + "/weights/inception_b2_1x7.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_params;
    inception_b2_1x7_params.padding_x = 3;
    inception_b2_1x7_params.padding_y = 0;
    inception_b2_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_params.dilation_x = 0;
    inception_b2_1x7_params.dilation_y = 0;
    vx_node inception_b2_1x7_node;
    inception_b2_1x7_node = vxConvolutionLayer(graph, inception_b2_1x7_reduce_relu, inception_b2_1x7_W, NULL, &inception_b2_1x7_params, sizeof(inception_b2_1x7_params ), inception_b2_1x7);
    ERROR_CHECK_OBJECT(inception_b2_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_node));

    // inception_b2_1x7_bn Layer
    vx_size inception_b2_1x7_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_scale;
    inception_b2_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_scale);
    vx_size inception_b2_1x7_bn_W_dims[1] = { 160 };
    vx_float32 inception_b2_1x7_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_bn_W;
    inception_b2_1x7_bn_W = vxCreateTensor(context,1, inception_b2_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_bn_W, dataFolder + "/weights/inception_b2_1x7_bn.f32"));
    vx_size inception_b2_1x7_bn_B_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_bn_B;
    inception_b2_1x7_bn_B = vxCreateTensor(context,1, inception_b2_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_bn_B, dataFolder + "/bias/inception_b2_1x7_bn.f32"));
    vx_size inception_b2_1x7_scale_W_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_scale_W;
    inception_b2_1x7_scale_W = vxCreateTensor(context,1, inception_b2_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_scale_W, dataFolder + "/weights/inception_b2_1x7_scale.f32"));
    vx_size inception_b2_1x7_scale_B_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_scale_B;
    inception_b2_1x7_scale_B = vxCreateTensor(context,1, inception_b2_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_scale_B, dataFolder + "/bias/inception_b2_1x7_scale.f32"));
    vx_node inception_b2_1x7_bn_node;
    inception_b2_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7, inception_b2_1x7_bn_W, inception_b2_1x7_bn_B, inception_b2_1x7_scale_W, inception_b2_1x7_scale_B, inception_b2_1x7_bn_eps, inception_b2_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_bn_node));

    // inception_b2_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_relu Layer
    vx_size inception_b2_1x7_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_relu;
    inception_b2_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_relu);
    vx_enum inception_b2_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_relu_param_a = 0;
    vx_float32 inception_b2_1x7_relu_param_b = 0;
    vx_node inception_b2_1x7_relu_node;
    inception_b2_1x7_relu_node = vxActivationLayer(graph, inception_b2_1x7_scale, inception_b2_1x7_relu_mode, inception_b2_1x7_relu_param_a, inception_b2_1x7_relu_param_b, inception_b2_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_relu_node));

    // inception_b2_7x1 Layer
    vx_size inception_b2_7x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1;
    inception_b2_7x1 = vxCreateVirtualTensor(graph,4, inception_b2_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1);
    vx_size inception_b2_7x1_W_dims[4] = { 1, 7, 160, 192 };
    vx_tensor inception_b2_7x1_W;
    inception_b2_7x1_W = vxCreateTensor(context,4, inception_b2_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_W, dataFolder + "/weights/inception_b2_7x1.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_params;
    inception_b2_7x1_params.padding_x = 0;
    inception_b2_7x1_params.padding_y = 3;
    inception_b2_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_params.dilation_x = 0;
    inception_b2_7x1_params.dilation_y = 0;
    vx_node inception_b2_7x1_node;
    inception_b2_7x1_node = vxConvolutionLayer(graph, inception_b2_1x7_relu, inception_b2_7x1_W, NULL, &inception_b2_7x1_params, sizeof(inception_b2_7x1_params ), inception_b2_7x1);
    ERROR_CHECK_OBJECT(inception_b2_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_node));

    // inception_b2_7x1_bn Layer
    vx_size inception_b2_7x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_scale;
    inception_b2_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_scale);
    vx_size inception_b2_7x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_7x1_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_bn_W;
    inception_b2_7x1_bn_W = vxCreateTensor(context,1, inception_b2_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_bn_W, dataFolder + "/weights/inception_b2_7x1_bn.f32"));
    vx_size inception_b2_7x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_bn_B;
    inception_b2_7x1_bn_B = vxCreateTensor(context,1, inception_b2_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_bn_B, dataFolder + "/bias/inception_b2_7x1_bn.f32"));
    vx_size inception_b2_7x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_scale_W;
    inception_b2_7x1_scale_W = vxCreateTensor(context,1, inception_b2_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_scale_W, dataFolder + "/weights/inception_b2_7x1_scale.f32"));
    vx_size inception_b2_7x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_scale_B;
    inception_b2_7x1_scale_B = vxCreateTensor(context,1, inception_b2_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_scale_B, dataFolder + "/bias/inception_b2_7x1_scale.f32"));
    vx_node inception_b2_7x1_bn_node;
    inception_b2_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1, inception_b2_7x1_bn_W, inception_b2_7x1_bn_B, inception_b2_7x1_scale_W, inception_b2_7x1_scale_B, inception_b2_7x1_bn_eps, inception_b2_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_bn_node));

    // inception_b2_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_relu Layer
    vx_size inception_b2_7x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_relu;
    inception_b2_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_relu);
    vx_enum inception_b2_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_relu_param_a = 0;
    vx_float32 inception_b2_7x1_relu_param_b = 0;
    vx_node inception_b2_7x1_relu_node;
    inception_b2_7x1_relu_node = vxActivationLayer(graph, inception_b2_7x1_scale, inception_b2_7x1_relu_mode, inception_b2_7x1_relu_param_a, inception_b2_7x1_relu_param_b, inception_b2_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_relu_node));

    // inception_b2_7x1_reduce Layer
    vx_size inception_b2_7x1_reduce_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_reduce;
    inception_b2_7x1_reduce = vxCreateVirtualTensor(graph,4, inception_b2_7x1_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce);
    vx_size inception_b2_7x1_reduce_W_dims[4] = { 1, 1, 768, 160 };
    vx_tensor inception_b2_7x1_reduce_W;
    inception_b2_7x1_reduce_W = vxCreateTensor(context,4, inception_b2_7x1_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_reduce_W, dataFolder + "/weights/inception_b2_7x1_reduce.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_reduce_params;
    inception_b2_7x1_reduce_params.padding_x = 0;
    inception_b2_7x1_reduce_params.padding_y = 0;
    inception_b2_7x1_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_reduce_params.dilation_x = 0;
    inception_b2_7x1_reduce_params.dilation_y = 0;
    vx_node inception_b2_7x1_reduce_node;
    inception_b2_7x1_reduce_node = vxConvolutionLayer(graph, inception_b1_concat_inception_b1_concat_0_split_2, inception_b2_7x1_reduce_W, NULL, &inception_b2_7x1_reduce_params, sizeof(inception_b2_7x1_reduce_params ), inception_b2_7x1_reduce);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_reduce_node));

    // inception_b2_7x1_reduce_bn Layer
    vx_size inception_b2_7x1_reduce_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_reduce_scale;
    inception_b2_7x1_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_scale);
    vx_size inception_b2_7x1_reduce_bn_W_dims[1] = { 160 };
    vx_float32 inception_b2_7x1_reduce_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_reduce_bn_W;
    inception_b2_7x1_reduce_bn_W = vxCreateTensor(context,1, inception_b2_7x1_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_reduce_bn_W, dataFolder + "/weights/inception_b2_7x1_reduce_bn.f32"));
    vx_size inception_b2_7x1_reduce_bn_B_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_reduce_bn_B;
    inception_b2_7x1_reduce_bn_B = vxCreateTensor(context,1, inception_b2_7x1_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_reduce_bn_B, dataFolder + "/bias/inception_b2_7x1_reduce_bn.f32"));
    vx_size inception_b2_7x1_reduce_scale_W_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_reduce_scale_W;
    inception_b2_7x1_reduce_scale_W = vxCreateTensor(context,1, inception_b2_7x1_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_reduce_scale_W, dataFolder + "/weights/inception_b2_7x1_reduce_scale.f32"));
    vx_size inception_b2_7x1_reduce_scale_B_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_reduce_scale_B;
    inception_b2_7x1_reduce_scale_B = vxCreateTensor(context,1, inception_b2_7x1_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_reduce_scale_B, dataFolder + "/bias/inception_b2_7x1_reduce_scale.f32"));
    vx_node inception_b2_7x1_reduce_bn_node;
    inception_b2_7x1_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1_reduce, inception_b2_7x1_reduce_bn_W, inception_b2_7x1_reduce_bn_B, inception_b2_7x1_reduce_scale_W, inception_b2_7x1_reduce_scale_B, inception_b2_7x1_reduce_bn_eps, inception_b2_7x1_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_reduce_bn_node));

    // inception_b2_7x1_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_reduce_relu Layer
    vx_size inception_b2_7x1_reduce_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_reduce_relu;
    inception_b2_7x1_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_relu);
    vx_enum inception_b2_7x1_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_reduce_relu_param_a = 0;
    vx_float32 inception_b2_7x1_reduce_relu_param_b = 0;
    vx_node inception_b2_7x1_reduce_relu_node;
    inception_b2_7x1_reduce_relu_node = vxActivationLayer(graph, inception_b2_7x1_reduce_scale, inception_b2_7x1_reduce_relu_mode, inception_b2_7x1_reduce_relu_param_a, inception_b2_7x1_reduce_relu_param_b, inception_b2_7x1_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_reduce_relu_node));

    // inception_b2_7x1_2 Layer
    vx_size inception_b2_7x1_2_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_2;
    inception_b2_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2);
    vx_size inception_b2_7x1_2_W_dims[4] = { 1, 7, 160, 160 };
    vx_tensor inception_b2_7x1_2_W;
    inception_b2_7x1_2_W = vxCreateTensor(context,4, inception_b2_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_W, dataFolder + "/weights/inception_b2_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_2_params;
    inception_b2_7x1_2_params.padding_x = 0;
    inception_b2_7x1_2_params.padding_y = 3;
    inception_b2_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_2_params.dilation_x = 0;
    inception_b2_7x1_2_params.dilation_y = 0;
    vx_node inception_b2_7x1_2_node;
    inception_b2_7x1_2_node = vxConvolutionLayer(graph, inception_b2_7x1_reduce_relu, inception_b2_7x1_2_W, NULL, &inception_b2_7x1_2_params, sizeof(inception_b2_7x1_2_params ), inception_b2_7x1_2);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_node));

    // inception_b2_7x1_2_bn Layer
    vx_size inception_b2_7x1_2_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_2_scale;
    inception_b2_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_scale);
    vx_size inception_b2_7x1_2_bn_W_dims[1] = { 160 };
    vx_float32 inception_b2_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_2_bn_W;
    inception_b2_7x1_2_bn_W = vxCreateTensor(context,1, inception_b2_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_bn_W, dataFolder + "/weights/inception_b2_7x1_2_bn.f32"));
    vx_size inception_b2_7x1_2_bn_B_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_2_bn_B;
    inception_b2_7x1_2_bn_B = vxCreateTensor(context,1, inception_b2_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_bn_B, dataFolder + "/bias/inception_b2_7x1_2_bn.f32"));
    vx_size inception_b2_7x1_2_scale_W_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_2_scale_W;
    inception_b2_7x1_2_scale_W = vxCreateTensor(context,1, inception_b2_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_scale_W, dataFolder + "/weights/inception_b2_7x1_2_scale.f32"));
    vx_size inception_b2_7x1_2_scale_B_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_2_scale_B;
    inception_b2_7x1_2_scale_B = vxCreateTensor(context,1, inception_b2_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_scale_B, dataFolder + "/bias/inception_b2_7x1_2_scale.f32"));
    vx_node inception_b2_7x1_2_bn_node;
    inception_b2_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1_2, inception_b2_7x1_2_bn_W, inception_b2_7x1_2_bn_B, inception_b2_7x1_2_scale_W, inception_b2_7x1_2_scale_B, inception_b2_7x1_2_bn_eps, inception_b2_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_bn_node));

    // inception_b2_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_2_relu Layer
    vx_size inception_b2_7x1_2_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_2_relu;
    inception_b2_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_relu);
    vx_enum inception_b2_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_2_relu_param_a = 0;
    vx_float32 inception_b2_7x1_2_relu_param_b = 0;
    vx_node inception_b2_7x1_2_relu_node;
    inception_b2_7x1_2_relu_node = vxActivationLayer(graph, inception_b2_7x1_2_scale, inception_b2_7x1_2_relu_mode, inception_b2_7x1_2_relu_param_a, inception_b2_7x1_2_relu_param_b, inception_b2_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_relu_node));

    // inception_b2_1x7_2 Layer
    vx_size inception_b2_1x7_2_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_2;
    inception_b2_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b2_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2);
    vx_size inception_b2_1x7_2_W_dims[4] = { 7, 1, 160, 160 };
    vx_tensor inception_b2_1x7_2_W;
    inception_b2_1x7_2_W = vxCreateTensor(context,4, inception_b2_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_W, dataFolder + "/weights/inception_b2_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_2_params;
    inception_b2_1x7_2_params.padding_x = 3;
    inception_b2_1x7_2_params.padding_y = 0;
    inception_b2_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_2_params.dilation_x = 0;
    inception_b2_1x7_2_params.dilation_y = 0;
    vx_node inception_b2_1x7_2_node;
    inception_b2_1x7_2_node = vxConvolutionLayer(graph, inception_b2_7x1_2_relu, inception_b2_1x7_2_W, NULL, &inception_b2_1x7_2_params, sizeof(inception_b2_1x7_2_params ), inception_b2_1x7_2);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_2_node));

    // inception_b2_1x7_2_bn Layer
    vx_size inception_b2_1x7_2_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_2_scale;
    inception_b2_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_scale);
    vx_size inception_b2_1x7_2_bn_W_dims[1] = { 160 };
    vx_float32 inception_b2_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_2_bn_W;
    inception_b2_1x7_2_bn_W = vxCreateTensor(context,1, inception_b2_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_bn_W, dataFolder + "/weights/inception_b2_1x7_2_bn.f32"));
    vx_size inception_b2_1x7_2_bn_B_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_2_bn_B;
    inception_b2_1x7_2_bn_B = vxCreateTensor(context,1, inception_b2_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_bn_B, dataFolder + "/bias/inception_b2_1x7_2_bn.f32"));
    vx_size inception_b2_1x7_2_scale_W_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_2_scale_W;
    inception_b2_1x7_2_scale_W = vxCreateTensor(context,1, inception_b2_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_scale_W, dataFolder + "/weights/inception_b2_1x7_2_scale.f32"));
    vx_size inception_b2_1x7_2_scale_B_dims[1] = { 160 };
    vx_tensor inception_b2_1x7_2_scale_B;
    inception_b2_1x7_2_scale_B = vxCreateTensor(context,1, inception_b2_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_scale_B, dataFolder + "/bias/inception_b2_1x7_2_scale.f32"));
    vx_node inception_b2_1x7_2_bn_node;
    inception_b2_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7_2, inception_b2_1x7_2_bn_W, inception_b2_1x7_2_bn_B, inception_b2_1x7_2_scale_W, inception_b2_1x7_2_scale_B, inception_b2_1x7_2_bn_eps, inception_b2_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_2_bn_node));

    // inception_b2_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_2_relu Layer
    vx_size inception_b2_1x7_2_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_1x7_2_relu;
    inception_b2_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_relu);
    vx_enum inception_b2_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_2_relu_param_a = 0;
    vx_float32 inception_b2_1x7_2_relu_param_b = 0;
    vx_node inception_b2_1x7_2_relu_node;
    inception_b2_1x7_2_relu_node = vxActivationLayer(graph, inception_b2_1x7_2_scale, inception_b2_1x7_2_relu_mode, inception_b2_1x7_2_relu_param_a, inception_b2_1x7_2_relu_param_b, inception_b2_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_2_relu_node));

    // inception_b2_7x1_3 Layer
    vx_size inception_b2_7x1_3_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_3;
    inception_b2_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b2_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3);
    vx_size inception_b2_7x1_3_W_dims[4] = { 1, 7, 160, 160 };
    vx_tensor inception_b2_7x1_3_W;
    inception_b2_7x1_3_W = vxCreateTensor(context,4, inception_b2_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_W, dataFolder + "/weights/inception_b2_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_3_params;
    inception_b2_7x1_3_params.padding_x = 0;
    inception_b2_7x1_3_params.padding_y = 3;
    inception_b2_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_3_params.dilation_x = 0;
    inception_b2_7x1_3_params.dilation_y = 0;
    vx_node inception_b2_7x1_3_node;
    inception_b2_7x1_3_node = vxConvolutionLayer(graph, inception_b2_1x7_2_relu, inception_b2_7x1_3_W, NULL, &inception_b2_7x1_3_params, sizeof(inception_b2_7x1_3_params ), inception_b2_7x1_3);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_3_node));

    // inception_b2_7x1_3_bn Layer
    vx_size inception_b2_7x1_3_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_3_scale;
    inception_b2_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_scale);
    vx_size inception_b2_7x1_3_bn_W_dims[1] = { 160 };
    vx_float32 inception_b2_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_3_bn_W;
    inception_b2_7x1_3_bn_W = vxCreateTensor(context,1, inception_b2_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_bn_W, dataFolder + "/weights/inception_b2_7x1_3_bn.f32"));
    vx_size inception_b2_7x1_3_bn_B_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_3_bn_B;
    inception_b2_7x1_3_bn_B = vxCreateTensor(context,1, inception_b2_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_bn_B, dataFolder + "/bias/inception_b2_7x1_3_bn.f32"));
    vx_size inception_b2_7x1_3_scale_W_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_3_scale_W;
    inception_b2_7x1_3_scale_W = vxCreateTensor(context,1, inception_b2_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_scale_W, dataFolder + "/weights/inception_b2_7x1_3_scale.f32"));
    vx_size inception_b2_7x1_3_scale_B_dims[1] = { 160 };
    vx_tensor inception_b2_7x1_3_scale_B;
    inception_b2_7x1_3_scale_B = vxCreateTensor(context,1, inception_b2_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_scale_B, dataFolder + "/bias/inception_b2_7x1_3_scale.f32"));
    vx_node inception_b2_7x1_3_bn_node;
    inception_b2_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1_3, inception_b2_7x1_3_bn_W, inception_b2_7x1_3_bn_B, inception_b2_7x1_3_scale_W, inception_b2_7x1_3_scale_B, inception_b2_7x1_3_bn_eps, inception_b2_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_3_bn_node));

    // inception_b2_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_3_relu Layer
    vx_size inception_b2_7x1_3_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b2_7x1_3_relu;
    inception_b2_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_relu);
    vx_enum inception_b2_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_3_relu_param_a = 0;
    vx_float32 inception_b2_7x1_3_relu_param_b = 0;
    vx_node inception_b2_7x1_3_relu_node;
    inception_b2_7x1_3_relu_node = vxActivationLayer(graph, inception_b2_7x1_3_scale, inception_b2_7x1_3_relu_mode, inception_b2_7x1_3_relu_param_a, inception_b2_7x1_3_relu_param_b, inception_b2_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_3_relu_node));

    // inception_b2_1x7_3 Layer
    vx_size inception_b2_1x7_3_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x7_3;
    inception_b2_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b2_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3);
    vx_size inception_b2_1x7_3_W_dims[4] = { 7, 1, 160, 192 };
    vx_tensor inception_b2_1x7_3_W;
    inception_b2_1x7_3_W = vxCreateTensor(context,4, inception_b2_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_W, dataFolder + "/weights/inception_b2_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_3_params;
    inception_b2_1x7_3_params.padding_x = 3;
    inception_b2_1x7_3_params.padding_y = 0;
    inception_b2_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_3_params.dilation_x = 0;
    inception_b2_1x7_3_params.dilation_y = 0;
    vx_node inception_b2_1x7_3_node;
    inception_b2_1x7_3_node = vxConvolutionLayer(graph, inception_b2_7x1_3_relu, inception_b2_1x7_3_W, NULL, &inception_b2_1x7_3_params, sizeof(inception_b2_1x7_3_params ), inception_b2_1x7_3);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_3_node));

    // inception_b2_1x7_3_bn Layer
    vx_size inception_b2_1x7_3_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x7_3_scale;
    inception_b2_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_scale);
    vx_size inception_b2_1x7_3_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_3_bn_W;
    inception_b2_1x7_3_bn_W = vxCreateTensor(context,1, inception_b2_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_bn_W, dataFolder + "/weights/inception_b2_1x7_3_bn.f32"));
    vx_size inception_b2_1x7_3_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x7_3_bn_B;
    inception_b2_1x7_3_bn_B = vxCreateTensor(context,1, inception_b2_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_bn_B, dataFolder + "/bias/inception_b2_1x7_3_bn.f32"));
    vx_size inception_b2_1x7_3_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_1x7_3_scale_W;
    inception_b2_1x7_3_scale_W = vxCreateTensor(context,1, inception_b2_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_scale_W, dataFolder + "/weights/inception_b2_1x7_3_scale.f32"));
    vx_size inception_b2_1x7_3_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x7_3_scale_B;
    inception_b2_1x7_3_scale_B = vxCreateTensor(context,1, inception_b2_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_scale_B, dataFolder + "/bias/inception_b2_1x7_3_scale.f32"));
    vx_node inception_b2_1x7_3_bn_node;
    inception_b2_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7_3, inception_b2_1x7_3_bn_W, inception_b2_1x7_3_bn_B, inception_b2_1x7_3_scale_W, inception_b2_1x7_3_scale_B, inception_b2_1x7_3_bn_eps, inception_b2_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_3_bn_node));

    // inception_b2_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_3_relu Layer
    vx_size inception_b2_1x7_3_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x7_3_relu;
    inception_b2_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_relu);
    vx_enum inception_b2_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_3_relu_param_a = 0;
    vx_float32 inception_b2_1x7_3_relu_param_b = 0;
    vx_node inception_b2_1x7_3_relu_node;
    inception_b2_1x7_3_relu_node = vxActivationLayer(graph, inception_b2_1x7_3_scale, inception_b2_1x7_3_relu_mode, inception_b2_1x7_3_relu_param_a, inception_b2_1x7_3_relu_param_b, inception_b2_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_3_relu_node));

    // inception_b2_pool_ave Layer
    vx_size inception_b2_pool_ave_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b2_pool_ave;
    inception_b2_pool_ave = vxCreateVirtualTensor(graph,4, inception_b2_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_pool_ave);
    vx_enum inception_b2_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b2_pool_ave_kernel_w = 3;
    vx_size inception_b2_pool_ave_kernel_h = 3;
    vx_size inception_b2_pool_ave_pad_w = 1;
    vx_size inception_b2_pool_ave_pad_h = 1;
    vx_enum inception_b2_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b2_pool_ave_node;
    inception_b2_pool_ave_node = vxPoolingLayer(graph, inception_b1_concat_inception_b1_concat_0_split_3, inception_b2_pool_ave_type, inception_b2_pool_ave_kernel_w, inception_b2_pool_ave_kernel_h, inception_b2_pool_ave_pad_w, inception_b2_pool_ave_pad_h, inception_b2_pool_ave_roundPolicy, inception_b2_pool_ave );
    ERROR_CHECK_OBJECT(inception_b2_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_pool_ave_node));

    // inception_b2_1x1 Layer
    vx_size inception_b2_1x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x1;
    inception_b2_1x1 = vxCreateVirtualTensor(graph,4, inception_b2_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1);
    vx_size inception_b2_1x1_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b2_1x1_W;
    inception_b2_1x1_W = vxCreateTensor(context,4, inception_b2_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_W, dataFolder + "/weights/inception_b2_1x1.f32"));
    vx_nn_convolution_params_t inception_b2_1x1_params;
    inception_b2_1x1_params.padding_x = 0;
    inception_b2_1x1_params.padding_y = 0;
    inception_b2_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x1_params.dilation_x = 0;
    inception_b2_1x1_params.dilation_y = 0;
    vx_node inception_b2_1x1_node;
    inception_b2_1x1_node = vxConvolutionLayer(graph, inception_b2_pool_ave, inception_b2_1x1_W, NULL, &inception_b2_1x1_params, sizeof(inception_b2_1x1_params ), inception_b2_1x1);
    ERROR_CHECK_OBJECT(inception_b2_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_node));

    // inception_b2_1x1_bn Layer
    vx_size inception_b2_1x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x1_scale;
    inception_b2_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_scale);
    vx_size inception_b2_1x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_1x1_bn_eps = 0.001;
    vx_tensor inception_b2_1x1_bn_W;
    inception_b2_1x1_bn_W = vxCreateTensor(context,1, inception_b2_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_bn_W, dataFolder + "/weights/inception_b2_1x1_bn.f32"));
    vx_size inception_b2_1x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x1_bn_B;
    inception_b2_1x1_bn_B = vxCreateTensor(context,1, inception_b2_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_bn_B, dataFolder + "/bias/inception_b2_1x1_bn.f32"));
    vx_size inception_b2_1x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_1x1_scale_W;
    inception_b2_1x1_scale_W = vxCreateTensor(context,1, inception_b2_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_scale_W, dataFolder + "/weights/inception_b2_1x1_scale.f32"));
    vx_size inception_b2_1x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x1_scale_B;
    inception_b2_1x1_scale_B = vxCreateTensor(context,1, inception_b2_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_scale_B, dataFolder + "/bias/inception_b2_1x1_scale.f32"));
    vx_node inception_b2_1x1_bn_node;
    inception_b2_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x1, inception_b2_1x1_bn_W, inception_b2_1x1_bn_B, inception_b2_1x1_scale_W, inception_b2_1x1_scale_B, inception_b2_1x1_bn_eps, inception_b2_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_bn_node));

    // inception_b2_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x1_relu Layer
    vx_size inception_b2_1x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x1_relu;
    inception_b2_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_relu);
    vx_enum inception_b2_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x1_relu_param_a = 0;
    vx_float32 inception_b2_1x1_relu_param_b = 0;
    vx_node inception_b2_1x1_relu_node;
    inception_b2_1x1_relu_node = vxActivationLayer(graph, inception_b2_1x1_scale, inception_b2_1x1_relu_mode, inception_b2_1x1_relu_param_a, inception_b2_1x1_relu_param_b, inception_b2_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_relu_node));

    // inception_b2_concat Layer
    vx_size inception_b2_concat_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b2_concat;
    inception_b2_concat = vxCreateVirtualTensor(graph,4, inception_b2_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_concat);
    vx_node inception_b2_concat_node;
    inception_b2_concat_node = vxConcatLayer(graph, inception_b2_concat, inception_b2_1x1_2_relu, inception_b2_7x1_relu, inception_b2_1x7_3_relu, inception_b2_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b2_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_concat_node));

    // inception_b2_concat_inception_b2_concat_0_split_0 Layer
    vx_size inception_b2_concat_inception_b2_concat_0_split_0_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b2_concat_inception_b2_concat_0_split_0;
    inception_b2_concat_inception_b2_concat_0_split_0 = vxCreateVirtualTensor(graph,4, inception_b2_concat_inception_b2_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_0);
    vx_node inception_b2_concat_inception_b2_concat_0_split_0_node;
    inception_b2_concat_inception_b2_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_b2_concat, (vx_reference)inception_b2_concat_inception_b2_concat_0_split_0);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_concat_inception_b2_concat_0_split_0_node));

    // inception_b2_concat_inception_b2_concat_0_split_1 Layer
    vx_size inception_b2_concat_inception_b2_concat_0_split_1_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b2_concat_inception_b2_concat_0_split_1;
    inception_b2_concat_inception_b2_concat_0_split_1 = vxCreateVirtualTensor(graph,4, inception_b2_concat_inception_b2_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_1);
    vx_node inception_b2_concat_inception_b2_concat_0_split_1_node;
    inception_b2_concat_inception_b2_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_b2_concat, (vx_reference)inception_b2_concat_inception_b2_concat_0_split_1);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_concat_inception_b2_concat_0_split_1_node));

    // inception_b2_concat_inception_b2_concat_0_split_2 Layer
    vx_size inception_b2_concat_inception_b2_concat_0_split_2_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b2_concat_inception_b2_concat_0_split_2;
    inception_b2_concat_inception_b2_concat_0_split_2 = vxCreateVirtualTensor(graph,4, inception_b2_concat_inception_b2_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_2);
    vx_node inception_b2_concat_inception_b2_concat_0_split_2_node;
    inception_b2_concat_inception_b2_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_b2_concat, (vx_reference)inception_b2_concat_inception_b2_concat_0_split_2);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_concat_inception_b2_concat_0_split_2_node));

    // inception_b2_concat_inception_b2_concat_0_split_3 Layer
    vx_size inception_b2_concat_inception_b2_concat_0_split_3_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b2_concat_inception_b2_concat_0_split_3;
    inception_b2_concat_inception_b2_concat_0_split_3 = vxCreateVirtualTensor(graph,4, inception_b2_concat_inception_b2_concat_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_3);
    vx_node inception_b2_concat_inception_b2_concat_0_split_3_node;
    inception_b2_concat_inception_b2_concat_0_split_3_node = vxCopyNode( graph, (vx_reference)inception_b2_concat, (vx_reference)inception_b2_concat_inception_b2_concat_0_split_3);
    ERROR_CHECK_OBJECT(inception_b2_concat_inception_b2_concat_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_concat_inception_b2_concat_0_split_3_node));

    // inception_b3_1x1_2 Layer
    vx_size inception_b3_1x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x1_2;
    inception_b3_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b3_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2);
    vx_size inception_b3_1x1_2_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b3_1x1_2_W;
    inception_b3_1x1_2_W = vxCreateTensor(context,4, inception_b3_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_W, dataFolder + "/weights/inception_b3_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b3_1x1_2_params;
    inception_b3_1x1_2_params.padding_x = 0;
    inception_b3_1x1_2_params.padding_y = 0;
    inception_b3_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x1_2_params.dilation_x = 0;
    inception_b3_1x1_2_params.dilation_y = 0;
    vx_node inception_b3_1x1_2_node;
    inception_b3_1x1_2_node = vxConvolutionLayer(graph, inception_b2_concat_inception_b2_concat_0_split_0, inception_b3_1x1_2_W, NULL, &inception_b3_1x1_2_params, sizeof(inception_b3_1x1_2_params ), inception_b3_1x1_2);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_2_node));

    // inception_b3_1x1_2_bn Layer
    vx_size inception_b3_1x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x1_2_scale;
    inception_b3_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_scale);
    vx_size inception_b3_1x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b3_1x1_2_bn_W;
    inception_b3_1x1_2_bn_W = vxCreateTensor(context,1, inception_b3_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_bn_W, dataFolder + "/weights/inception_b3_1x1_2_bn.f32"));
    vx_size inception_b3_1x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x1_2_bn_B;
    inception_b3_1x1_2_bn_B = vxCreateTensor(context,1, inception_b3_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_bn_B, dataFolder + "/bias/inception_b3_1x1_2_bn.f32"));
    vx_size inception_b3_1x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_1x1_2_scale_W;
    inception_b3_1x1_2_scale_W = vxCreateTensor(context,1, inception_b3_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_scale_W, dataFolder + "/weights/inception_b3_1x1_2_scale.f32"));
    vx_size inception_b3_1x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x1_2_scale_B;
    inception_b3_1x1_2_scale_B = vxCreateTensor(context,1, inception_b3_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_scale_B, dataFolder + "/bias/inception_b3_1x1_2_scale.f32"));
    vx_node inception_b3_1x1_2_bn_node;
    inception_b3_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x1_2, inception_b3_1x1_2_bn_W, inception_b3_1x1_2_bn_B, inception_b3_1x1_2_scale_W, inception_b3_1x1_2_scale_B, inception_b3_1x1_2_bn_eps, inception_b3_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_2_bn_node));

    // inception_b3_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x1_2_relu Layer
    vx_size inception_b3_1x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x1_2_relu;
    inception_b3_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_relu);
    vx_enum inception_b3_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x1_2_relu_param_a = 0;
    vx_float32 inception_b3_1x1_2_relu_param_b = 0;
    vx_node inception_b3_1x1_2_relu_node;
    inception_b3_1x1_2_relu_node = vxActivationLayer(graph, inception_b3_1x1_2_scale, inception_b3_1x1_2_relu_mode, inception_b3_1x1_2_relu_param_a, inception_b3_1x1_2_relu_param_b, inception_b3_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_2_relu_node));

    // inception_b3_1x7_reduce Layer
    vx_size inception_b3_1x7_reduce_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_reduce;
    inception_b3_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b3_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce);
    vx_size inception_b3_1x7_reduce_W_dims[4] = { 1, 1, 768, 160 };
    vx_tensor inception_b3_1x7_reduce_W;
    inception_b3_1x7_reduce_W = vxCreateTensor(context,4, inception_b3_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_W, dataFolder + "/weights/inception_b3_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_reduce_params;
    inception_b3_1x7_reduce_params.padding_x = 0;
    inception_b3_1x7_reduce_params.padding_y = 0;
    inception_b3_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_reduce_params.dilation_x = 0;
    inception_b3_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b3_1x7_reduce_node;
    inception_b3_1x7_reduce_node = vxConvolutionLayer(graph, inception_b2_concat_inception_b2_concat_0_split_1, inception_b3_1x7_reduce_W, NULL, &inception_b3_1x7_reduce_params, sizeof(inception_b3_1x7_reduce_params ), inception_b3_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_reduce_node));

    // inception_b3_1x7_reduce_bn Layer
    vx_size inception_b3_1x7_reduce_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_reduce_scale;
    inception_b3_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_scale);
    vx_size inception_b3_1x7_reduce_bn_W_dims[1] = { 160 };
    vx_float32 inception_b3_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_reduce_bn_W;
    inception_b3_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b3_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_bn_W, dataFolder + "/weights/inception_b3_1x7_reduce_bn.f32"));
    vx_size inception_b3_1x7_reduce_bn_B_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_reduce_bn_B;
    inception_b3_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b3_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_bn_B, dataFolder + "/bias/inception_b3_1x7_reduce_bn.f32"));
    vx_size inception_b3_1x7_reduce_scale_W_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_reduce_scale_W;
    inception_b3_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b3_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_scale_W, dataFolder + "/weights/inception_b3_1x7_reduce_scale.f32"));
    vx_size inception_b3_1x7_reduce_scale_B_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_reduce_scale_B;
    inception_b3_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b3_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_scale_B, dataFolder + "/bias/inception_b3_1x7_reduce_scale.f32"));
    vx_node inception_b3_1x7_reduce_bn_node;
    inception_b3_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7_reduce, inception_b3_1x7_reduce_bn_W, inception_b3_1x7_reduce_bn_B, inception_b3_1x7_reduce_scale_W, inception_b3_1x7_reduce_scale_B, inception_b3_1x7_reduce_bn_eps, inception_b3_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_reduce_bn_node));

    // inception_b3_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_reduce_relu Layer
    vx_size inception_b3_1x7_reduce_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_reduce_relu;
    inception_b3_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_relu);
    vx_enum inception_b3_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b3_1x7_reduce_relu_param_b = 0;
    vx_node inception_b3_1x7_reduce_relu_node;
    inception_b3_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b3_1x7_reduce_scale, inception_b3_1x7_reduce_relu_mode, inception_b3_1x7_reduce_relu_param_a, inception_b3_1x7_reduce_relu_param_b, inception_b3_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_reduce_relu_node));

    // inception_b3_1x7 Layer
    vx_size inception_b3_1x7_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7;
    inception_b3_1x7 = vxCreateVirtualTensor(graph,4, inception_b3_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7);
    vx_size inception_b3_1x7_W_dims[4] = { 7, 1, 160, 160 };
    vx_tensor inception_b3_1x7_W;
    inception_b3_1x7_W = vxCreateTensor(context,4, inception_b3_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_W, dataFolder + "/weights/inception_b3_1x7.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_params;
    inception_b3_1x7_params.padding_x = 3;
    inception_b3_1x7_params.padding_y = 0;
    inception_b3_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_params.dilation_x = 0;
    inception_b3_1x7_params.dilation_y = 0;
    vx_node inception_b3_1x7_node;
    inception_b3_1x7_node = vxConvolutionLayer(graph, inception_b3_1x7_reduce_relu, inception_b3_1x7_W, NULL, &inception_b3_1x7_params, sizeof(inception_b3_1x7_params ), inception_b3_1x7);
    ERROR_CHECK_OBJECT(inception_b3_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_node));

    // inception_b3_1x7_bn Layer
    vx_size inception_b3_1x7_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_scale;
    inception_b3_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_scale);
    vx_size inception_b3_1x7_bn_W_dims[1] = { 160 };
    vx_float32 inception_b3_1x7_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_bn_W;
    inception_b3_1x7_bn_W = vxCreateTensor(context,1, inception_b3_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_bn_W, dataFolder + "/weights/inception_b3_1x7_bn.f32"));
    vx_size inception_b3_1x7_bn_B_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_bn_B;
    inception_b3_1x7_bn_B = vxCreateTensor(context,1, inception_b3_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_bn_B, dataFolder + "/bias/inception_b3_1x7_bn.f32"));
    vx_size inception_b3_1x7_scale_W_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_scale_W;
    inception_b3_1x7_scale_W = vxCreateTensor(context,1, inception_b3_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_scale_W, dataFolder + "/weights/inception_b3_1x7_scale.f32"));
    vx_size inception_b3_1x7_scale_B_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_scale_B;
    inception_b3_1x7_scale_B = vxCreateTensor(context,1, inception_b3_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_scale_B, dataFolder + "/bias/inception_b3_1x7_scale.f32"));
    vx_node inception_b3_1x7_bn_node;
    inception_b3_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7, inception_b3_1x7_bn_W, inception_b3_1x7_bn_B, inception_b3_1x7_scale_W, inception_b3_1x7_scale_B, inception_b3_1x7_bn_eps, inception_b3_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_bn_node));

    // inception_b3_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_relu Layer
    vx_size inception_b3_1x7_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_relu;
    inception_b3_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_relu);
    vx_enum inception_b3_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_relu_param_a = 0;
    vx_float32 inception_b3_1x7_relu_param_b = 0;
    vx_node inception_b3_1x7_relu_node;
    inception_b3_1x7_relu_node = vxActivationLayer(graph, inception_b3_1x7_scale, inception_b3_1x7_relu_mode, inception_b3_1x7_relu_param_a, inception_b3_1x7_relu_param_b, inception_b3_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_relu_node));

    // inception_b3_7x1 Layer
    vx_size inception_b3_7x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1;
    inception_b3_7x1 = vxCreateVirtualTensor(graph,4, inception_b3_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1);
    vx_size inception_b3_7x1_W_dims[4] = { 1, 7, 160, 192 };
    vx_tensor inception_b3_7x1_W;
    inception_b3_7x1_W = vxCreateTensor(context,4, inception_b3_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_W, dataFolder + "/weights/inception_b3_7x1.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_params;
    inception_b3_7x1_params.padding_x = 0;
    inception_b3_7x1_params.padding_y = 3;
    inception_b3_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_params.dilation_x = 0;
    inception_b3_7x1_params.dilation_y = 0;
    vx_node inception_b3_7x1_node;
    inception_b3_7x1_node = vxConvolutionLayer(graph, inception_b3_1x7_relu, inception_b3_7x1_W, NULL, &inception_b3_7x1_params, sizeof(inception_b3_7x1_params ), inception_b3_7x1);
    ERROR_CHECK_OBJECT(inception_b3_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_node));

    // inception_b3_7x1_bn Layer
    vx_size inception_b3_7x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_scale;
    inception_b3_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_scale);
    vx_size inception_b3_7x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_7x1_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_bn_W;
    inception_b3_7x1_bn_W = vxCreateTensor(context,1, inception_b3_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_bn_W, dataFolder + "/weights/inception_b3_7x1_bn.f32"));
    vx_size inception_b3_7x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_bn_B;
    inception_b3_7x1_bn_B = vxCreateTensor(context,1, inception_b3_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_bn_B, dataFolder + "/bias/inception_b3_7x1_bn.f32"));
    vx_size inception_b3_7x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_scale_W;
    inception_b3_7x1_scale_W = vxCreateTensor(context,1, inception_b3_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_scale_W, dataFolder + "/weights/inception_b3_7x1_scale.f32"));
    vx_size inception_b3_7x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_scale_B;
    inception_b3_7x1_scale_B = vxCreateTensor(context,1, inception_b3_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_scale_B, dataFolder + "/bias/inception_b3_7x1_scale.f32"));
    vx_node inception_b3_7x1_bn_node;
    inception_b3_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1, inception_b3_7x1_bn_W, inception_b3_7x1_bn_B, inception_b3_7x1_scale_W, inception_b3_7x1_scale_B, inception_b3_7x1_bn_eps, inception_b3_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_bn_node));

    // inception_b3_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_relu Layer
    vx_size inception_b3_7x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_relu;
    inception_b3_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_relu);
    vx_enum inception_b3_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_relu_param_a = 0;
    vx_float32 inception_b3_7x1_relu_param_b = 0;
    vx_node inception_b3_7x1_relu_node;
    inception_b3_7x1_relu_node = vxActivationLayer(graph, inception_b3_7x1_scale, inception_b3_7x1_relu_mode, inception_b3_7x1_relu_param_a, inception_b3_7x1_relu_param_b, inception_b3_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_relu_node));

    // inception_b3_7x1_reduce Layer
    vx_size inception_b3_7x1_reduce_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_reduce;
    inception_b3_7x1_reduce = vxCreateVirtualTensor(graph,4, inception_b3_7x1_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce);
    vx_size inception_b3_7x1_reduce_W_dims[4] = { 1, 1, 768, 160 };
    vx_tensor inception_b3_7x1_reduce_W;
    inception_b3_7x1_reduce_W = vxCreateTensor(context,4, inception_b3_7x1_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_reduce_W, dataFolder + "/weights/inception_b3_7x1_reduce.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_reduce_params;
    inception_b3_7x1_reduce_params.padding_x = 0;
    inception_b3_7x1_reduce_params.padding_y = 0;
    inception_b3_7x1_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_reduce_params.dilation_x = 0;
    inception_b3_7x1_reduce_params.dilation_y = 0;
    vx_node inception_b3_7x1_reduce_node;
    inception_b3_7x1_reduce_node = vxConvolutionLayer(graph, inception_b2_concat_inception_b2_concat_0_split_2, inception_b3_7x1_reduce_W, NULL, &inception_b3_7x1_reduce_params, sizeof(inception_b3_7x1_reduce_params ), inception_b3_7x1_reduce);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_reduce_node));

    // inception_b3_7x1_reduce_bn Layer
    vx_size inception_b3_7x1_reduce_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_reduce_scale;
    inception_b3_7x1_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_scale);
    vx_size inception_b3_7x1_reduce_bn_W_dims[1] = { 160 };
    vx_float32 inception_b3_7x1_reduce_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_reduce_bn_W;
    inception_b3_7x1_reduce_bn_W = vxCreateTensor(context,1, inception_b3_7x1_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_reduce_bn_W, dataFolder + "/weights/inception_b3_7x1_reduce_bn.f32"));
    vx_size inception_b3_7x1_reduce_bn_B_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_reduce_bn_B;
    inception_b3_7x1_reduce_bn_B = vxCreateTensor(context,1, inception_b3_7x1_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_reduce_bn_B, dataFolder + "/bias/inception_b3_7x1_reduce_bn.f32"));
    vx_size inception_b3_7x1_reduce_scale_W_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_reduce_scale_W;
    inception_b3_7x1_reduce_scale_W = vxCreateTensor(context,1, inception_b3_7x1_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_reduce_scale_W, dataFolder + "/weights/inception_b3_7x1_reduce_scale.f32"));
    vx_size inception_b3_7x1_reduce_scale_B_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_reduce_scale_B;
    inception_b3_7x1_reduce_scale_B = vxCreateTensor(context,1, inception_b3_7x1_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_reduce_scale_B, dataFolder + "/bias/inception_b3_7x1_reduce_scale.f32"));
    vx_node inception_b3_7x1_reduce_bn_node;
    inception_b3_7x1_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1_reduce, inception_b3_7x1_reduce_bn_W, inception_b3_7x1_reduce_bn_B, inception_b3_7x1_reduce_scale_W, inception_b3_7x1_reduce_scale_B, inception_b3_7x1_reduce_bn_eps, inception_b3_7x1_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_reduce_bn_node));

    // inception_b3_7x1_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_reduce_relu Layer
    vx_size inception_b3_7x1_reduce_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_reduce_relu;
    inception_b3_7x1_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_relu);
    vx_enum inception_b3_7x1_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_reduce_relu_param_a = 0;
    vx_float32 inception_b3_7x1_reduce_relu_param_b = 0;
    vx_node inception_b3_7x1_reduce_relu_node;
    inception_b3_7x1_reduce_relu_node = vxActivationLayer(graph, inception_b3_7x1_reduce_scale, inception_b3_7x1_reduce_relu_mode, inception_b3_7x1_reduce_relu_param_a, inception_b3_7x1_reduce_relu_param_b, inception_b3_7x1_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_reduce_relu_node));

    // inception_b3_7x1_2 Layer
    vx_size inception_b3_7x1_2_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_2;
    inception_b3_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2);
    vx_size inception_b3_7x1_2_W_dims[4] = { 1, 7, 160, 160 };
    vx_tensor inception_b3_7x1_2_W;
    inception_b3_7x1_2_W = vxCreateTensor(context,4, inception_b3_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_W, dataFolder + "/weights/inception_b3_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_2_params;
    inception_b3_7x1_2_params.padding_x = 0;
    inception_b3_7x1_2_params.padding_y = 3;
    inception_b3_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_2_params.dilation_x = 0;
    inception_b3_7x1_2_params.dilation_y = 0;
    vx_node inception_b3_7x1_2_node;
    inception_b3_7x1_2_node = vxConvolutionLayer(graph, inception_b3_7x1_reduce_relu, inception_b3_7x1_2_W, NULL, &inception_b3_7x1_2_params, sizeof(inception_b3_7x1_2_params ), inception_b3_7x1_2);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_node));

    // inception_b3_7x1_2_bn Layer
    vx_size inception_b3_7x1_2_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_2_scale;
    inception_b3_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_scale);
    vx_size inception_b3_7x1_2_bn_W_dims[1] = { 160 };
    vx_float32 inception_b3_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_2_bn_W;
    inception_b3_7x1_2_bn_W = vxCreateTensor(context,1, inception_b3_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_bn_W, dataFolder + "/weights/inception_b3_7x1_2_bn.f32"));
    vx_size inception_b3_7x1_2_bn_B_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_2_bn_B;
    inception_b3_7x1_2_bn_B = vxCreateTensor(context,1, inception_b3_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_bn_B, dataFolder + "/bias/inception_b3_7x1_2_bn.f32"));
    vx_size inception_b3_7x1_2_scale_W_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_2_scale_W;
    inception_b3_7x1_2_scale_W = vxCreateTensor(context,1, inception_b3_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_scale_W, dataFolder + "/weights/inception_b3_7x1_2_scale.f32"));
    vx_size inception_b3_7x1_2_scale_B_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_2_scale_B;
    inception_b3_7x1_2_scale_B = vxCreateTensor(context,1, inception_b3_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_scale_B, dataFolder + "/bias/inception_b3_7x1_2_scale.f32"));
    vx_node inception_b3_7x1_2_bn_node;
    inception_b3_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1_2, inception_b3_7x1_2_bn_W, inception_b3_7x1_2_bn_B, inception_b3_7x1_2_scale_W, inception_b3_7x1_2_scale_B, inception_b3_7x1_2_bn_eps, inception_b3_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_bn_node));

    // inception_b3_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_2_relu Layer
    vx_size inception_b3_7x1_2_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_2_relu;
    inception_b3_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_relu);
    vx_enum inception_b3_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_2_relu_param_a = 0;
    vx_float32 inception_b3_7x1_2_relu_param_b = 0;
    vx_node inception_b3_7x1_2_relu_node;
    inception_b3_7x1_2_relu_node = vxActivationLayer(graph, inception_b3_7x1_2_scale, inception_b3_7x1_2_relu_mode, inception_b3_7x1_2_relu_param_a, inception_b3_7x1_2_relu_param_b, inception_b3_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_relu_node));

    // inception_b3_1x7_2 Layer
    vx_size inception_b3_1x7_2_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_2;
    inception_b3_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b3_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2);
    vx_size inception_b3_1x7_2_W_dims[4] = { 7, 1, 160, 160 };
    vx_tensor inception_b3_1x7_2_W;
    inception_b3_1x7_2_W = vxCreateTensor(context,4, inception_b3_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_W, dataFolder + "/weights/inception_b3_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_2_params;
    inception_b3_1x7_2_params.padding_x = 3;
    inception_b3_1x7_2_params.padding_y = 0;
    inception_b3_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_2_params.dilation_x = 0;
    inception_b3_1x7_2_params.dilation_y = 0;
    vx_node inception_b3_1x7_2_node;
    inception_b3_1x7_2_node = vxConvolutionLayer(graph, inception_b3_7x1_2_relu, inception_b3_1x7_2_W, NULL, &inception_b3_1x7_2_params, sizeof(inception_b3_1x7_2_params ), inception_b3_1x7_2);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_2_node));

    // inception_b3_1x7_2_bn Layer
    vx_size inception_b3_1x7_2_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_2_scale;
    inception_b3_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_scale);
    vx_size inception_b3_1x7_2_bn_W_dims[1] = { 160 };
    vx_float32 inception_b3_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_2_bn_W;
    inception_b3_1x7_2_bn_W = vxCreateTensor(context,1, inception_b3_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_bn_W, dataFolder + "/weights/inception_b3_1x7_2_bn.f32"));
    vx_size inception_b3_1x7_2_bn_B_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_2_bn_B;
    inception_b3_1x7_2_bn_B = vxCreateTensor(context,1, inception_b3_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_bn_B, dataFolder + "/bias/inception_b3_1x7_2_bn.f32"));
    vx_size inception_b3_1x7_2_scale_W_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_2_scale_W;
    inception_b3_1x7_2_scale_W = vxCreateTensor(context,1, inception_b3_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_scale_W, dataFolder + "/weights/inception_b3_1x7_2_scale.f32"));
    vx_size inception_b3_1x7_2_scale_B_dims[1] = { 160 };
    vx_tensor inception_b3_1x7_2_scale_B;
    inception_b3_1x7_2_scale_B = vxCreateTensor(context,1, inception_b3_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_scale_B, dataFolder + "/bias/inception_b3_1x7_2_scale.f32"));
    vx_node inception_b3_1x7_2_bn_node;
    inception_b3_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7_2, inception_b3_1x7_2_bn_W, inception_b3_1x7_2_bn_B, inception_b3_1x7_2_scale_W, inception_b3_1x7_2_scale_B, inception_b3_1x7_2_bn_eps, inception_b3_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_2_bn_node));

    // inception_b3_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_2_relu Layer
    vx_size inception_b3_1x7_2_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_1x7_2_relu;
    inception_b3_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_relu);
    vx_enum inception_b3_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_2_relu_param_a = 0;
    vx_float32 inception_b3_1x7_2_relu_param_b = 0;
    vx_node inception_b3_1x7_2_relu_node;
    inception_b3_1x7_2_relu_node = vxActivationLayer(graph, inception_b3_1x7_2_scale, inception_b3_1x7_2_relu_mode, inception_b3_1x7_2_relu_param_a, inception_b3_1x7_2_relu_param_b, inception_b3_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_2_relu_node));

    // inception_b3_7x1_3 Layer
    vx_size inception_b3_7x1_3_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_3;
    inception_b3_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b3_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3);
    vx_size inception_b3_7x1_3_W_dims[4] = { 1, 7, 160, 160 };
    vx_tensor inception_b3_7x1_3_W;
    inception_b3_7x1_3_W = vxCreateTensor(context,4, inception_b3_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_W, dataFolder + "/weights/inception_b3_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_3_params;
    inception_b3_7x1_3_params.padding_x = 0;
    inception_b3_7x1_3_params.padding_y = 3;
    inception_b3_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_3_params.dilation_x = 0;
    inception_b3_7x1_3_params.dilation_y = 0;
    vx_node inception_b3_7x1_3_node;
    inception_b3_7x1_3_node = vxConvolutionLayer(graph, inception_b3_1x7_2_relu, inception_b3_7x1_3_W, NULL, &inception_b3_7x1_3_params, sizeof(inception_b3_7x1_3_params ), inception_b3_7x1_3);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_3_node));

    // inception_b3_7x1_3_bn Layer
    vx_size inception_b3_7x1_3_scale_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_3_scale;
    inception_b3_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_scale);
    vx_size inception_b3_7x1_3_bn_W_dims[1] = { 160 };
    vx_float32 inception_b3_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_3_bn_W;
    inception_b3_7x1_3_bn_W = vxCreateTensor(context,1, inception_b3_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_bn_W, dataFolder + "/weights/inception_b3_7x1_3_bn.f32"));
    vx_size inception_b3_7x1_3_bn_B_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_3_bn_B;
    inception_b3_7x1_3_bn_B = vxCreateTensor(context,1, inception_b3_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_bn_B, dataFolder + "/bias/inception_b3_7x1_3_bn.f32"));
    vx_size inception_b3_7x1_3_scale_W_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_3_scale_W;
    inception_b3_7x1_3_scale_W = vxCreateTensor(context,1, inception_b3_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_scale_W, dataFolder + "/weights/inception_b3_7x1_3_scale.f32"));
    vx_size inception_b3_7x1_3_scale_B_dims[1] = { 160 };
    vx_tensor inception_b3_7x1_3_scale_B;
    inception_b3_7x1_3_scale_B = vxCreateTensor(context,1, inception_b3_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_scale_B, dataFolder + "/bias/inception_b3_7x1_3_scale.f32"));
    vx_node inception_b3_7x1_3_bn_node;
    inception_b3_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1_3, inception_b3_7x1_3_bn_W, inception_b3_7x1_3_bn_B, inception_b3_7x1_3_scale_W, inception_b3_7x1_3_scale_B, inception_b3_7x1_3_bn_eps, inception_b3_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_3_bn_node));

    // inception_b3_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_3_relu Layer
    vx_size inception_b3_7x1_3_relu_dims[4] = { 17, 17, 160, 1 };
    vx_tensor inception_b3_7x1_3_relu;
    inception_b3_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_relu);
    vx_enum inception_b3_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_3_relu_param_a = 0;
    vx_float32 inception_b3_7x1_3_relu_param_b = 0;
    vx_node inception_b3_7x1_3_relu_node;
    inception_b3_7x1_3_relu_node = vxActivationLayer(graph, inception_b3_7x1_3_scale, inception_b3_7x1_3_relu_mode, inception_b3_7x1_3_relu_param_a, inception_b3_7x1_3_relu_param_b, inception_b3_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_3_relu_node));

    // inception_b3_1x7_3 Layer
    vx_size inception_b3_1x7_3_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x7_3;
    inception_b3_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b3_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3);
    vx_size inception_b3_1x7_3_W_dims[4] = { 7, 1, 160, 192 };
    vx_tensor inception_b3_1x7_3_W;
    inception_b3_1x7_3_W = vxCreateTensor(context,4, inception_b3_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_W, dataFolder + "/weights/inception_b3_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_3_params;
    inception_b3_1x7_3_params.padding_x = 3;
    inception_b3_1x7_3_params.padding_y = 0;
    inception_b3_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_3_params.dilation_x = 0;
    inception_b3_1x7_3_params.dilation_y = 0;
    vx_node inception_b3_1x7_3_node;
    inception_b3_1x7_3_node = vxConvolutionLayer(graph, inception_b3_7x1_3_relu, inception_b3_1x7_3_W, NULL, &inception_b3_1x7_3_params, sizeof(inception_b3_1x7_3_params ), inception_b3_1x7_3);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_3_node));

    // inception_b3_1x7_3_bn Layer
    vx_size inception_b3_1x7_3_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x7_3_scale;
    inception_b3_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_scale);
    vx_size inception_b3_1x7_3_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_3_bn_W;
    inception_b3_1x7_3_bn_W = vxCreateTensor(context,1, inception_b3_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_bn_W, dataFolder + "/weights/inception_b3_1x7_3_bn.f32"));
    vx_size inception_b3_1x7_3_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x7_3_bn_B;
    inception_b3_1x7_3_bn_B = vxCreateTensor(context,1, inception_b3_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_bn_B, dataFolder + "/bias/inception_b3_1x7_3_bn.f32"));
    vx_size inception_b3_1x7_3_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_1x7_3_scale_W;
    inception_b3_1x7_3_scale_W = vxCreateTensor(context,1, inception_b3_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_scale_W, dataFolder + "/weights/inception_b3_1x7_3_scale.f32"));
    vx_size inception_b3_1x7_3_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x7_3_scale_B;
    inception_b3_1x7_3_scale_B = vxCreateTensor(context,1, inception_b3_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_scale_B, dataFolder + "/bias/inception_b3_1x7_3_scale.f32"));
    vx_node inception_b3_1x7_3_bn_node;
    inception_b3_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7_3, inception_b3_1x7_3_bn_W, inception_b3_1x7_3_bn_B, inception_b3_1x7_3_scale_W, inception_b3_1x7_3_scale_B, inception_b3_1x7_3_bn_eps, inception_b3_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_3_bn_node));

    // inception_b3_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_3_relu Layer
    vx_size inception_b3_1x7_3_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x7_3_relu;
    inception_b3_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_relu);
    vx_enum inception_b3_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_3_relu_param_a = 0;
    vx_float32 inception_b3_1x7_3_relu_param_b = 0;
    vx_node inception_b3_1x7_3_relu_node;
    inception_b3_1x7_3_relu_node = vxActivationLayer(graph, inception_b3_1x7_3_scale, inception_b3_1x7_3_relu_mode, inception_b3_1x7_3_relu_param_a, inception_b3_1x7_3_relu_param_b, inception_b3_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_3_relu_node));

    // inception_b3_pool_ave Layer
    vx_size inception_b3_pool_ave_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b3_pool_ave;
    inception_b3_pool_ave = vxCreateVirtualTensor(graph,4, inception_b3_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_pool_ave);
    vx_enum inception_b3_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b3_pool_ave_kernel_w = 3;
    vx_size inception_b3_pool_ave_kernel_h = 3;
    vx_size inception_b3_pool_ave_pad_w = 1;
    vx_size inception_b3_pool_ave_pad_h = 1;
    vx_enum inception_b3_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b3_pool_ave_node;
    inception_b3_pool_ave_node = vxPoolingLayer(graph, inception_b2_concat_inception_b2_concat_0_split_3, inception_b3_pool_ave_type, inception_b3_pool_ave_kernel_w, inception_b3_pool_ave_kernel_h, inception_b3_pool_ave_pad_w, inception_b3_pool_ave_pad_h, inception_b3_pool_ave_roundPolicy, inception_b3_pool_ave );
    ERROR_CHECK_OBJECT(inception_b3_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_pool_ave_node));

    // inception_b3_1x1 Layer
    vx_size inception_b3_1x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x1;
    inception_b3_1x1 = vxCreateVirtualTensor(graph,4, inception_b3_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1);
    vx_size inception_b3_1x1_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b3_1x1_W;
    inception_b3_1x1_W = vxCreateTensor(context,4, inception_b3_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_W, dataFolder + "/weights/inception_b3_1x1.f32"));
    vx_nn_convolution_params_t inception_b3_1x1_params;
    inception_b3_1x1_params.padding_x = 0;
    inception_b3_1x1_params.padding_y = 0;
    inception_b3_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x1_params.dilation_x = 0;
    inception_b3_1x1_params.dilation_y = 0;
    vx_node inception_b3_1x1_node;
    inception_b3_1x1_node = vxConvolutionLayer(graph, inception_b3_pool_ave, inception_b3_1x1_W, NULL, &inception_b3_1x1_params, sizeof(inception_b3_1x1_params ), inception_b3_1x1);
    ERROR_CHECK_OBJECT(inception_b3_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_node));

    // inception_b3_1x1_bn Layer
    vx_size inception_b3_1x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x1_scale;
    inception_b3_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_scale);
    vx_size inception_b3_1x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_1x1_bn_eps = 0.001;
    vx_tensor inception_b3_1x1_bn_W;
    inception_b3_1x1_bn_W = vxCreateTensor(context,1, inception_b3_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_bn_W, dataFolder + "/weights/inception_b3_1x1_bn.f32"));
    vx_size inception_b3_1x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x1_bn_B;
    inception_b3_1x1_bn_B = vxCreateTensor(context,1, inception_b3_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_bn_B, dataFolder + "/bias/inception_b3_1x1_bn.f32"));
    vx_size inception_b3_1x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_1x1_scale_W;
    inception_b3_1x1_scale_W = vxCreateTensor(context,1, inception_b3_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_scale_W, dataFolder + "/weights/inception_b3_1x1_scale.f32"));
    vx_size inception_b3_1x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x1_scale_B;
    inception_b3_1x1_scale_B = vxCreateTensor(context,1, inception_b3_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_scale_B, dataFolder + "/bias/inception_b3_1x1_scale.f32"));
    vx_node inception_b3_1x1_bn_node;
    inception_b3_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x1, inception_b3_1x1_bn_W, inception_b3_1x1_bn_B, inception_b3_1x1_scale_W, inception_b3_1x1_scale_B, inception_b3_1x1_bn_eps, inception_b3_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_bn_node));

    // inception_b3_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x1_relu Layer
    vx_size inception_b3_1x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x1_relu;
    inception_b3_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_relu);
    vx_enum inception_b3_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x1_relu_param_a = 0;
    vx_float32 inception_b3_1x1_relu_param_b = 0;
    vx_node inception_b3_1x1_relu_node;
    inception_b3_1x1_relu_node = vxActivationLayer(graph, inception_b3_1x1_scale, inception_b3_1x1_relu_mode, inception_b3_1x1_relu_param_a, inception_b3_1x1_relu_param_b, inception_b3_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_relu_node));

    // inception_b3_concat Layer
    vx_size inception_b3_concat_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b3_concat;
    inception_b3_concat = vxCreateVirtualTensor(graph,4, inception_b3_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_concat);
    vx_node inception_b3_concat_node;
    inception_b3_concat_node = vxConcatLayer(graph, inception_b3_concat, inception_b3_1x1_2_relu, inception_b3_7x1_relu, inception_b3_1x7_3_relu, inception_b3_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b3_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_concat_node));

    // inception_b3_concat_inception_b3_concat_0_split_0 Layer
    vx_size inception_b3_concat_inception_b3_concat_0_split_0_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b3_concat_inception_b3_concat_0_split_0;
    inception_b3_concat_inception_b3_concat_0_split_0 = vxCreateVirtualTensor(graph,4, inception_b3_concat_inception_b3_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_0);
    vx_node inception_b3_concat_inception_b3_concat_0_split_0_node;
    inception_b3_concat_inception_b3_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_b3_concat, (vx_reference)inception_b3_concat_inception_b3_concat_0_split_0);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_concat_inception_b3_concat_0_split_0_node));

    // inception_b3_concat_inception_b3_concat_0_split_1 Layer
    vx_size inception_b3_concat_inception_b3_concat_0_split_1_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b3_concat_inception_b3_concat_0_split_1;
    inception_b3_concat_inception_b3_concat_0_split_1 = vxCreateVirtualTensor(graph,4, inception_b3_concat_inception_b3_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_1);
    vx_node inception_b3_concat_inception_b3_concat_0_split_1_node;
    inception_b3_concat_inception_b3_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_b3_concat, (vx_reference)inception_b3_concat_inception_b3_concat_0_split_1);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_concat_inception_b3_concat_0_split_1_node));

    // inception_b3_concat_inception_b3_concat_0_split_2 Layer
    vx_size inception_b3_concat_inception_b3_concat_0_split_2_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b3_concat_inception_b3_concat_0_split_2;
    inception_b3_concat_inception_b3_concat_0_split_2 = vxCreateVirtualTensor(graph,4, inception_b3_concat_inception_b3_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_2);
    vx_node inception_b3_concat_inception_b3_concat_0_split_2_node;
    inception_b3_concat_inception_b3_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_b3_concat, (vx_reference)inception_b3_concat_inception_b3_concat_0_split_2);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_concat_inception_b3_concat_0_split_2_node));

    // inception_b3_concat_inception_b3_concat_0_split_3 Layer
    vx_size inception_b3_concat_inception_b3_concat_0_split_3_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b3_concat_inception_b3_concat_0_split_3;
    inception_b3_concat_inception_b3_concat_0_split_3 = vxCreateVirtualTensor(graph,4, inception_b3_concat_inception_b3_concat_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_3);
    vx_node inception_b3_concat_inception_b3_concat_0_split_3_node;
    inception_b3_concat_inception_b3_concat_0_split_3_node = vxCopyNode( graph, (vx_reference)inception_b3_concat, (vx_reference)inception_b3_concat_inception_b3_concat_0_split_3);
    ERROR_CHECK_OBJECT(inception_b3_concat_inception_b3_concat_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_concat_inception_b3_concat_0_split_3_node));

    // inception_b4_1x1_2 Layer
    vx_size inception_b4_1x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x1_2;
    inception_b4_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b4_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2);
    vx_size inception_b4_1x1_2_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b4_1x1_2_W;
    inception_b4_1x1_2_W = vxCreateTensor(context,4, inception_b4_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_W, dataFolder + "/weights/inception_b4_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b4_1x1_2_params;
    inception_b4_1x1_2_params.padding_x = 0;
    inception_b4_1x1_2_params.padding_y = 0;
    inception_b4_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x1_2_params.dilation_x = 0;
    inception_b4_1x1_2_params.dilation_y = 0;
    vx_node inception_b4_1x1_2_node;
    inception_b4_1x1_2_node = vxConvolutionLayer(graph, inception_b3_concat_inception_b3_concat_0_split_0, inception_b4_1x1_2_W, NULL, &inception_b4_1x1_2_params, sizeof(inception_b4_1x1_2_params ), inception_b4_1x1_2);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_2_node));

    // inception_b4_1x1_2_bn Layer
    vx_size inception_b4_1x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x1_2_scale;
    inception_b4_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_scale);
    vx_size inception_b4_1x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b4_1x1_2_bn_W;
    inception_b4_1x1_2_bn_W = vxCreateTensor(context,1, inception_b4_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_bn_W, dataFolder + "/weights/inception_b4_1x1_2_bn.f32"));
    vx_size inception_b4_1x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x1_2_bn_B;
    inception_b4_1x1_2_bn_B = vxCreateTensor(context,1, inception_b4_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_bn_B, dataFolder + "/bias/inception_b4_1x1_2_bn.f32"));
    vx_size inception_b4_1x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x1_2_scale_W;
    inception_b4_1x1_2_scale_W = vxCreateTensor(context,1, inception_b4_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_scale_W, dataFolder + "/weights/inception_b4_1x1_2_scale.f32"));
    vx_size inception_b4_1x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x1_2_scale_B;
    inception_b4_1x1_2_scale_B = vxCreateTensor(context,1, inception_b4_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_scale_B, dataFolder + "/bias/inception_b4_1x1_2_scale.f32"));
    vx_node inception_b4_1x1_2_bn_node;
    inception_b4_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x1_2, inception_b4_1x1_2_bn_W, inception_b4_1x1_2_bn_B, inception_b4_1x1_2_scale_W, inception_b4_1x1_2_scale_B, inception_b4_1x1_2_bn_eps, inception_b4_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_2_bn_node));

    // inception_b4_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x1_2_relu Layer
    vx_size inception_b4_1x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x1_2_relu;
    inception_b4_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_relu);
    vx_enum inception_b4_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x1_2_relu_param_a = 0;
    vx_float32 inception_b4_1x1_2_relu_param_b = 0;
    vx_node inception_b4_1x1_2_relu_node;
    inception_b4_1x1_2_relu_node = vxActivationLayer(graph, inception_b4_1x1_2_scale, inception_b4_1x1_2_relu_mode, inception_b4_1x1_2_relu_param_a, inception_b4_1x1_2_relu_param_b, inception_b4_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_2_relu_node));

    // inception_b4_1x7_reduce Layer
    vx_size inception_b4_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_reduce;
    inception_b4_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b4_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce);
    vx_size inception_b4_1x7_reduce_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b4_1x7_reduce_W;
    inception_b4_1x7_reduce_W = vxCreateTensor(context,4, inception_b4_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_W, dataFolder + "/weights/inception_b4_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_reduce_params;
    inception_b4_1x7_reduce_params.padding_x = 0;
    inception_b4_1x7_reduce_params.padding_y = 0;
    inception_b4_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_reduce_params.dilation_x = 0;
    inception_b4_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b4_1x7_reduce_node;
    inception_b4_1x7_reduce_node = vxConvolutionLayer(graph, inception_b3_concat_inception_b3_concat_0_split_1, inception_b4_1x7_reduce_W, NULL, &inception_b4_1x7_reduce_params, sizeof(inception_b4_1x7_reduce_params ), inception_b4_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_reduce_node));

    // inception_b4_1x7_reduce_bn Layer
    vx_size inception_b4_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_reduce_scale;
    inception_b4_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_scale);
    vx_size inception_b4_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_reduce_bn_W;
    inception_b4_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b4_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_bn_W, dataFolder + "/weights/inception_b4_1x7_reduce_bn.f32"));
    vx_size inception_b4_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_reduce_bn_B;
    inception_b4_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b4_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_bn_B, dataFolder + "/bias/inception_b4_1x7_reduce_bn.f32"));
    vx_size inception_b4_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_reduce_scale_W;
    inception_b4_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b4_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_scale_W, dataFolder + "/weights/inception_b4_1x7_reduce_scale.f32"));
    vx_size inception_b4_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_reduce_scale_B;
    inception_b4_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b4_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_scale_B, dataFolder + "/bias/inception_b4_1x7_reduce_scale.f32"));
    vx_node inception_b4_1x7_reduce_bn_node;
    inception_b4_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7_reduce, inception_b4_1x7_reduce_bn_W, inception_b4_1x7_reduce_bn_B, inception_b4_1x7_reduce_scale_W, inception_b4_1x7_reduce_scale_B, inception_b4_1x7_reduce_bn_eps, inception_b4_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_reduce_bn_node));

    // inception_b4_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_reduce_relu Layer
    vx_size inception_b4_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_reduce_relu;
    inception_b4_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_relu);
    vx_enum inception_b4_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b4_1x7_reduce_relu_param_b = 0;
    vx_node inception_b4_1x7_reduce_relu_node;
    inception_b4_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b4_1x7_reduce_scale, inception_b4_1x7_reduce_relu_mode, inception_b4_1x7_reduce_relu_param_a, inception_b4_1x7_reduce_relu_param_b, inception_b4_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_reduce_relu_node));

    // inception_b4_1x7 Layer
    vx_size inception_b4_1x7_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7;
    inception_b4_1x7 = vxCreateVirtualTensor(graph,4, inception_b4_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7);
    vx_size inception_b4_1x7_W_dims[4] = { 7, 1, 192, 192 };
    vx_tensor inception_b4_1x7_W;
    inception_b4_1x7_W = vxCreateTensor(context,4, inception_b4_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_W, dataFolder + "/weights/inception_b4_1x7.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_params;
    inception_b4_1x7_params.padding_x = 3;
    inception_b4_1x7_params.padding_y = 0;
    inception_b4_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_params.dilation_x = 0;
    inception_b4_1x7_params.dilation_y = 0;
    vx_node inception_b4_1x7_node;
    inception_b4_1x7_node = vxConvolutionLayer(graph, inception_b4_1x7_reduce_relu, inception_b4_1x7_W, NULL, &inception_b4_1x7_params, sizeof(inception_b4_1x7_params ), inception_b4_1x7);
    ERROR_CHECK_OBJECT(inception_b4_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_node));

    // inception_b4_1x7_bn Layer
    vx_size inception_b4_1x7_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_scale;
    inception_b4_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_scale);
    vx_size inception_b4_1x7_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x7_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_bn_W;
    inception_b4_1x7_bn_W = vxCreateTensor(context,1, inception_b4_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_bn_W, dataFolder + "/weights/inception_b4_1x7_bn.f32"));
    vx_size inception_b4_1x7_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_bn_B;
    inception_b4_1x7_bn_B = vxCreateTensor(context,1, inception_b4_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_bn_B, dataFolder + "/bias/inception_b4_1x7_bn.f32"));
    vx_size inception_b4_1x7_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_scale_W;
    inception_b4_1x7_scale_W = vxCreateTensor(context,1, inception_b4_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_scale_W, dataFolder + "/weights/inception_b4_1x7_scale.f32"));
    vx_size inception_b4_1x7_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_scale_B;
    inception_b4_1x7_scale_B = vxCreateTensor(context,1, inception_b4_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_scale_B, dataFolder + "/bias/inception_b4_1x7_scale.f32"));
    vx_node inception_b4_1x7_bn_node;
    inception_b4_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7, inception_b4_1x7_bn_W, inception_b4_1x7_bn_B, inception_b4_1x7_scale_W, inception_b4_1x7_scale_B, inception_b4_1x7_bn_eps, inception_b4_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_bn_node));

    // inception_b4_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_relu Layer
    vx_size inception_b4_1x7_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_relu;
    inception_b4_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_relu);
    vx_enum inception_b4_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_relu_param_a = 0;
    vx_float32 inception_b4_1x7_relu_param_b = 0;
    vx_node inception_b4_1x7_relu_node;
    inception_b4_1x7_relu_node = vxActivationLayer(graph, inception_b4_1x7_scale, inception_b4_1x7_relu_mode, inception_b4_1x7_relu_param_a, inception_b4_1x7_relu_param_b, inception_b4_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_relu_node));

    // inception_b4_7x1 Layer
    vx_size inception_b4_7x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1;
    inception_b4_7x1 = vxCreateVirtualTensor(graph,4, inception_b4_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1);
    vx_size inception_b4_7x1_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b4_7x1_W;
    inception_b4_7x1_W = vxCreateTensor(context,4, inception_b4_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_W, dataFolder + "/weights/inception_b4_7x1.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_params;
    inception_b4_7x1_params.padding_x = 0;
    inception_b4_7x1_params.padding_y = 3;
    inception_b4_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_params.dilation_x = 0;
    inception_b4_7x1_params.dilation_y = 0;
    vx_node inception_b4_7x1_node;
    inception_b4_7x1_node = vxConvolutionLayer(graph, inception_b4_1x7_relu, inception_b4_7x1_W, NULL, &inception_b4_7x1_params, sizeof(inception_b4_7x1_params ), inception_b4_7x1);
    ERROR_CHECK_OBJECT(inception_b4_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_node));

    // inception_b4_7x1_bn Layer
    vx_size inception_b4_7x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_scale;
    inception_b4_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_scale);
    vx_size inception_b4_7x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_7x1_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_bn_W;
    inception_b4_7x1_bn_W = vxCreateTensor(context,1, inception_b4_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_bn_W, dataFolder + "/weights/inception_b4_7x1_bn.f32"));
    vx_size inception_b4_7x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_bn_B;
    inception_b4_7x1_bn_B = vxCreateTensor(context,1, inception_b4_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_bn_B, dataFolder + "/bias/inception_b4_7x1_bn.f32"));
    vx_size inception_b4_7x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_scale_W;
    inception_b4_7x1_scale_W = vxCreateTensor(context,1, inception_b4_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_scale_W, dataFolder + "/weights/inception_b4_7x1_scale.f32"));
    vx_size inception_b4_7x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_scale_B;
    inception_b4_7x1_scale_B = vxCreateTensor(context,1, inception_b4_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_scale_B, dataFolder + "/bias/inception_b4_7x1_scale.f32"));
    vx_node inception_b4_7x1_bn_node;
    inception_b4_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1, inception_b4_7x1_bn_W, inception_b4_7x1_bn_B, inception_b4_7x1_scale_W, inception_b4_7x1_scale_B, inception_b4_7x1_bn_eps, inception_b4_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_bn_node));

    // inception_b4_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_relu Layer
    vx_size inception_b4_7x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_relu;
    inception_b4_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_relu);
    vx_enum inception_b4_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_relu_param_a = 0;
    vx_float32 inception_b4_7x1_relu_param_b = 0;
    vx_node inception_b4_7x1_relu_node;
    inception_b4_7x1_relu_node = vxActivationLayer(graph, inception_b4_7x1_scale, inception_b4_7x1_relu_mode, inception_b4_7x1_relu_param_a, inception_b4_7x1_relu_param_b, inception_b4_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_relu_node));

    // inception_b4_7x1_reduce Layer
    vx_size inception_b4_7x1_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_reduce;
    inception_b4_7x1_reduce = vxCreateVirtualTensor(graph,4, inception_b4_7x1_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce);
    vx_size inception_b4_7x1_reduce_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b4_7x1_reduce_W;
    inception_b4_7x1_reduce_W = vxCreateTensor(context,4, inception_b4_7x1_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_reduce_W, dataFolder + "/weights/inception_b4_7x1_reduce.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_reduce_params;
    inception_b4_7x1_reduce_params.padding_x = 0;
    inception_b4_7x1_reduce_params.padding_y = 0;
    inception_b4_7x1_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_reduce_params.dilation_x = 0;
    inception_b4_7x1_reduce_params.dilation_y = 0;
    vx_node inception_b4_7x1_reduce_node;
    inception_b4_7x1_reduce_node = vxConvolutionLayer(graph, inception_b3_concat_inception_b3_concat_0_split_2, inception_b4_7x1_reduce_W, NULL, &inception_b4_7x1_reduce_params, sizeof(inception_b4_7x1_reduce_params ), inception_b4_7x1_reduce);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_reduce_node));

    // inception_b4_7x1_reduce_bn Layer
    vx_size inception_b4_7x1_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_reduce_scale;
    inception_b4_7x1_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_scale);
    vx_size inception_b4_7x1_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_7x1_reduce_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_reduce_bn_W;
    inception_b4_7x1_reduce_bn_W = vxCreateTensor(context,1, inception_b4_7x1_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_reduce_bn_W, dataFolder + "/weights/inception_b4_7x1_reduce_bn.f32"));
    vx_size inception_b4_7x1_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_reduce_bn_B;
    inception_b4_7x1_reduce_bn_B = vxCreateTensor(context,1, inception_b4_7x1_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_reduce_bn_B, dataFolder + "/bias/inception_b4_7x1_reduce_bn.f32"));
    vx_size inception_b4_7x1_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_reduce_scale_W;
    inception_b4_7x1_reduce_scale_W = vxCreateTensor(context,1, inception_b4_7x1_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_reduce_scale_W, dataFolder + "/weights/inception_b4_7x1_reduce_scale.f32"));
    vx_size inception_b4_7x1_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_reduce_scale_B;
    inception_b4_7x1_reduce_scale_B = vxCreateTensor(context,1, inception_b4_7x1_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_reduce_scale_B, dataFolder + "/bias/inception_b4_7x1_reduce_scale.f32"));
    vx_node inception_b4_7x1_reduce_bn_node;
    inception_b4_7x1_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1_reduce, inception_b4_7x1_reduce_bn_W, inception_b4_7x1_reduce_bn_B, inception_b4_7x1_reduce_scale_W, inception_b4_7x1_reduce_scale_B, inception_b4_7x1_reduce_bn_eps, inception_b4_7x1_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_reduce_bn_node));

    // inception_b4_7x1_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_reduce_relu Layer
    vx_size inception_b4_7x1_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_reduce_relu;
    inception_b4_7x1_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_relu);
    vx_enum inception_b4_7x1_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_reduce_relu_param_a = 0;
    vx_float32 inception_b4_7x1_reduce_relu_param_b = 0;
    vx_node inception_b4_7x1_reduce_relu_node;
    inception_b4_7x1_reduce_relu_node = vxActivationLayer(graph, inception_b4_7x1_reduce_scale, inception_b4_7x1_reduce_relu_mode, inception_b4_7x1_reduce_relu_param_a, inception_b4_7x1_reduce_relu_param_b, inception_b4_7x1_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_reduce_relu_node));

    // inception_b4_7x1_2 Layer
    vx_size inception_b4_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2;
    inception_b4_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2);
    vx_size inception_b4_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b4_7x1_2_W;
    inception_b4_7x1_2_W = vxCreateTensor(context,4, inception_b4_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_W, dataFolder + "/weights/inception_b4_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_2_params;
    inception_b4_7x1_2_params.padding_x = 0;
    inception_b4_7x1_2_params.padding_y = 3;
    inception_b4_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_2_params.dilation_x = 0;
    inception_b4_7x1_2_params.dilation_y = 0;
    vx_node inception_b4_7x1_2_node;
    inception_b4_7x1_2_node = vxConvolutionLayer(graph, inception_b4_7x1_reduce_relu, inception_b4_7x1_2_W, NULL, &inception_b4_7x1_2_params, sizeof(inception_b4_7x1_2_params ), inception_b4_7x1_2);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_node));

    // inception_b4_7x1_2_bn Layer
    vx_size inception_b4_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_scale;
    inception_b4_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_scale);
    vx_size inception_b4_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_2_bn_W;
    inception_b4_7x1_2_bn_W = vxCreateTensor(context,1, inception_b4_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_bn_W, dataFolder + "/weights/inception_b4_7x1_2_bn.f32"));
    vx_size inception_b4_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_bn_B;
    inception_b4_7x1_2_bn_B = vxCreateTensor(context,1, inception_b4_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_bn_B, dataFolder + "/bias/inception_b4_7x1_2_bn.f32"));
    vx_size inception_b4_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_scale_W;
    inception_b4_7x1_2_scale_W = vxCreateTensor(context,1, inception_b4_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_scale_W, dataFolder + "/weights/inception_b4_7x1_2_scale.f32"));
    vx_size inception_b4_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_scale_B;
    inception_b4_7x1_2_scale_B = vxCreateTensor(context,1, inception_b4_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_scale_B, dataFolder + "/bias/inception_b4_7x1_2_scale.f32"));
    vx_node inception_b4_7x1_2_bn_node;
    inception_b4_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1_2, inception_b4_7x1_2_bn_W, inception_b4_7x1_2_bn_B, inception_b4_7x1_2_scale_W, inception_b4_7x1_2_scale_B, inception_b4_7x1_2_bn_eps, inception_b4_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_bn_node));

    // inception_b4_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_2_relu Layer
    vx_size inception_b4_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_relu;
    inception_b4_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_relu);
    vx_enum inception_b4_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_2_relu_param_a = 0;
    vx_float32 inception_b4_7x1_2_relu_param_b = 0;
    vx_node inception_b4_7x1_2_relu_node;
    inception_b4_7x1_2_relu_node = vxActivationLayer(graph, inception_b4_7x1_2_scale, inception_b4_7x1_2_relu_mode, inception_b4_7x1_2_relu_param_a, inception_b4_7x1_2_relu_param_b, inception_b4_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_relu_node));

    // inception_b4_1x7_2 Layer
    vx_size inception_b4_1x7_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_2;
    inception_b4_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b4_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2);
    vx_size inception_b4_1x7_2_W_dims[4] = { 7, 1, 192, 192 };
    vx_tensor inception_b4_1x7_2_W;
    inception_b4_1x7_2_W = vxCreateTensor(context,4, inception_b4_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_W, dataFolder + "/weights/inception_b4_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_2_params;
    inception_b4_1x7_2_params.padding_x = 3;
    inception_b4_1x7_2_params.padding_y = 0;
    inception_b4_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_2_params.dilation_x = 0;
    inception_b4_1x7_2_params.dilation_y = 0;
    vx_node inception_b4_1x7_2_node;
    inception_b4_1x7_2_node = vxConvolutionLayer(graph, inception_b4_7x1_2_relu, inception_b4_1x7_2_W, NULL, &inception_b4_1x7_2_params, sizeof(inception_b4_1x7_2_params ), inception_b4_1x7_2);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_2_node));

    // inception_b4_1x7_2_bn Layer
    vx_size inception_b4_1x7_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_2_scale;
    inception_b4_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_scale);
    vx_size inception_b4_1x7_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_2_bn_W;
    inception_b4_1x7_2_bn_W = vxCreateTensor(context,1, inception_b4_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_bn_W, dataFolder + "/weights/inception_b4_1x7_2_bn.f32"));
    vx_size inception_b4_1x7_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_2_bn_B;
    inception_b4_1x7_2_bn_B = vxCreateTensor(context,1, inception_b4_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_bn_B, dataFolder + "/bias/inception_b4_1x7_2_bn.f32"));
    vx_size inception_b4_1x7_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_2_scale_W;
    inception_b4_1x7_2_scale_W = vxCreateTensor(context,1, inception_b4_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_scale_W, dataFolder + "/weights/inception_b4_1x7_2_scale.f32"));
    vx_size inception_b4_1x7_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_2_scale_B;
    inception_b4_1x7_2_scale_B = vxCreateTensor(context,1, inception_b4_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_scale_B, dataFolder + "/bias/inception_b4_1x7_2_scale.f32"));
    vx_node inception_b4_1x7_2_bn_node;
    inception_b4_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7_2, inception_b4_1x7_2_bn_W, inception_b4_1x7_2_bn_B, inception_b4_1x7_2_scale_W, inception_b4_1x7_2_scale_B, inception_b4_1x7_2_bn_eps, inception_b4_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_2_bn_node));

    // inception_b4_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_2_relu Layer
    vx_size inception_b4_1x7_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_2_relu;
    inception_b4_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_relu);
    vx_enum inception_b4_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_2_relu_param_a = 0;
    vx_float32 inception_b4_1x7_2_relu_param_b = 0;
    vx_node inception_b4_1x7_2_relu_node;
    inception_b4_1x7_2_relu_node = vxActivationLayer(graph, inception_b4_1x7_2_scale, inception_b4_1x7_2_relu_mode, inception_b4_1x7_2_relu_param_a, inception_b4_1x7_2_relu_param_b, inception_b4_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_2_relu_node));

    // inception_b4_7x1_3 Layer
    vx_size inception_b4_7x1_3_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_3;
    inception_b4_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b4_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3);
    vx_size inception_b4_7x1_3_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b4_7x1_3_W;
    inception_b4_7x1_3_W = vxCreateTensor(context,4, inception_b4_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_W, dataFolder + "/weights/inception_b4_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_3_params;
    inception_b4_7x1_3_params.padding_x = 0;
    inception_b4_7x1_3_params.padding_y = 3;
    inception_b4_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_3_params.dilation_x = 0;
    inception_b4_7x1_3_params.dilation_y = 0;
    vx_node inception_b4_7x1_3_node;
    inception_b4_7x1_3_node = vxConvolutionLayer(graph, inception_b4_1x7_2_relu, inception_b4_7x1_3_W, NULL, &inception_b4_7x1_3_params, sizeof(inception_b4_7x1_3_params ), inception_b4_7x1_3);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_3_node));

    // inception_b4_7x1_3_bn Layer
    vx_size inception_b4_7x1_3_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_3_scale;
    inception_b4_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_scale);
    vx_size inception_b4_7x1_3_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_3_bn_W;
    inception_b4_7x1_3_bn_W = vxCreateTensor(context,1, inception_b4_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_bn_W, dataFolder + "/weights/inception_b4_7x1_3_bn.f32"));
    vx_size inception_b4_7x1_3_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_3_bn_B;
    inception_b4_7x1_3_bn_B = vxCreateTensor(context,1, inception_b4_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_bn_B, dataFolder + "/bias/inception_b4_7x1_3_bn.f32"));
    vx_size inception_b4_7x1_3_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_3_scale_W;
    inception_b4_7x1_3_scale_W = vxCreateTensor(context,1, inception_b4_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_scale_W, dataFolder + "/weights/inception_b4_7x1_3_scale.f32"));
    vx_size inception_b4_7x1_3_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_3_scale_B;
    inception_b4_7x1_3_scale_B = vxCreateTensor(context,1, inception_b4_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_scale_B, dataFolder + "/bias/inception_b4_7x1_3_scale.f32"));
    vx_node inception_b4_7x1_3_bn_node;
    inception_b4_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1_3, inception_b4_7x1_3_bn_W, inception_b4_7x1_3_bn_B, inception_b4_7x1_3_scale_W, inception_b4_7x1_3_scale_B, inception_b4_7x1_3_bn_eps, inception_b4_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_3_bn_node));

    // inception_b4_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_3_relu Layer
    vx_size inception_b4_7x1_3_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_3_relu;
    inception_b4_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_relu);
    vx_enum inception_b4_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_3_relu_param_a = 0;
    vx_float32 inception_b4_7x1_3_relu_param_b = 0;
    vx_node inception_b4_7x1_3_relu_node;
    inception_b4_7x1_3_relu_node = vxActivationLayer(graph, inception_b4_7x1_3_scale, inception_b4_7x1_3_relu_mode, inception_b4_7x1_3_relu_param_a, inception_b4_7x1_3_relu_param_b, inception_b4_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_3_relu_node));

    // inception_b4_1x7_3 Layer
    vx_size inception_b4_1x7_3_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_3;
    inception_b4_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b4_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3);
    vx_size inception_b4_1x7_3_W_dims[4] = { 7, 1, 192, 192 };
    vx_tensor inception_b4_1x7_3_W;
    inception_b4_1x7_3_W = vxCreateTensor(context,4, inception_b4_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_W, dataFolder + "/weights/inception_b4_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_3_params;
    inception_b4_1x7_3_params.padding_x = 3;
    inception_b4_1x7_3_params.padding_y = 0;
    inception_b4_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_3_params.dilation_x = 0;
    inception_b4_1x7_3_params.dilation_y = 0;
    vx_node inception_b4_1x7_3_node;
    inception_b4_1x7_3_node = vxConvolutionLayer(graph, inception_b4_7x1_3_relu, inception_b4_1x7_3_W, NULL, &inception_b4_1x7_3_params, sizeof(inception_b4_1x7_3_params ), inception_b4_1x7_3);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_3_node));

    // inception_b4_1x7_3_bn Layer
    vx_size inception_b4_1x7_3_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_3_scale;
    inception_b4_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_scale);
    vx_size inception_b4_1x7_3_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_3_bn_W;
    inception_b4_1x7_3_bn_W = vxCreateTensor(context,1, inception_b4_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_bn_W, dataFolder + "/weights/inception_b4_1x7_3_bn.f32"));
    vx_size inception_b4_1x7_3_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_3_bn_B;
    inception_b4_1x7_3_bn_B = vxCreateTensor(context,1, inception_b4_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_bn_B, dataFolder + "/bias/inception_b4_1x7_3_bn.f32"));
    vx_size inception_b4_1x7_3_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_3_scale_W;
    inception_b4_1x7_3_scale_W = vxCreateTensor(context,1, inception_b4_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_scale_W, dataFolder + "/weights/inception_b4_1x7_3_scale.f32"));
    vx_size inception_b4_1x7_3_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_3_scale_B;
    inception_b4_1x7_3_scale_B = vxCreateTensor(context,1, inception_b4_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_scale_B, dataFolder + "/bias/inception_b4_1x7_3_scale.f32"));
    vx_node inception_b4_1x7_3_bn_node;
    inception_b4_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7_3, inception_b4_1x7_3_bn_W, inception_b4_1x7_3_bn_B, inception_b4_1x7_3_scale_W, inception_b4_1x7_3_scale_B, inception_b4_1x7_3_bn_eps, inception_b4_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_3_bn_node));

    // inception_b4_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_3_relu Layer
    vx_size inception_b4_1x7_3_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_3_relu;
    inception_b4_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_relu);
    vx_enum inception_b4_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_3_relu_param_a = 0;
    vx_float32 inception_b4_1x7_3_relu_param_b = 0;
    vx_node inception_b4_1x7_3_relu_node;
    inception_b4_1x7_3_relu_node = vxActivationLayer(graph, inception_b4_1x7_3_scale, inception_b4_1x7_3_relu_mode, inception_b4_1x7_3_relu_param_a, inception_b4_1x7_3_relu_param_b, inception_b4_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_3_relu_node));

    // inception_b4_pool_ave Layer
    vx_size inception_b4_pool_ave_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b4_pool_ave;
    inception_b4_pool_ave = vxCreateVirtualTensor(graph,4, inception_b4_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_pool_ave);
    vx_enum inception_b4_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b4_pool_ave_kernel_w = 3;
    vx_size inception_b4_pool_ave_kernel_h = 3;
    vx_size inception_b4_pool_ave_pad_w = 1;
    vx_size inception_b4_pool_ave_pad_h = 1;
    vx_enum inception_b4_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b4_pool_ave_node;
    inception_b4_pool_ave_node = vxPoolingLayer(graph, inception_b3_concat_inception_b3_concat_0_split_3, inception_b4_pool_ave_type, inception_b4_pool_ave_kernel_w, inception_b4_pool_ave_kernel_h, inception_b4_pool_ave_pad_w, inception_b4_pool_ave_pad_h, inception_b4_pool_ave_roundPolicy, inception_b4_pool_ave );
    ERROR_CHECK_OBJECT(inception_b4_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_pool_ave_node));

    // inception_b4_1x1 Layer
    vx_size inception_b4_1x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x1;
    inception_b4_1x1 = vxCreateVirtualTensor(graph,4, inception_b4_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1);
    vx_size inception_b4_1x1_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor inception_b4_1x1_W;
    inception_b4_1x1_W = vxCreateTensor(context,4, inception_b4_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_W, dataFolder + "/weights/inception_b4_1x1.f32"));
    vx_nn_convolution_params_t inception_b4_1x1_params;
    inception_b4_1x1_params.padding_x = 0;
    inception_b4_1x1_params.padding_y = 0;
    inception_b4_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x1_params.dilation_x = 0;
    inception_b4_1x1_params.dilation_y = 0;
    vx_node inception_b4_1x1_node;
    inception_b4_1x1_node = vxConvolutionLayer(graph, inception_b4_pool_ave, inception_b4_1x1_W, NULL, &inception_b4_1x1_params, sizeof(inception_b4_1x1_params ), inception_b4_1x1);
    ERROR_CHECK_OBJECT(inception_b4_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_node));

    // inception_b4_1x1_bn Layer
    vx_size inception_b4_1x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x1_scale;
    inception_b4_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_scale);
    vx_size inception_b4_1x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x1_bn_eps = 0.001;
    vx_tensor inception_b4_1x1_bn_W;
    inception_b4_1x1_bn_W = vxCreateTensor(context,1, inception_b4_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_bn_W, dataFolder + "/weights/inception_b4_1x1_bn.f32"));
    vx_size inception_b4_1x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x1_bn_B;
    inception_b4_1x1_bn_B = vxCreateTensor(context,1, inception_b4_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_bn_B, dataFolder + "/bias/inception_b4_1x1_bn.f32"));
    vx_size inception_b4_1x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x1_scale_W;
    inception_b4_1x1_scale_W = vxCreateTensor(context,1, inception_b4_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_scale_W, dataFolder + "/weights/inception_b4_1x1_scale.f32"));
    vx_size inception_b4_1x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x1_scale_B;
    inception_b4_1x1_scale_B = vxCreateTensor(context,1, inception_b4_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_scale_B, dataFolder + "/bias/inception_b4_1x1_scale.f32"));
    vx_node inception_b4_1x1_bn_node;
    inception_b4_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x1, inception_b4_1x1_bn_W, inception_b4_1x1_bn_B, inception_b4_1x1_scale_W, inception_b4_1x1_scale_B, inception_b4_1x1_bn_eps, inception_b4_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_bn_node));

    // inception_b4_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x1_relu Layer
    vx_size inception_b4_1x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x1_relu;
    inception_b4_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_relu);
    vx_enum inception_b4_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x1_relu_param_a = 0;
    vx_float32 inception_b4_1x1_relu_param_b = 0;
    vx_node inception_b4_1x1_relu_node;
    inception_b4_1x1_relu_node = vxActivationLayer(graph, inception_b4_1x1_scale, inception_b4_1x1_relu_mode, inception_b4_1x1_relu_param_a, inception_b4_1x1_relu_param_b, inception_b4_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_relu_node));

    // inception_b4_concat Layer
    vx_size inception_b4_concat_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b4_concat;
    inception_b4_concat = vxCreateVirtualTensor(graph,4, inception_b4_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_concat);
    vx_node inception_b4_concat_node;
    inception_b4_concat_node = vxConcatLayer(graph, inception_b4_concat, inception_b4_1x1_2_relu, inception_b4_7x1_relu, inception_b4_1x7_3_relu, inception_b4_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b4_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_concat_node));

    // inception_b4_concat_inception_b4_concat_0_split_0 Layer
    vx_size inception_b4_concat_inception_b4_concat_0_split_0_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b4_concat_inception_b4_concat_0_split_0;
    inception_b4_concat_inception_b4_concat_0_split_0 = vxCreateVirtualTensor(graph,4, inception_b4_concat_inception_b4_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_concat_inception_b4_concat_0_split_0);
    vx_node inception_b4_concat_inception_b4_concat_0_split_0_node;
    inception_b4_concat_inception_b4_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_b4_concat, (vx_reference)inception_b4_concat_inception_b4_concat_0_split_0);
    ERROR_CHECK_OBJECT(inception_b4_concat_inception_b4_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_concat_inception_b4_concat_0_split_0_node));

    // inception_b4_concat_inception_b4_concat_0_split_1 Layer
    vx_size inception_b4_concat_inception_b4_concat_0_split_1_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b4_concat_inception_b4_concat_0_split_1;
    inception_b4_concat_inception_b4_concat_0_split_1 = vxCreateVirtualTensor(graph,4, inception_b4_concat_inception_b4_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_concat_inception_b4_concat_0_split_1);
    vx_node inception_b4_concat_inception_b4_concat_0_split_1_node;
    inception_b4_concat_inception_b4_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_b4_concat, (vx_reference)inception_b4_concat_inception_b4_concat_0_split_1);
    ERROR_CHECK_OBJECT(inception_b4_concat_inception_b4_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_concat_inception_b4_concat_0_split_1_node));

    // inception_b4_concat_inception_b4_concat_0_split_2 Layer
    vx_size inception_b4_concat_inception_b4_concat_0_split_2_dims[4] = { 17, 17, 768, 1 };
    vx_tensor inception_b4_concat_inception_b4_concat_0_split_2;
    inception_b4_concat_inception_b4_concat_0_split_2 = vxCreateVirtualTensor(graph,4, inception_b4_concat_inception_b4_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_concat_inception_b4_concat_0_split_2);
    vx_node inception_b4_concat_inception_b4_concat_0_split_2_node;
    inception_b4_concat_inception_b4_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_b4_concat, (vx_reference)inception_b4_concat_inception_b4_concat_0_split_2);
    ERROR_CHECK_OBJECT(inception_b4_concat_inception_b4_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_concat_inception_b4_concat_0_split_2_node));

    // reduction_b_3x3_reduce Layer
    vx_size reduction_b_3x3_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_3x3_reduce;
    reduction_b_3x3_reduce = vxCreateVirtualTensor(graph,4, reduction_b_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce);
    vx_size reduction_b_3x3_reduce_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor reduction_b_3x3_reduce_W;
    reduction_b_3x3_reduce_W = vxCreateTensor(context,4, reduction_b_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_W, dataFolder + "/weights/reduction_b_3x3_reduce.f32"));
    vx_nn_convolution_params_t reduction_b_3x3_reduce_params;
    reduction_b_3x3_reduce_params.padding_x = 0;
    reduction_b_3x3_reduce_params.padding_y = 0;
    reduction_b_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_3x3_reduce_params.dilation_x = 0;
    reduction_b_3x3_reduce_params.dilation_y = 0;
    vx_node reduction_b_3x3_reduce_node;
    reduction_b_3x3_reduce_node = vxConvolutionLayer(graph, inception_b4_concat_inception_b4_concat_0_split_0, reduction_b_3x3_reduce_W, NULL, &reduction_b_3x3_reduce_params, sizeof(reduction_b_3x3_reduce_params ), reduction_b_3x3_reduce);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_reduce_node));

    // reduction_b_3x3_reduce_bn Layer
    vx_size reduction_b_3x3_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_3x3_reduce_scale;
    reduction_b_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, reduction_b_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_scale);
    vx_size reduction_b_3x3_reduce_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_3x3_reduce_bn_eps = 0.001;
    vx_tensor reduction_b_3x3_reduce_bn_W;
    reduction_b_3x3_reduce_bn_W = vxCreateTensor(context,1, reduction_b_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_bn_W, dataFolder + "/weights/reduction_b_3x3_reduce_bn.f32"));
    vx_size reduction_b_3x3_reduce_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_reduce_bn_B;
    reduction_b_3x3_reduce_bn_B = vxCreateTensor(context,1, reduction_b_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_bn_B, dataFolder + "/bias/reduction_b_3x3_reduce_bn.f32"));
    vx_size reduction_b_3x3_reduce_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_reduce_scale_W;
    reduction_b_3x3_reduce_scale_W = vxCreateTensor(context,1, reduction_b_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_scale_W, dataFolder + "/weights/reduction_b_3x3_reduce_scale.f32"));
    vx_size reduction_b_3x3_reduce_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_reduce_scale_B;
    reduction_b_3x3_reduce_scale_B = vxCreateTensor(context,1, reduction_b_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_scale_B, dataFolder + "/bias/reduction_b_3x3_reduce_scale.f32"));
    vx_node reduction_b_3x3_reduce_bn_node;
    reduction_b_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, reduction_b_3x3_reduce, reduction_b_3x3_reduce_bn_W, reduction_b_3x3_reduce_bn_B, reduction_b_3x3_reduce_scale_W, reduction_b_3x3_reduce_scale_B, reduction_b_3x3_reduce_bn_eps, reduction_b_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_reduce_bn_node));

    // reduction_b_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_3x3_reduce_relu Layer
    vx_size reduction_b_3x3_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_3x3_reduce_relu;
    reduction_b_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, reduction_b_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_relu);
    vx_enum reduction_b_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_3x3_reduce_relu_param_a = 0;
    vx_float32 reduction_b_3x3_reduce_relu_param_b = 0;
    vx_node reduction_b_3x3_reduce_relu_node;
    reduction_b_3x3_reduce_relu_node = vxActivationLayer(graph, reduction_b_3x3_reduce_scale, reduction_b_3x3_reduce_relu_mode, reduction_b_3x3_reduce_relu_param_a, reduction_b_3x3_reduce_relu_param_b, reduction_b_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_reduce_relu_node));

    // reduction_b_3x3 Layer
    vx_size reduction_b_3x3_dims[4] = { 8, 8, 320, 1 };
    vx_tensor reduction_b_3x3;
    reduction_b_3x3 = vxCreateVirtualTensor(graph,4, reduction_b_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3);
    vx_size reduction_b_3x3_W_dims[4] = { 3, 3, 192, 320 };
    vx_tensor reduction_b_3x3_W;
    reduction_b_3x3_W = vxCreateTensor(context,4, reduction_b_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_W, dataFolder + "/weights/reduction_b_3x3.f32"));
    vx_nn_convolution_params_t reduction_b_3x3_params;
    reduction_b_3x3_params.padding_x = 0;
    reduction_b_3x3_params.padding_y = 0;
    reduction_b_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_3x3_params.dilation_x = 0;
    reduction_b_3x3_params.dilation_y = 0;
    vx_node reduction_b_3x3_node;
    reduction_b_3x3_node = vxConvolutionLayer(graph, reduction_b_3x3_reduce_relu, reduction_b_3x3_W, NULL, &reduction_b_3x3_params, sizeof(reduction_b_3x3_params ), reduction_b_3x3);
    ERROR_CHECK_OBJECT(reduction_b_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_node));

    // reduction_b_3x3_bn Layer
    vx_size reduction_b_3x3_scale_dims[4] = { 8, 8, 320, 1 };
    vx_tensor reduction_b_3x3_scale;
    reduction_b_3x3_scale = vxCreateVirtualTensor(graph,4, reduction_b_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_scale);
    vx_size reduction_b_3x3_bn_W_dims[1] = { 320 };
    vx_float32 reduction_b_3x3_bn_eps = 0.001;
    vx_tensor reduction_b_3x3_bn_W;
    reduction_b_3x3_bn_W = vxCreateTensor(context,1, reduction_b_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_bn_W, dataFolder + "/weights/reduction_b_3x3_bn.f32"));
    vx_size reduction_b_3x3_bn_B_dims[1] = { 320 };
    vx_tensor reduction_b_3x3_bn_B;
    reduction_b_3x3_bn_B = vxCreateTensor(context,1, reduction_b_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_bn_B, dataFolder + "/bias/reduction_b_3x3_bn.f32"));
    vx_size reduction_b_3x3_scale_W_dims[1] = { 320 };
    vx_tensor reduction_b_3x3_scale_W;
    reduction_b_3x3_scale_W = vxCreateTensor(context,1, reduction_b_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_scale_W, dataFolder + "/weights/reduction_b_3x3_scale.f32"));
    vx_size reduction_b_3x3_scale_B_dims[1] = { 320 };
    vx_tensor reduction_b_3x3_scale_B;
    reduction_b_3x3_scale_B = vxCreateTensor(context,1, reduction_b_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_scale_B, dataFolder + "/bias/reduction_b_3x3_scale.f32"));
    vx_node reduction_b_3x3_bn_node;
    reduction_b_3x3_bn_node = vxBatchNormalizationLayer(graph, reduction_b_3x3, reduction_b_3x3_bn_W, reduction_b_3x3_bn_B, reduction_b_3x3_scale_W, reduction_b_3x3_scale_B, reduction_b_3x3_bn_eps, reduction_b_3x3_scale);
    ERROR_CHECK_OBJECT(reduction_b_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_bn_node));

    // reduction_b_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_3x3_relu Layer
    vx_size reduction_b_3x3_relu_dims[4] = { 8, 8, 320, 1 };
    vx_tensor reduction_b_3x3_relu;
    reduction_b_3x3_relu = vxCreateVirtualTensor(graph,4, reduction_b_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_relu);
    vx_enum reduction_b_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_3x3_relu_param_a = 0;
    vx_float32 reduction_b_3x3_relu_param_b = 0;
    vx_node reduction_b_3x3_relu_node;
    reduction_b_3x3_relu_node = vxActivationLayer(graph, reduction_b_3x3_scale, reduction_b_3x3_relu_mode, reduction_b_3x3_relu_param_a, reduction_b_3x3_relu_param_b, reduction_b_3x3_relu);
    ERROR_CHECK_OBJECT(reduction_b_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_relu_node));

    // reduction_b_1x7_reduce Layer
    vx_size reduction_b_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_1x7_reduce;
    reduction_b_1x7_reduce = vxCreateVirtualTensor(graph,4, reduction_b_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce);
    vx_size reduction_b_1x7_reduce_W_dims[4] = { 1, 1, 768, 192 };
    vx_tensor reduction_b_1x7_reduce_W;
    reduction_b_1x7_reduce_W = vxCreateTensor(context,4, reduction_b_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_W, dataFolder + "/weights/reduction_b_1x7_reduce.f32"));
    vx_nn_convolution_params_t reduction_b_1x7_reduce_params;
    reduction_b_1x7_reduce_params.padding_x = 0;
    reduction_b_1x7_reduce_params.padding_y = 0;
    reduction_b_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_1x7_reduce_params.dilation_x = 0;
    reduction_b_1x7_reduce_params.dilation_y = 0;
    vx_node reduction_b_1x7_reduce_node;
    reduction_b_1x7_reduce_node = vxConvolutionLayer(graph, inception_b4_concat_inception_b4_concat_0_split_1, reduction_b_1x7_reduce_W, NULL, &reduction_b_1x7_reduce_params, sizeof(reduction_b_1x7_reduce_params ), reduction_b_1x7_reduce);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_reduce_node));

    // reduction_b_1x7_reduce_bn Layer
    vx_size reduction_b_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_1x7_reduce_scale;
    reduction_b_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, reduction_b_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_scale);
    vx_size reduction_b_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_1x7_reduce_bn_eps = 0.001;
    vx_tensor reduction_b_1x7_reduce_bn_W;
    reduction_b_1x7_reduce_bn_W = vxCreateTensor(context,1, reduction_b_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_bn_W, dataFolder + "/weights/reduction_b_1x7_reduce_bn.f32"));
    vx_size reduction_b_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_1x7_reduce_bn_B;
    reduction_b_1x7_reduce_bn_B = vxCreateTensor(context,1, reduction_b_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_bn_B, dataFolder + "/bias/reduction_b_1x7_reduce_bn.f32"));
    vx_size reduction_b_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_1x7_reduce_scale_W;
    reduction_b_1x7_reduce_scale_W = vxCreateTensor(context,1, reduction_b_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_scale_W, dataFolder + "/weights/reduction_b_1x7_reduce_scale.f32"));
    vx_size reduction_b_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_1x7_reduce_scale_B;
    reduction_b_1x7_reduce_scale_B = vxCreateTensor(context,1, reduction_b_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_scale_B, dataFolder + "/bias/reduction_b_1x7_reduce_scale.f32"));
    vx_node reduction_b_1x7_reduce_bn_node;
    reduction_b_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, reduction_b_1x7_reduce, reduction_b_1x7_reduce_bn_W, reduction_b_1x7_reduce_bn_B, reduction_b_1x7_reduce_scale_W, reduction_b_1x7_reduce_scale_B, reduction_b_1x7_reduce_bn_eps, reduction_b_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_reduce_bn_node));

    // reduction_b_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_1x7_reduce_relu Layer
    vx_size reduction_b_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_1x7_reduce_relu;
    reduction_b_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, reduction_b_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_relu);
    vx_enum reduction_b_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_1x7_reduce_relu_param_a = 0;
    vx_float32 reduction_b_1x7_reduce_relu_param_b = 0;
    vx_node reduction_b_1x7_reduce_relu_node;
    reduction_b_1x7_reduce_relu_node = vxActivationLayer(graph, reduction_b_1x7_reduce_scale, reduction_b_1x7_reduce_relu_mode, reduction_b_1x7_reduce_relu_param_a, reduction_b_1x7_reduce_relu_param_b, reduction_b_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_reduce_relu_node));

    // reduction_b_1x7 Layer
    vx_size reduction_b_1x7_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_1x7;
    reduction_b_1x7 = vxCreateVirtualTensor(graph,4, reduction_b_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7);
    vx_size reduction_b_1x7_W_dims[4] = { 7, 1, 192, 192 };
    vx_tensor reduction_b_1x7_W;
    reduction_b_1x7_W = vxCreateTensor(context,4, reduction_b_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_W, dataFolder + "/weights/reduction_b_1x7.f32"));
    vx_nn_convolution_params_t reduction_b_1x7_params;
    reduction_b_1x7_params.padding_x = 3;
    reduction_b_1x7_params.padding_y = 0;
    reduction_b_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_1x7_params.dilation_x = 0;
    reduction_b_1x7_params.dilation_y = 0;
    vx_node reduction_b_1x7_node;
    reduction_b_1x7_node = vxConvolutionLayer(graph, reduction_b_1x7_reduce_relu, reduction_b_1x7_W, NULL, &reduction_b_1x7_params, sizeof(reduction_b_1x7_params ), reduction_b_1x7);
    ERROR_CHECK_OBJECT(reduction_b_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_node));

    // reduction_b_1x7_bn Layer
    vx_size reduction_b_1x7_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_1x7_scale;
    reduction_b_1x7_scale = vxCreateVirtualTensor(graph,4, reduction_b_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_scale);
    vx_size reduction_b_1x7_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_1x7_bn_eps = 0.001;
    vx_tensor reduction_b_1x7_bn_W;
    reduction_b_1x7_bn_W = vxCreateTensor(context,1, reduction_b_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_bn_W, dataFolder + "/weights/reduction_b_1x7_bn.f32"));
    vx_size reduction_b_1x7_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_1x7_bn_B;
    reduction_b_1x7_bn_B = vxCreateTensor(context,1, reduction_b_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_bn_B, dataFolder + "/bias/reduction_b_1x7_bn.f32"));
    vx_size reduction_b_1x7_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_1x7_scale_W;
    reduction_b_1x7_scale_W = vxCreateTensor(context,1, reduction_b_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_scale_W, dataFolder + "/weights/reduction_b_1x7_scale.f32"));
    vx_size reduction_b_1x7_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_1x7_scale_B;
    reduction_b_1x7_scale_B = vxCreateTensor(context,1, reduction_b_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_scale_B, dataFolder + "/bias/reduction_b_1x7_scale.f32"));
    vx_node reduction_b_1x7_bn_node;
    reduction_b_1x7_bn_node = vxBatchNormalizationLayer(graph, reduction_b_1x7, reduction_b_1x7_bn_W, reduction_b_1x7_bn_B, reduction_b_1x7_scale_W, reduction_b_1x7_scale_B, reduction_b_1x7_bn_eps, reduction_b_1x7_scale);
    ERROR_CHECK_OBJECT(reduction_b_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_bn_node));

    // reduction_b_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_1x7_relu Layer
    vx_size reduction_b_1x7_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_1x7_relu;
    reduction_b_1x7_relu = vxCreateVirtualTensor(graph,4, reduction_b_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_relu);
    vx_enum reduction_b_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_1x7_relu_param_a = 0;
    vx_float32 reduction_b_1x7_relu_param_b = 0;
    vx_node reduction_b_1x7_relu_node;
    reduction_b_1x7_relu_node = vxActivationLayer(graph, reduction_b_1x7_scale, reduction_b_1x7_relu_mode, reduction_b_1x7_relu_param_a, reduction_b_1x7_relu_param_b, reduction_b_1x7_relu);
    ERROR_CHECK_OBJECT(reduction_b_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_relu_node));

    // reduction_b_7x1 Layer
    vx_size reduction_b_7x1_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_7x1;
    reduction_b_7x1 = vxCreateVirtualTensor(graph,4, reduction_b_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_7x1);
    vx_size reduction_b_7x1_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor reduction_b_7x1_W;
    reduction_b_7x1_W = vxCreateTensor(context,4, reduction_b_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_W, dataFolder + "/weights/reduction_b_7x1.f32"));
    vx_nn_convolution_params_t reduction_b_7x1_params;
    reduction_b_7x1_params.padding_x = 0;
    reduction_b_7x1_params.padding_y = 3;
    reduction_b_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_7x1_params.dilation_x = 0;
    reduction_b_7x1_params.dilation_y = 0;
    vx_node reduction_b_7x1_node;
    reduction_b_7x1_node = vxConvolutionLayer(graph, reduction_b_1x7_relu, reduction_b_7x1_W, NULL, &reduction_b_7x1_params, sizeof(reduction_b_7x1_params ), reduction_b_7x1);
    ERROR_CHECK_OBJECT(reduction_b_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_7x1_node));

    // reduction_b_7x1_bn Layer
    vx_size reduction_b_7x1_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_7x1_scale;
    reduction_b_7x1_scale = vxCreateVirtualTensor(graph,4, reduction_b_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_scale);
    vx_size reduction_b_7x1_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_7x1_bn_eps = 0.001;
    vx_tensor reduction_b_7x1_bn_W;
    reduction_b_7x1_bn_W = vxCreateTensor(context,1, reduction_b_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_bn_W, dataFolder + "/weights/reduction_b_7x1_bn.f32"));
    vx_size reduction_b_7x1_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_7x1_bn_B;
    reduction_b_7x1_bn_B = vxCreateTensor(context,1, reduction_b_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_bn_B, dataFolder + "/bias/reduction_b_7x1_bn.f32"));
    vx_size reduction_b_7x1_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_7x1_scale_W;
    reduction_b_7x1_scale_W = vxCreateTensor(context,1, reduction_b_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_scale_W, dataFolder + "/weights/reduction_b_7x1_scale.f32"));
    vx_size reduction_b_7x1_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_7x1_scale_B;
    reduction_b_7x1_scale_B = vxCreateTensor(context,1, reduction_b_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_scale_B, dataFolder + "/bias/reduction_b_7x1_scale.f32"));
    vx_node reduction_b_7x1_bn_node;
    reduction_b_7x1_bn_node = vxBatchNormalizationLayer(graph, reduction_b_7x1, reduction_b_7x1_bn_W, reduction_b_7x1_bn_B, reduction_b_7x1_scale_W, reduction_b_7x1_scale_B, reduction_b_7x1_bn_eps, reduction_b_7x1_scale);
    ERROR_CHECK_OBJECT(reduction_b_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_7x1_bn_node));

    // reduction_b_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_7x1_relu Layer
    vx_size reduction_b_7x1_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_7x1_relu;
    reduction_b_7x1_relu = vxCreateVirtualTensor(graph,4, reduction_b_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_relu);
    vx_enum reduction_b_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_7x1_relu_param_a = 0;
    vx_float32 reduction_b_7x1_relu_param_b = 0;
    vx_node reduction_b_7x1_relu_node;
    reduction_b_7x1_relu_node = vxActivationLayer(graph, reduction_b_7x1_scale, reduction_b_7x1_relu_mode, reduction_b_7x1_relu_param_a, reduction_b_7x1_relu_param_b, reduction_b_7x1_relu);
    ERROR_CHECK_OBJECT(reduction_b_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_7x1_relu_node));

    // reduction_b_3x3_2 Layer
    vx_size reduction_b_3x3_2_dims[4] = { 8, 8, 192, 1 };
    vx_tensor reduction_b_3x3_2;
    reduction_b_3x3_2 = vxCreateVirtualTensor(graph,4, reduction_b_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2);
    vx_size reduction_b_3x3_2_W_dims[4] = { 3, 3, 192, 192 };
    vx_tensor reduction_b_3x3_2_W;
    reduction_b_3x3_2_W = vxCreateTensor(context,4, reduction_b_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_W, dataFolder + "/weights/reduction_b_3x3_2.f32"));
    vx_nn_convolution_params_t reduction_b_3x3_2_params;
    reduction_b_3x3_2_params.padding_x = 0;
    reduction_b_3x3_2_params.padding_y = 0;
    reduction_b_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_3x3_2_params.dilation_x = 0;
    reduction_b_3x3_2_params.dilation_y = 0;
    vx_node reduction_b_3x3_2_node;
    reduction_b_3x3_2_node = vxConvolutionLayer(graph, reduction_b_7x1_relu, reduction_b_3x3_2_W, NULL, &reduction_b_3x3_2_params, sizeof(reduction_b_3x3_2_params ), reduction_b_3x3_2);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_2_node));

    // reduction_b_3x3_2_bn Layer
    vx_size reduction_b_3x3_2_scale_dims[4] = { 8, 8, 192, 1 };
    vx_tensor reduction_b_3x3_2_scale;
    reduction_b_3x3_2_scale = vxCreateVirtualTensor(graph,4, reduction_b_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_scale);
    vx_size reduction_b_3x3_2_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_3x3_2_bn_eps = 0.001;
    vx_tensor reduction_b_3x3_2_bn_W;
    reduction_b_3x3_2_bn_W = vxCreateTensor(context,1, reduction_b_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_bn_W, dataFolder + "/weights/reduction_b_3x3_2_bn.f32"));
    vx_size reduction_b_3x3_2_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_2_bn_B;
    reduction_b_3x3_2_bn_B = vxCreateTensor(context,1, reduction_b_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_bn_B, dataFolder + "/bias/reduction_b_3x3_2_bn.f32"));
    vx_size reduction_b_3x3_2_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_2_scale_W;
    reduction_b_3x3_2_scale_W = vxCreateTensor(context,1, reduction_b_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_scale_W, dataFolder + "/weights/reduction_b_3x3_2_scale.f32"));
    vx_size reduction_b_3x3_2_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_2_scale_B;
    reduction_b_3x3_2_scale_B = vxCreateTensor(context,1, reduction_b_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_scale_B, dataFolder + "/bias/reduction_b_3x3_2_scale.f32"));
    vx_node reduction_b_3x3_2_bn_node;
    reduction_b_3x3_2_bn_node = vxBatchNormalizationLayer(graph, reduction_b_3x3_2, reduction_b_3x3_2_bn_W, reduction_b_3x3_2_bn_B, reduction_b_3x3_2_scale_W, reduction_b_3x3_2_scale_B, reduction_b_3x3_2_bn_eps, reduction_b_3x3_2_scale);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_2_bn_node));

    // reduction_b_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_3x3_2_relu Layer
    vx_size reduction_b_3x3_2_relu_dims[4] = { 8, 8, 192, 1 };
    vx_tensor reduction_b_3x3_2_relu;
    reduction_b_3x3_2_relu = vxCreateVirtualTensor(graph,4, reduction_b_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_relu);
    vx_enum reduction_b_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_3x3_2_relu_param_a = 0;
    vx_float32 reduction_b_3x3_2_relu_param_b = 0;
    vx_node reduction_b_3x3_2_relu_node;
    reduction_b_3x3_2_relu_node = vxActivationLayer(graph, reduction_b_3x3_2_scale, reduction_b_3x3_2_relu_mode, reduction_b_3x3_2_relu_param_a, reduction_b_3x3_2_relu_param_b, reduction_b_3x3_2_relu);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_2_relu_node));

    // reduction_b_pool Layer
    vx_size reduction_b_pool_dims[4] = { 8, 8, 768, 1 };
    vx_tensor reduction_b_pool;
    reduction_b_pool = vxCreateVirtualTensor(graph,4, reduction_b_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_pool);
    vx_enum reduction_b_pool_type = VX_NN_POOLING_MAX;
    vx_size reduction_b_pool_kernel_w = 3;
    vx_size reduction_b_pool_kernel_h = 3;
    vx_size reduction_b_pool_pad_w = 0;
    vx_size reduction_b_pool_pad_h = 0;
    vx_enum reduction_b_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node reduction_b_pool_node;
    reduction_b_pool_node = vxPoolingLayer(graph, inception_b4_concat_inception_b4_concat_0_split_2, reduction_b_pool_type, reduction_b_pool_kernel_w, reduction_b_pool_kernel_h, reduction_b_pool_pad_w, reduction_b_pool_pad_h, reduction_b_pool_roundPolicy, reduction_b_pool );
    ERROR_CHECK_OBJECT(reduction_b_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_pool_node));

    // reduction_b_concat Layer
    vx_size reduction_b_concat_dims[4] = { 8, 8, 1280, 1 };
    vx_tensor reduction_b_concat;
    reduction_b_concat = vxCreateVirtualTensor(graph,4, reduction_b_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_concat);
    vx_node reduction_b_concat_node;
    reduction_b_concat_node = vxConcatLayer(graph, reduction_b_concat, reduction_b_3x3_relu, reduction_b_3x3_2_relu, reduction_b_pool, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(reduction_b_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_concat_node));

    // reduction_b_concat_reduction_b_concat_0_split_0 Layer
    vx_size reduction_b_concat_reduction_b_concat_0_split_0_dims[4] = { 8, 8, 1280, 1 };
    vx_tensor reduction_b_concat_reduction_b_concat_0_split_0;
    reduction_b_concat_reduction_b_concat_0_split_0 = vxCreateVirtualTensor(graph,4, reduction_b_concat_reduction_b_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_0);
    vx_node reduction_b_concat_reduction_b_concat_0_split_0_node;
    reduction_b_concat_reduction_b_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)reduction_b_concat, (vx_reference)reduction_b_concat_reduction_b_concat_0_split_0);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_concat_reduction_b_concat_0_split_0_node));

    // reduction_b_concat_reduction_b_concat_0_split_1 Layer
    vx_size reduction_b_concat_reduction_b_concat_0_split_1_dims[4] = { 8, 8, 1280, 1 };
    vx_tensor reduction_b_concat_reduction_b_concat_0_split_1;
    reduction_b_concat_reduction_b_concat_0_split_1 = vxCreateVirtualTensor(graph,4, reduction_b_concat_reduction_b_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_1);
    vx_node reduction_b_concat_reduction_b_concat_0_split_1_node;
    reduction_b_concat_reduction_b_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)reduction_b_concat, (vx_reference)reduction_b_concat_reduction_b_concat_0_split_1);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_concat_reduction_b_concat_0_split_1_node));

    // reduction_b_concat_reduction_b_concat_0_split_2 Layer
    vx_size reduction_b_concat_reduction_b_concat_0_split_2_dims[4] = { 8, 8, 1280, 1 };
    vx_tensor reduction_b_concat_reduction_b_concat_0_split_2;
    reduction_b_concat_reduction_b_concat_0_split_2 = vxCreateVirtualTensor(graph,4, reduction_b_concat_reduction_b_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_2);
    vx_node reduction_b_concat_reduction_b_concat_0_split_2_node;
    reduction_b_concat_reduction_b_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)reduction_b_concat, (vx_reference)reduction_b_concat_reduction_b_concat_0_split_2);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_concat_reduction_b_concat_0_split_2_node));

    // reduction_b_concat_reduction_b_concat_0_split_3 Layer
    vx_size reduction_b_concat_reduction_b_concat_0_split_3_dims[4] = { 8, 8, 1280, 1 };
    vx_tensor reduction_b_concat_reduction_b_concat_0_split_3;
    reduction_b_concat_reduction_b_concat_0_split_3 = vxCreateVirtualTensor(graph,4, reduction_b_concat_reduction_b_concat_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_3);
    vx_node reduction_b_concat_reduction_b_concat_0_split_3_node;
    reduction_b_concat_reduction_b_concat_0_split_3_node = vxCopyNode( graph, (vx_reference)reduction_b_concat, (vx_reference)reduction_b_concat_reduction_b_concat_0_split_3);
    ERROR_CHECK_OBJECT(reduction_b_concat_reduction_b_concat_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_concat_reduction_b_concat_0_split_3_node));

    // inception_c1_1x1_2 Layer
    vx_size inception_c1_1x1_2_dims[4] = { 8, 8, 320, 1 };
    vx_tensor inception_c1_1x1_2;
    inception_c1_1x1_2 = vxCreateVirtualTensor(graph,4, inception_c1_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2);
    vx_size inception_c1_1x1_2_W_dims[4] = { 1, 1, 1280, 320 };
    vx_tensor inception_c1_1x1_2_W;
    inception_c1_1x1_2_W = vxCreateTensor(context,4, inception_c1_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_W, dataFolder + "/weights/inception_c1_1x1_2.f32"));
    vx_nn_convolution_params_t inception_c1_1x1_2_params;
    inception_c1_1x1_2_params.padding_x = 0;
    inception_c1_1x1_2_params.padding_y = 0;
    inception_c1_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x1_2_params.dilation_x = 0;
    inception_c1_1x1_2_params.dilation_y = 0;
    vx_node inception_c1_1x1_2_node;
    inception_c1_1x1_2_node = vxConvolutionLayer(graph, reduction_b_concat_reduction_b_concat_0_split_0, inception_c1_1x1_2_W, NULL, &inception_c1_1x1_2_params, sizeof(inception_c1_1x1_2_params ), inception_c1_1x1_2);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_2_node));

    // inception_c1_1x1_2_bn Layer
    vx_size inception_c1_1x1_2_scale_dims[4] = { 8, 8, 320, 1 };
    vx_tensor inception_c1_1x1_2_scale;
    inception_c1_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_scale);
    vx_size inception_c1_1x1_2_bn_W_dims[1] = { 320 };
    vx_float32 inception_c1_1x1_2_bn_eps = 0.001;
    vx_tensor inception_c1_1x1_2_bn_W;
    inception_c1_1x1_2_bn_W = vxCreateTensor(context,1, inception_c1_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_bn_W, dataFolder + "/weights/inception_c1_1x1_2_bn.f32"));
    vx_size inception_c1_1x1_2_bn_B_dims[1] = { 320 };
    vx_tensor inception_c1_1x1_2_bn_B;
    inception_c1_1x1_2_bn_B = vxCreateTensor(context,1, inception_c1_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_bn_B, dataFolder + "/bias/inception_c1_1x1_2_bn.f32"));
    vx_size inception_c1_1x1_2_scale_W_dims[1] = { 320 };
    vx_tensor inception_c1_1x1_2_scale_W;
    inception_c1_1x1_2_scale_W = vxCreateTensor(context,1, inception_c1_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_scale_W, dataFolder + "/weights/inception_c1_1x1_2_scale.f32"));
    vx_size inception_c1_1x1_2_scale_B_dims[1] = { 320 };
    vx_tensor inception_c1_1x1_2_scale_B;
    inception_c1_1x1_2_scale_B = vxCreateTensor(context,1, inception_c1_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_scale_B, dataFolder + "/bias/inception_c1_1x1_2_scale.f32"));
    vx_node inception_c1_1x1_2_bn_node;
    inception_c1_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x1_2, inception_c1_1x1_2_bn_W, inception_c1_1x1_2_bn_B, inception_c1_1x1_2_scale_W, inception_c1_1x1_2_scale_B, inception_c1_1x1_2_bn_eps, inception_c1_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_2_bn_node));

    // inception_c1_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x1_2_relu Layer
    vx_size inception_c1_1x1_2_relu_dims[4] = { 8, 8, 320, 1 };
    vx_tensor inception_c1_1x1_2_relu;
    inception_c1_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_relu);
    vx_enum inception_c1_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x1_2_relu_param_a = 0;
    vx_float32 inception_c1_1x1_2_relu_param_b = 0;
    vx_node inception_c1_1x1_2_relu_node;
    inception_c1_1x1_2_relu_node = vxActivationLayer(graph, inception_c1_1x1_2_scale, inception_c1_1x1_2_relu_mode, inception_c1_1x1_2_relu_param_a, inception_c1_1x1_2_relu_param_b, inception_c1_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_2_relu_node));

    // inception_c1_1x3_reduce Layer
    vx_size inception_c1_1x3_reduce_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_reduce;
    inception_c1_1x3_reduce = vxCreateVirtualTensor(graph,4, inception_c1_1x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce);
    vx_size inception_c1_1x3_reduce_W_dims[4] = { 1, 1, 1280, 384 };
    vx_tensor inception_c1_1x3_reduce_W;
    inception_c1_1x3_reduce_W = vxCreateTensor(context,4, inception_c1_1x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_reduce_W, dataFolder + "/weights/inception_c1_1x3_reduce.f32"));
    vx_nn_convolution_params_t inception_c1_1x3_reduce_params;
    inception_c1_1x3_reduce_params.padding_x = 0;
    inception_c1_1x3_reduce_params.padding_y = 0;
    inception_c1_1x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x3_reduce_params.dilation_x = 0;
    inception_c1_1x3_reduce_params.dilation_y = 0;
    vx_node inception_c1_1x3_reduce_node;
    inception_c1_1x3_reduce_node = vxConvolutionLayer(graph, reduction_b_concat_reduction_b_concat_0_split_1, inception_c1_1x3_reduce_W, NULL, &inception_c1_1x3_reduce_params, sizeof(inception_c1_1x3_reduce_params ), inception_c1_1x3_reduce);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_reduce_node));

    // inception_c1_1x3_reduce_bn Layer
    vx_size inception_c1_1x3_reduce_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_reduce_scale;
    inception_c1_1x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_scale);
    vx_size inception_c1_1x3_reduce_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_1x3_reduce_bn_eps = 0.001;
    vx_tensor inception_c1_1x3_reduce_bn_W;
    inception_c1_1x3_reduce_bn_W = vxCreateTensor(context,1, inception_c1_1x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_reduce_bn_W, dataFolder + "/weights/inception_c1_1x3_reduce_bn.f32"));
    vx_size inception_c1_1x3_reduce_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_reduce_bn_B;
    inception_c1_1x3_reduce_bn_B = vxCreateTensor(context,1, inception_c1_1x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_reduce_bn_B, dataFolder + "/bias/inception_c1_1x3_reduce_bn.f32"));
    vx_size inception_c1_1x3_reduce_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_reduce_scale_W;
    inception_c1_1x3_reduce_scale_W = vxCreateTensor(context,1, inception_c1_1x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_reduce_scale_W, dataFolder + "/weights/inception_c1_1x3_reduce_scale.f32"));
    vx_size inception_c1_1x3_reduce_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_reduce_scale_B;
    inception_c1_1x3_reduce_scale_B = vxCreateTensor(context,1, inception_c1_1x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_reduce_scale_B, dataFolder + "/bias/inception_c1_1x3_reduce_scale.f32"));
    vx_node inception_c1_1x3_reduce_bn_node;
    inception_c1_1x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x3_reduce, inception_c1_1x3_reduce_bn_W, inception_c1_1x3_reduce_bn_B, inception_c1_1x3_reduce_scale_W, inception_c1_1x3_reduce_scale_B, inception_c1_1x3_reduce_bn_eps, inception_c1_1x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_reduce_bn_node));

    // inception_c1_1x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x3_reduce_relu Layer
    vx_size inception_c1_1x3_reduce_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_reduce_relu;
    inception_c1_1x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_relu);
    vx_enum inception_c1_1x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x3_reduce_relu_param_a = 0;
    vx_float32 inception_c1_1x3_reduce_relu_param_b = 0;
    vx_node inception_c1_1x3_reduce_relu_node;
    inception_c1_1x3_reduce_relu_node = vxActivationLayer(graph, inception_c1_1x3_reduce_scale, inception_c1_1x3_reduce_relu_mode, inception_c1_1x3_reduce_relu_param_a, inception_c1_1x3_reduce_relu_param_b, inception_c1_1x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_reduce_relu_node));

    // inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0 Layer
    vx_size inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0;
    inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0);
    vx_node inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0_node;
    inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_c1_1x3_reduce_relu, (vx_reference)inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0_node));

    // inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1 Layer
    vx_size inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1;
    inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1);
    vx_node inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1_node;
    inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_c1_1x3_reduce_relu, (vx_reference)inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1);
    ERROR_CHECK_OBJECT(inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1_node));

    // inception_c1_1x3 Layer
    vx_size inception_c1_1x3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3;
    inception_c1_1x3 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3);
    vx_size inception_c1_1x3_W_dims[4] = { 3, 1, 384, 384 };
    vx_tensor inception_c1_1x3_W;
    inception_c1_1x3_W = vxCreateTensor(context,4, inception_c1_1x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_W, dataFolder + "/weights/inception_c1_1x3.f32"));
    vx_nn_convolution_params_t inception_c1_1x3_params;
    inception_c1_1x3_params.padding_x = 1;
    inception_c1_1x3_params.padding_y = 0;
    inception_c1_1x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x3_params.dilation_x = 0;
    inception_c1_1x3_params.dilation_y = 0;
    vx_node inception_c1_1x3_node;
    inception_c1_1x3_node = vxConvolutionLayer(graph, inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0, inception_c1_1x3_W, NULL, &inception_c1_1x3_params, sizeof(inception_c1_1x3_params ), inception_c1_1x3);
    ERROR_CHECK_OBJECT(inception_c1_1x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_node));

    // inception_c1_1x3_bn Layer
    vx_size inception_c1_1x3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_scale;
    inception_c1_1x3_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_scale);
    vx_size inception_c1_1x3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_1x3_bn_eps = 0.001;
    vx_tensor inception_c1_1x3_bn_W;
    inception_c1_1x3_bn_W = vxCreateTensor(context,1, inception_c1_1x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_bn_W, dataFolder + "/weights/inception_c1_1x3_bn.f32"));
    vx_size inception_c1_1x3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_bn_B;
    inception_c1_1x3_bn_B = vxCreateTensor(context,1, inception_c1_1x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_bn_B, dataFolder + "/bias/inception_c1_1x3_bn.f32"));
    vx_size inception_c1_1x3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_scale_W;
    inception_c1_1x3_scale_W = vxCreateTensor(context,1, inception_c1_1x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_scale_W, dataFolder + "/weights/inception_c1_1x3_scale.f32"));
    vx_size inception_c1_1x3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_scale_B;
    inception_c1_1x3_scale_B = vxCreateTensor(context,1, inception_c1_1x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_scale_B, dataFolder + "/bias/inception_c1_1x3_scale.f32"));
    vx_node inception_c1_1x3_bn_node;
    inception_c1_1x3_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x3, inception_c1_1x3_bn_W, inception_c1_1x3_bn_B, inception_c1_1x3_scale_W, inception_c1_1x3_scale_B, inception_c1_1x3_bn_eps, inception_c1_1x3_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_bn_node));

    // inception_c1_1x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x3_relu Layer
    vx_size inception_c1_1x3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_relu;
    inception_c1_1x3_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_relu);
    vx_enum inception_c1_1x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x3_relu_param_a = 0;
    vx_float32 inception_c1_1x3_relu_param_b = 0;
    vx_node inception_c1_1x3_relu_node;
    inception_c1_1x3_relu_node = vxActivationLayer(graph, inception_c1_1x3_scale, inception_c1_1x3_relu_mode, inception_c1_1x3_relu_param_a, inception_c1_1x3_relu_param_b, inception_c1_1x3_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_relu_node));

    // inception_c1_3x1 Layer
    vx_size inception_c1_3x1_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x1;
    inception_c1_3x1 = vxCreateVirtualTensor(graph,4, inception_c1_3x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1);
    vx_size inception_c1_3x1_W_dims[4] = { 1, 3, 384, 384 };
    vx_tensor inception_c1_3x1_W;
    inception_c1_3x1_W = vxCreateTensor(context,4, inception_c1_3x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_W, dataFolder + "/weights/inception_c1_3x1.f32"));
    vx_nn_convolution_params_t inception_c1_3x1_params;
    inception_c1_3x1_params.padding_x = 0;
    inception_c1_3x1_params.padding_y = 1;
    inception_c1_3x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x1_params.dilation_x = 0;
    inception_c1_3x1_params.dilation_y = 0;
    vx_node inception_c1_3x1_node;
    inception_c1_3x1_node = vxConvolutionLayer(graph, inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1, inception_c1_3x1_W, NULL, &inception_c1_3x1_params, sizeof(inception_c1_3x1_params ), inception_c1_3x1);
    ERROR_CHECK_OBJECT(inception_c1_3x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_node));

    // inception_c1_3x1_bn Layer
    vx_size inception_c1_3x1_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x1_scale;
    inception_c1_3x1_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_scale);
    vx_size inception_c1_3x1_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_3x1_bn_eps = 0.001;
    vx_tensor inception_c1_3x1_bn_W;
    inception_c1_3x1_bn_W = vxCreateTensor(context,1, inception_c1_3x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_bn_W, dataFolder + "/weights/inception_c1_3x1_bn.f32"));
    vx_size inception_c1_3x1_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_3x1_bn_B;
    inception_c1_3x1_bn_B = vxCreateTensor(context,1, inception_c1_3x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_bn_B, dataFolder + "/bias/inception_c1_3x1_bn.f32"));
    vx_size inception_c1_3x1_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_3x1_scale_W;
    inception_c1_3x1_scale_W = vxCreateTensor(context,1, inception_c1_3x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_scale_W, dataFolder + "/weights/inception_c1_3x1_scale.f32"));
    vx_size inception_c1_3x1_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_3x1_scale_B;
    inception_c1_3x1_scale_B = vxCreateTensor(context,1, inception_c1_3x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_scale_B, dataFolder + "/bias/inception_c1_3x1_scale.f32"));
    vx_node inception_c1_3x1_bn_node;
    inception_c1_3x1_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x1, inception_c1_3x1_bn_W, inception_c1_3x1_bn_B, inception_c1_3x1_scale_W, inception_c1_3x1_scale_B, inception_c1_3x1_bn_eps, inception_c1_3x1_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_bn_node));

    // inception_c1_3x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x1_relu Layer
    vx_size inception_c1_3x1_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x1_relu;
    inception_c1_3x1_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_relu);
    vx_enum inception_c1_3x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x1_relu_param_a = 0;
    vx_float32 inception_c1_3x1_relu_param_b = 0;
    vx_node inception_c1_3x1_relu_node;
    inception_c1_3x1_relu_node = vxActivationLayer(graph, inception_c1_3x1_scale, inception_c1_3x1_relu_mode, inception_c1_3x1_relu_param_a, inception_c1_3x1_relu_param_b, inception_c1_3x1_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_relu_node));

    // inception_c1_3x3_reduce Layer
    vx_size inception_c1_3x3_reduce_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c1_3x3_reduce;
    inception_c1_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_c1_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce);
    vx_size inception_c1_3x3_reduce_W_dims[4] = { 1, 1, 1280, 448 };
    vx_tensor inception_c1_3x3_reduce_W;
    inception_c1_3x3_reduce_W = vxCreateTensor(context,4, inception_c1_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_reduce_W, dataFolder + "/weights/inception_c1_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_c1_3x3_reduce_params;
    inception_c1_3x3_reduce_params.padding_x = 0;
    inception_c1_3x3_reduce_params.padding_y = 0;
    inception_c1_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x3_reduce_params.dilation_x = 0;
    inception_c1_3x3_reduce_params.dilation_y = 0;
    vx_node inception_c1_3x3_reduce_node;
    inception_c1_3x3_reduce_node = vxConvolutionLayer(graph, reduction_b_concat_reduction_b_concat_0_split_2, inception_c1_3x3_reduce_W, NULL, &inception_c1_3x3_reduce_params, sizeof(inception_c1_3x3_reduce_params ), inception_c1_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_reduce_node));

    // inception_c1_3x3_reduce_bn Layer
    vx_size inception_c1_3x3_reduce_scale_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c1_3x3_reduce_scale;
    inception_c1_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_scale);
    vx_size inception_c1_3x3_reduce_bn_W_dims[1] = { 448 };
    vx_float32 inception_c1_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_c1_3x3_reduce_bn_W;
    inception_c1_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_c1_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_reduce_bn_W, dataFolder + "/weights/inception_c1_3x3_reduce_bn.f32"));
    vx_size inception_c1_3x3_reduce_bn_B_dims[1] = { 448 };
    vx_tensor inception_c1_3x3_reduce_bn_B;
    inception_c1_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_c1_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_reduce_bn_B, dataFolder + "/bias/inception_c1_3x3_reduce_bn.f32"));
    vx_size inception_c1_3x3_reduce_scale_W_dims[1] = { 448 };
    vx_tensor inception_c1_3x3_reduce_scale_W;
    inception_c1_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_c1_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_reduce_scale_W, dataFolder + "/weights/inception_c1_3x3_reduce_scale.f32"));
    vx_size inception_c1_3x3_reduce_scale_B_dims[1] = { 448 };
    vx_tensor inception_c1_3x3_reduce_scale_B;
    inception_c1_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_c1_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_reduce_scale_B, dataFolder + "/bias/inception_c1_3x3_reduce_scale.f32"));
    vx_node inception_c1_3x3_reduce_bn_node;
    inception_c1_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x3_reduce, inception_c1_3x3_reduce_bn_W, inception_c1_3x3_reduce_bn_B, inception_c1_3x3_reduce_scale_W, inception_c1_3x3_reduce_scale_B, inception_c1_3x3_reduce_bn_eps, inception_c1_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_reduce_bn_node));

    // inception_c1_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x3_reduce_relu Layer
    vx_size inception_c1_3x3_reduce_relu_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c1_3x3_reduce_relu;
    inception_c1_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_relu);
    vx_enum inception_c1_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_c1_3x3_reduce_relu_param_b = 0;
    vx_node inception_c1_3x3_reduce_relu_node;
    inception_c1_3x3_reduce_relu_node = vxActivationLayer(graph, inception_c1_3x3_reduce_scale, inception_c1_3x3_reduce_relu_mode, inception_c1_3x3_reduce_relu_param_a, inception_c1_3x3_reduce_relu_param_b, inception_c1_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_reduce_relu_node));

    // inception_c1_3x3 Layer
    vx_size inception_c1_3x3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x3;
    inception_c1_3x3 = vxCreateVirtualTensor(graph,4, inception_c1_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3);
    vx_size inception_c1_3x3_W_dims[4] = { 3, 3, 448, 384 };
    vx_tensor inception_c1_3x3_W;
    inception_c1_3x3_W = vxCreateTensor(context,4, inception_c1_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_W, dataFolder + "/weights/inception_c1_3x3.f32"));
    vx_nn_convolution_params_t inception_c1_3x3_params;
    inception_c1_3x3_params.padding_x = 1;
    inception_c1_3x3_params.padding_y = 1;
    inception_c1_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x3_params.dilation_x = 0;
    inception_c1_3x3_params.dilation_y = 0;
    vx_node inception_c1_3x3_node;
    inception_c1_3x3_node = vxConvolutionLayer(graph, inception_c1_3x3_reduce_relu, inception_c1_3x3_W, NULL, &inception_c1_3x3_params, sizeof(inception_c1_3x3_params ), inception_c1_3x3);
    ERROR_CHECK_OBJECT(inception_c1_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_node));

    // inception_c1_3x3_bn Layer
    vx_size inception_c1_3x3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x3_scale;
    inception_c1_3x3_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_scale);
    vx_size inception_c1_3x3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_3x3_bn_eps = 0.001;
    vx_tensor inception_c1_3x3_bn_W;
    inception_c1_3x3_bn_W = vxCreateTensor(context,1, inception_c1_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_bn_W, dataFolder + "/weights/inception_c1_3x3_bn.f32"));
    vx_size inception_c1_3x3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_3x3_bn_B;
    inception_c1_3x3_bn_B = vxCreateTensor(context,1, inception_c1_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_bn_B, dataFolder + "/bias/inception_c1_3x3_bn.f32"));
    vx_size inception_c1_3x3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_3x3_scale_W;
    inception_c1_3x3_scale_W = vxCreateTensor(context,1, inception_c1_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_scale_W, dataFolder + "/weights/inception_c1_3x3_scale.f32"));
    vx_size inception_c1_3x3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_3x3_scale_B;
    inception_c1_3x3_scale_B = vxCreateTensor(context,1, inception_c1_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x3_scale_B, dataFolder + "/bias/inception_c1_3x3_scale.f32"));
    vx_node inception_c1_3x3_bn_node;
    inception_c1_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x3, inception_c1_3x3_bn_W, inception_c1_3x3_bn_B, inception_c1_3x3_scale_W, inception_c1_3x3_scale_B, inception_c1_3x3_bn_eps, inception_c1_3x3_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_bn_node));

    // inception_c1_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x3_relu Layer
    vx_size inception_c1_3x3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x3_relu;
    inception_c1_3x3_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_relu);
    vx_enum inception_c1_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x3_relu_param_a = 0;
    vx_float32 inception_c1_3x3_relu_param_b = 0;
    vx_node inception_c1_3x3_relu_node;
    inception_c1_3x3_relu_node = vxActivationLayer(graph, inception_c1_3x3_scale, inception_c1_3x3_relu_mode, inception_c1_3x3_relu_param_a, inception_c1_3x3_relu_param_b, inception_c1_3x3_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_relu_node));

    // inception_c1_3x3_inception_c1_3x3_relu_0_split_0 Layer
    vx_size inception_c1_3x3_inception_c1_3x3_relu_0_split_0_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x3_inception_c1_3x3_relu_0_split_0;
    inception_c1_3x3_inception_c1_3x3_relu_0_split_0 = vxCreateVirtualTensor(graph,4, inception_c1_3x3_inception_c1_3x3_relu_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_inception_c1_3x3_relu_0_split_0);
    vx_node inception_c1_3x3_inception_c1_3x3_relu_0_split_0_node;
    inception_c1_3x3_inception_c1_3x3_relu_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_c1_3x3_relu, (vx_reference)inception_c1_3x3_inception_c1_3x3_relu_0_split_0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_inception_c1_3x3_relu_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_inception_c1_3x3_relu_0_split_0_node));

    // inception_c1_3x3_inception_c1_3x3_relu_0_split_1 Layer
    vx_size inception_c1_3x3_inception_c1_3x3_relu_0_split_1_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x3_inception_c1_3x3_relu_0_split_1;
    inception_c1_3x3_inception_c1_3x3_relu_0_split_1 = vxCreateVirtualTensor(graph,4, inception_c1_3x3_inception_c1_3x3_relu_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x3_inception_c1_3x3_relu_0_split_1);
    vx_node inception_c1_3x3_inception_c1_3x3_relu_0_split_1_node;
    inception_c1_3x3_inception_c1_3x3_relu_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_c1_3x3_relu, (vx_reference)inception_c1_3x3_inception_c1_3x3_relu_0_split_1);
    ERROR_CHECK_OBJECT(inception_c1_3x3_inception_c1_3x3_relu_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x3_inception_c1_3x3_relu_0_split_1_node));

    // inception_c1_1x3_2 Layer
    vx_size inception_c1_1x3_2_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_2;
    inception_c1_1x3_2 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2);
    vx_size inception_c1_1x3_2_W_dims[4] = { 3, 1, 384, 384 };
    vx_tensor inception_c1_1x3_2_W;
    inception_c1_1x3_2_W = vxCreateTensor(context,4, inception_c1_1x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_W, dataFolder + "/weights/inception_c1_1x3_2.f32"));
    vx_nn_convolution_params_t inception_c1_1x3_2_params;
    inception_c1_1x3_2_params.padding_x = 1;
    inception_c1_1x3_2_params.padding_y = 0;
    inception_c1_1x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x3_2_params.dilation_x = 0;
    inception_c1_1x3_2_params.dilation_y = 0;
    vx_node inception_c1_1x3_2_node;
    inception_c1_1x3_2_node = vxConvolutionLayer(graph, inception_c1_3x3_inception_c1_3x3_relu_0_split_0, inception_c1_1x3_2_W, NULL, &inception_c1_1x3_2_params, sizeof(inception_c1_1x3_2_params ), inception_c1_1x3_2);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_2_node));

    // inception_c1_1x3_2_bn Layer
    vx_size inception_c1_1x3_2_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_2_scale;
    inception_c1_1x3_2_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_scale);
    vx_size inception_c1_1x3_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_1x3_2_bn_eps = 0.001;
    vx_tensor inception_c1_1x3_2_bn_W;
    inception_c1_1x3_2_bn_W = vxCreateTensor(context,1, inception_c1_1x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_bn_W, dataFolder + "/weights/inception_c1_1x3_2_bn.f32"));
    vx_size inception_c1_1x3_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_2_bn_B;
    inception_c1_1x3_2_bn_B = vxCreateTensor(context,1, inception_c1_1x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_bn_B, dataFolder + "/bias/inception_c1_1x3_2_bn.f32"));
    vx_size inception_c1_1x3_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_2_scale_W;
    inception_c1_1x3_2_scale_W = vxCreateTensor(context,1, inception_c1_1x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_scale_W, dataFolder + "/weights/inception_c1_1x3_2_scale.f32"));
    vx_size inception_c1_1x3_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x3_2_scale_B;
    inception_c1_1x3_2_scale_B = vxCreateTensor(context,1, inception_c1_1x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_scale_B, dataFolder + "/bias/inception_c1_1x3_2_scale.f32"));
    vx_node inception_c1_1x3_2_bn_node;
    inception_c1_1x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x3_2, inception_c1_1x3_2_bn_W, inception_c1_1x3_2_bn_B, inception_c1_1x3_2_scale_W, inception_c1_1x3_2_scale_B, inception_c1_1x3_2_bn_eps, inception_c1_1x3_2_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_2_bn_node));

    // inception_c1_1x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x3_2_relu Layer
    vx_size inception_c1_1x3_2_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x3_2_relu;
    inception_c1_1x3_2_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_relu);
    vx_enum inception_c1_1x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x3_2_relu_param_a = 0;
    vx_float32 inception_c1_1x3_2_relu_param_b = 0;
    vx_node inception_c1_1x3_2_relu_node;
    inception_c1_1x3_2_relu_node = vxActivationLayer(graph, inception_c1_1x3_2_scale, inception_c1_1x3_2_relu_mode, inception_c1_1x3_2_relu_param_a, inception_c1_1x3_2_relu_param_b, inception_c1_1x3_2_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_2_relu_node));

    // inception_c1_3x1_2 Layer
    vx_size inception_c1_3x1_2_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x1_2;
    inception_c1_3x1_2 = vxCreateVirtualTensor(graph,4, inception_c1_3x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2);
    vx_size inception_c1_3x1_2_W_dims[4] = { 1, 3, 384, 384 };
    vx_tensor inception_c1_3x1_2_W;
    inception_c1_3x1_2_W = vxCreateTensor(context,4, inception_c1_3x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_W, dataFolder + "/weights/inception_c1_3x1_2.f32"));
    vx_nn_convolution_params_t inception_c1_3x1_2_params;
    inception_c1_3x1_2_params.padding_x = 0;
    inception_c1_3x1_2_params.padding_y = 1;
    inception_c1_3x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x1_2_params.dilation_x = 0;
    inception_c1_3x1_2_params.dilation_y = 0;
    vx_node inception_c1_3x1_2_node;
    inception_c1_3x1_2_node = vxConvolutionLayer(graph, inception_c1_3x3_inception_c1_3x3_relu_0_split_1, inception_c1_3x1_2_W, NULL, &inception_c1_3x1_2_params, sizeof(inception_c1_3x1_2_params ), inception_c1_3x1_2);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_2_node));

    // inception_c1_3x1_2_bn Layer
    vx_size inception_c1_3x1_2_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x1_2_scale;
    inception_c1_3x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_scale);
    vx_size inception_c1_3x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_3x1_2_bn_eps = 0.001;
    vx_tensor inception_c1_3x1_2_bn_W;
    inception_c1_3x1_2_bn_W = vxCreateTensor(context,1, inception_c1_3x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_bn_W, dataFolder + "/weights/inception_c1_3x1_2_bn.f32"));
    vx_size inception_c1_3x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_3x1_2_bn_B;
    inception_c1_3x1_2_bn_B = vxCreateTensor(context,1, inception_c1_3x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_bn_B, dataFolder + "/bias/inception_c1_3x1_2_bn.f32"));
    vx_size inception_c1_3x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_3x1_2_scale_W;
    inception_c1_3x1_2_scale_W = vxCreateTensor(context,1, inception_c1_3x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_scale_W, dataFolder + "/weights/inception_c1_3x1_2_scale.f32"));
    vx_size inception_c1_3x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_3x1_2_scale_B;
    inception_c1_3x1_2_scale_B = vxCreateTensor(context,1, inception_c1_3x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_scale_B, dataFolder + "/bias/inception_c1_3x1_2_scale.f32"));
    vx_node inception_c1_3x1_2_bn_node;
    inception_c1_3x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x1_2, inception_c1_3x1_2_bn_W, inception_c1_3x1_2_bn_B, inception_c1_3x1_2_scale_W, inception_c1_3x1_2_scale_B, inception_c1_3x1_2_bn_eps, inception_c1_3x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_2_bn_node));

    // inception_c1_3x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x1_2_relu Layer
    vx_size inception_c1_3x1_2_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_3x1_2_relu;
    inception_c1_3x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_relu);
    vx_enum inception_c1_3x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x1_2_relu_param_a = 0;
    vx_float32 inception_c1_3x1_2_relu_param_b = 0;
    vx_node inception_c1_3x1_2_relu_node;
    inception_c1_3x1_2_relu_node = vxActivationLayer(graph, inception_c1_3x1_2_scale, inception_c1_3x1_2_relu_mode, inception_c1_3x1_2_relu_param_a, inception_c1_3x1_2_relu_param_b, inception_c1_3x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_2_relu_node));

    // inception_c1_pool Layer
    vx_size inception_c1_pool_dims[4] = { 8, 8, 1280, 1 };
    vx_tensor inception_c1_pool;
    inception_c1_pool = vxCreateVirtualTensor(graph,4, inception_c1_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_pool);
    vx_enum inception_c1_pool_type = VX_NN_POOLING_AVG;
    vx_size inception_c1_pool_kernel_w = 3;
    vx_size inception_c1_pool_kernel_h = 3;
    vx_size inception_c1_pool_pad_w = 1;
    vx_size inception_c1_pool_pad_h = 1;
    vx_enum inception_c1_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_c1_pool_node;
    inception_c1_pool_node = vxPoolingLayer(graph, reduction_b_concat_reduction_b_concat_0_split_3, inception_c1_pool_type, inception_c1_pool_kernel_w, inception_c1_pool_kernel_h, inception_c1_pool_pad_w, inception_c1_pool_pad_h, inception_c1_pool_roundPolicy, inception_c1_pool );
    ERROR_CHECK_OBJECT(inception_c1_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_pool_node));

    // inception_c1_1x1 Layer
    vx_size inception_c1_1x1_dims[4] = { 8, 8, 192, 1 };
    vx_tensor inception_c1_1x1;
    inception_c1_1x1 = vxCreateVirtualTensor(graph,4, inception_c1_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1);
    vx_size inception_c1_1x1_W_dims[4] = { 1, 1, 1280, 192 };
    vx_tensor inception_c1_1x1_W;
    inception_c1_1x1_W = vxCreateTensor(context,4, inception_c1_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_W, dataFolder + "/weights/inception_c1_1x1.f32"));
    vx_nn_convolution_params_t inception_c1_1x1_params;
    inception_c1_1x1_params.padding_x = 0;
    inception_c1_1x1_params.padding_y = 0;
    inception_c1_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x1_params.dilation_x = 0;
    inception_c1_1x1_params.dilation_y = 0;
    vx_node inception_c1_1x1_node;
    inception_c1_1x1_node = vxConvolutionLayer(graph, inception_c1_pool, inception_c1_1x1_W, NULL, &inception_c1_1x1_params, sizeof(inception_c1_1x1_params ), inception_c1_1x1);
    ERROR_CHECK_OBJECT(inception_c1_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_node));

    // inception_c1_1x1_bn Layer
    vx_size inception_c1_1x1_scale_dims[4] = { 8, 8, 192, 1 };
    vx_tensor inception_c1_1x1_scale;
    inception_c1_1x1_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_scale);
    vx_size inception_c1_1x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_c1_1x1_bn_eps = 0.001;
    vx_tensor inception_c1_1x1_bn_W;
    inception_c1_1x1_bn_W = vxCreateTensor(context,1, inception_c1_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_bn_W, dataFolder + "/weights/inception_c1_1x1_bn.f32"));
    vx_size inception_c1_1x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_c1_1x1_bn_B;
    inception_c1_1x1_bn_B = vxCreateTensor(context,1, inception_c1_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_bn_B, dataFolder + "/bias/inception_c1_1x1_bn.f32"));
    vx_size inception_c1_1x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_c1_1x1_scale_W;
    inception_c1_1x1_scale_W = vxCreateTensor(context,1, inception_c1_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_scale_W, dataFolder + "/weights/inception_c1_1x1_scale.f32"));
    vx_size inception_c1_1x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_c1_1x1_scale_B;
    inception_c1_1x1_scale_B = vxCreateTensor(context,1, inception_c1_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_scale_B, dataFolder + "/bias/inception_c1_1x1_scale.f32"));
    vx_node inception_c1_1x1_bn_node;
    inception_c1_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x1, inception_c1_1x1_bn_W, inception_c1_1x1_bn_B, inception_c1_1x1_scale_W, inception_c1_1x1_scale_B, inception_c1_1x1_bn_eps, inception_c1_1x1_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_bn_node));

    // inception_c1_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x1_relu Layer
    vx_size inception_c1_1x1_relu_dims[4] = { 8, 8, 192, 1 };
    vx_tensor inception_c1_1x1_relu;
    inception_c1_1x1_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_relu);
    vx_enum inception_c1_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x1_relu_param_a = 0;
    vx_float32 inception_c1_1x1_relu_param_b = 0;
    vx_node inception_c1_1x1_relu_node;
    inception_c1_1x1_relu_node = vxActivationLayer(graph, inception_c1_1x1_scale, inception_c1_1x1_relu_mode, inception_c1_1x1_relu_param_a, inception_c1_1x1_relu_param_b, inception_c1_1x1_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_relu_node));

    // inception_c1_concat Layer
    vx_size inception_c1_concat_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c1_concat;
    inception_c1_concat = vxCreateVirtualTensor(graph,4, inception_c1_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_concat);
    vx_node inception_c1_concat_node;
    inception_c1_concat_node = vxConcatLayer(graph, inception_c1_concat, inception_c1_1x1_2_relu, inception_c1_1x3_relu, inception_c1_3x1_relu, inception_c1_1x3_2_relu, inception_c1_3x1_2_relu, inception_c1_1x1_relu, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_c1_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_concat_node));

    // inception_c1_concat_inception_c1_concat_0_split_0 Layer
    vx_size inception_c1_concat_inception_c1_concat_0_split_0_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c1_concat_inception_c1_concat_0_split_0;
    inception_c1_concat_inception_c1_concat_0_split_0 = vxCreateVirtualTensor(graph,4, inception_c1_concat_inception_c1_concat_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_0);
    vx_node inception_c1_concat_inception_c1_concat_0_split_0_node;
    inception_c1_concat_inception_c1_concat_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_c1_concat, (vx_reference)inception_c1_concat_inception_c1_concat_0_split_0);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_concat_inception_c1_concat_0_split_0_node));

    // inception_c1_concat_inception_c1_concat_0_split_1 Layer
    vx_size inception_c1_concat_inception_c1_concat_0_split_1_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c1_concat_inception_c1_concat_0_split_1;
    inception_c1_concat_inception_c1_concat_0_split_1 = vxCreateVirtualTensor(graph,4, inception_c1_concat_inception_c1_concat_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_1);
    vx_node inception_c1_concat_inception_c1_concat_0_split_1_node;
    inception_c1_concat_inception_c1_concat_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_c1_concat, (vx_reference)inception_c1_concat_inception_c1_concat_0_split_1);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_concat_inception_c1_concat_0_split_1_node));

    // inception_c1_concat_inception_c1_concat_0_split_2 Layer
    vx_size inception_c1_concat_inception_c1_concat_0_split_2_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c1_concat_inception_c1_concat_0_split_2;
    inception_c1_concat_inception_c1_concat_0_split_2 = vxCreateVirtualTensor(graph,4, inception_c1_concat_inception_c1_concat_0_split_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_2);
    vx_node inception_c1_concat_inception_c1_concat_0_split_2_node;
    inception_c1_concat_inception_c1_concat_0_split_2_node = vxCopyNode( graph, (vx_reference)inception_c1_concat, (vx_reference)inception_c1_concat_inception_c1_concat_0_split_2);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_concat_inception_c1_concat_0_split_2_node));

    // inception_c1_concat_inception_c1_concat_0_split_3 Layer
    vx_size inception_c1_concat_inception_c1_concat_0_split_3_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c1_concat_inception_c1_concat_0_split_3;
    inception_c1_concat_inception_c1_concat_0_split_3 = vxCreateVirtualTensor(graph,4, inception_c1_concat_inception_c1_concat_0_split_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_3);
    vx_node inception_c1_concat_inception_c1_concat_0_split_3_node;
    inception_c1_concat_inception_c1_concat_0_split_3_node = vxCopyNode( graph, (vx_reference)inception_c1_concat, (vx_reference)inception_c1_concat_inception_c1_concat_0_split_3);
    ERROR_CHECK_OBJECT(inception_c1_concat_inception_c1_concat_0_split_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_concat_inception_c1_concat_0_split_3_node));

    // inception_c2_1x1_2 Layer
    vx_size inception_c2_1x1_2_dims[4] = { 8, 8, 320, 1 };
    vx_tensor inception_c2_1x1_2;
    inception_c2_1x1_2 = vxCreateVirtualTensor(graph,4, inception_c2_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2);
    vx_size inception_c2_1x1_2_W_dims[4] = { 1, 1, 2048, 320 };
    vx_tensor inception_c2_1x1_2_W;
    inception_c2_1x1_2_W = vxCreateTensor(context,4, inception_c2_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_W, dataFolder + "/weights/inception_c2_1x1_2.f32"));
    vx_nn_convolution_params_t inception_c2_1x1_2_params;
    inception_c2_1x1_2_params.padding_x = 0;
    inception_c2_1x1_2_params.padding_y = 0;
    inception_c2_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x1_2_params.dilation_x = 0;
    inception_c2_1x1_2_params.dilation_y = 0;
    vx_node inception_c2_1x1_2_node;
    inception_c2_1x1_2_node = vxConvolutionLayer(graph, inception_c1_concat_inception_c1_concat_0_split_0, inception_c2_1x1_2_W, NULL, &inception_c2_1x1_2_params, sizeof(inception_c2_1x1_2_params ), inception_c2_1x1_2);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_2_node));

    // inception_c2_1x1_2_bn Layer
    vx_size inception_c2_1x1_2_scale_dims[4] = { 8, 8, 320, 1 };
    vx_tensor inception_c2_1x1_2_scale;
    inception_c2_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_scale);
    vx_size inception_c2_1x1_2_bn_W_dims[1] = { 320 };
    vx_float32 inception_c2_1x1_2_bn_eps = 0.001;
    vx_tensor inception_c2_1x1_2_bn_W;
    inception_c2_1x1_2_bn_W = vxCreateTensor(context,1, inception_c2_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_bn_W, dataFolder + "/weights/inception_c2_1x1_2_bn.f32"));
    vx_size inception_c2_1x1_2_bn_B_dims[1] = { 320 };
    vx_tensor inception_c2_1x1_2_bn_B;
    inception_c2_1x1_2_bn_B = vxCreateTensor(context,1, inception_c2_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_bn_B, dataFolder + "/bias/inception_c2_1x1_2_bn.f32"));
    vx_size inception_c2_1x1_2_scale_W_dims[1] = { 320 };
    vx_tensor inception_c2_1x1_2_scale_W;
    inception_c2_1x1_2_scale_W = vxCreateTensor(context,1, inception_c2_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_scale_W, dataFolder + "/weights/inception_c2_1x1_2_scale.f32"));
    vx_size inception_c2_1x1_2_scale_B_dims[1] = { 320 };
    vx_tensor inception_c2_1x1_2_scale_B;
    inception_c2_1x1_2_scale_B = vxCreateTensor(context,1, inception_c2_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_scale_B, dataFolder + "/bias/inception_c2_1x1_2_scale.f32"));
    vx_node inception_c2_1x1_2_bn_node;
    inception_c2_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x1_2, inception_c2_1x1_2_bn_W, inception_c2_1x1_2_bn_B, inception_c2_1x1_2_scale_W, inception_c2_1x1_2_scale_B, inception_c2_1x1_2_bn_eps, inception_c2_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_2_bn_node));

    // inception_c2_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x1_2_relu Layer
    vx_size inception_c2_1x1_2_relu_dims[4] = { 8, 8, 320, 1 };
    vx_tensor inception_c2_1x1_2_relu;
    inception_c2_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_relu);
    vx_enum inception_c2_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x1_2_relu_param_a = 0;
    vx_float32 inception_c2_1x1_2_relu_param_b = 0;
    vx_node inception_c2_1x1_2_relu_node;
    inception_c2_1x1_2_relu_node = vxActivationLayer(graph, inception_c2_1x1_2_scale, inception_c2_1x1_2_relu_mode, inception_c2_1x1_2_relu_param_a, inception_c2_1x1_2_relu_param_b, inception_c2_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_2_relu_node));

    // inception_c2_1x3_reduce Layer
    vx_size inception_c2_1x3_reduce_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_reduce;
    inception_c2_1x3_reduce = vxCreateVirtualTensor(graph,4, inception_c2_1x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce);
    vx_size inception_c2_1x3_reduce_W_dims[4] = { 1, 1, 2048, 384 };
    vx_tensor inception_c2_1x3_reduce_W;
    inception_c2_1x3_reduce_W = vxCreateTensor(context,4, inception_c2_1x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_reduce_W, dataFolder + "/weights/inception_c2_1x3_reduce.f32"));
    vx_nn_convolution_params_t inception_c2_1x3_reduce_params;
    inception_c2_1x3_reduce_params.padding_x = 0;
    inception_c2_1x3_reduce_params.padding_y = 0;
    inception_c2_1x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x3_reduce_params.dilation_x = 0;
    inception_c2_1x3_reduce_params.dilation_y = 0;
    vx_node inception_c2_1x3_reduce_node;
    inception_c2_1x3_reduce_node = vxConvolutionLayer(graph, inception_c1_concat_inception_c1_concat_0_split_1, inception_c2_1x3_reduce_W, NULL, &inception_c2_1x3_reduce_params, sizeof(inception_c2_1x3_reduce_params ), inception_c2_1x3_reduce);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_reduce_node));

    // inception_c2_1x3_reduce_bn Layer
    vx_size inception_c2_1x3_reduce_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_reduce_scale;
    inception_c2_1x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_scale);
    vx_size inception_c2_1x3_reduce_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_1x3_reduce_bn_eps = 0.001;
    vx_tensor inception_c2_1x3_reduce_bn_W;
    inception_c2_1x3_reduce_bn_W = vxCreateTensor(context,1, inception_c2_1x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_reduce_bn_W, dataFolder + "/weights/inception_c2_1x3_reduce_bn.f32"));
    vx_size inception_c2_1x3_reduce_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_reduce_bn_B;
    inception_c2_1x3_reduce_bn_B = vxCreateTensor(context,1, inception_c2_1x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_reduce_bn_B, dataFolder + "/bias/inception_c2_1x3_reduce_bn.f32"));
    vx_size inception_c2_1x3_reduce_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_reduce_scale_W;
    inception_c2_1x3_reduce_scale_W = vxCreateTensor(context,1, inception_c2_1x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_reduce_scale_W, dataFolder + "/weights/inception_c2_1x3_reduce_scale.f32"));
    vx_size inception_c2_1x3_reduce_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_reduce_scale_B;
    inception_c2_1x3_reduce_scale_B = vxCreateTensor(context,1, inception_c2_1x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_reduce_scale_B, dataFolder + "/bias/inception_c2_1x3_reduce_scale.f32"));
    vx_node inception_c2_1x3_reduce_bn_node;
    inception_c2_1x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x3_reduce, inception_c2_1x3_reduce_bn_W, inception_c2_1x3_reduce_bn_B, inception_c2_1x3_reduce_scale_W, inception_c2_1x3_reduce_scale_B, inception_c2_1x3_reduce_bn_eps, inception_c2_1x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_reduce_bn_node));

    // inception_c2_1x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x3_reduce_relu Layer
    vx_size inception_c2_1x3_reduce_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_reduce_relu;
    inception_c2_1x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_relu);
    vx_enum inception_c2_1x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x3_reduce_relu_param_a = 0;
    vx_float32 inception_c2_1x3_reduce_relu_param_b = 0;
    vx_node inception_c2_1x3_reduce_relu_node;
    inception_c2_1x3_reduce_relu_node = vxActivationLayer(graph, inception_c2_1x3_reduce_scale, inception_c2_1x3_reduce_relu_mode, inception_c2_1x3_reduce_relu_param_a, inception_c2_1x3_reduce_relu_param_b, inception_c2_1x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_reduce_relu_node));

    // inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0 Layer
    vx_size inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0;
    inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0);
    vx_node inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0_node;
    inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_c2_1x3_reduce_relu, (vx_reference)inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0_node));

    // inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1 Layer
    vx_size inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1;
    inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1);
    vx_node inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1_node;
    inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_c2_1x3_reduce_relu, (vx_reference)inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1);
    ERROR_CHECK_OBJECT(inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1_node));

    // inception_c2_1x3 Layer
    vx_size inception_c2_1x3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3;
    inception_c2_1x3 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3);
    vx_size inception_c2_1x3_W_dims[4] = { 3, 1, 384, 384 };
    vx_tensor inception_c2_1x3_W;
    inception_c2_1x3_W = vxCreateTensor(context,4, inception_c2_1x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_W, dataFolder + "/weights/inception_c2_1x3.f32"));
    vx_nn_convolution_params_t inception_c2_1x3_params;
    inception_c2_1x3_params.padding_x = 1;
    inception_c2_1x3_params.padding_y = 0;
    inception_c2_1x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x3_params.dilation_x = 0;
    inception_c2_1x3_params.dilation_y = 0;
    vx_node inception_c2_1x3_node;
    inception_c2_1x3_node = vxConvolutionLayer(graph, inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0, inception_c2_1x3_W, NULL, &inception_c2_1x3_params, sizeof(inception_c2_1x3_params ), inception_c2_1x3);
    ERROR_CHECK_OBJECT(inception_c2_1x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_node));

    // inception_c2_1x3_bn Layer
    vx_size inception_c2_1x3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_scale;
    inception_c2_1x3_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_scale);
    vx_size inception_c2_1x3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_1x3_bn_eps = 0.001;
    vx_tensor inception_c2_1x3_bn_W;
    inception_c2_1x3_bn_W = vxCreateTensor(context,1, inception_c2_1x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_bn_W, dataFolder + "/weights/inception_c2_1x3_bn.f32"));
    vx_size inception_c2_1x3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_bn_B;
    inception_c2_1x3_bn_B = vxCreateTensor(context,1, inception_c2_1x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_bn_B, dataFolder + "/bias/inception_c2_1x3_bn.f32"));
    vx_size inception_c2_1x3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_scale_W;
    inception_c2_1x3_scale_W = vxCreateTensor(context,1, inception_c2_1x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_scale_W, dataFolder + "/weights/inception_c2_1x3_scale.f32"));
    vx_size inception_c2_1x3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_scale_B;
    inception_c2_1x3_scale_B = vxCreateTensor(context,1, inception_c2_1x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_scale_B, dataFolder + "/bias/inception_c2_1x3_scale.f32"));
    vx_node inception_c2_1x3_bn_node;
    inception_c2_1x3_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x3, inception_c2_1x3_bn_W, inception_c2_1x3_bn_B, inception_c2_1x3_scale_W, inception_c2_1x3_scale_B, inception_c2_1x3_bn_eps, inception_c2_1x3_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_bn_node));

    // inception_c2_1x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x3_relu Layer
    vx_size inception_c2_1x3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_relu;
    inception_c2_1x3_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_relu);
    vx_enum inception_c2_1x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x3_relu_param_a = 0;
    vx_float32 inception_c2_1x3_relu_param_b = 0;
    vx_node inception_c2_1x3_relu_node;
    inception_c2_1x3_relu_node = vxActivationLayer(graph, inception_c2_1x3_scale, inception_c2_1x3_relu_mode, inception_c2_1x3_relu_param_a, inception_c2_1x3_relu_param_b, inception_c2_1x3_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_relu_node));

    // inception_c2_3x1 Layer
    vx_size inception_c2_3x1_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x1;
    inception_c2_3x1 = vxCreateVirtualTensor(graph,4, inception_c2_3x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1);
    vx_size inception_c2_3x1_W_dims[4] = { 1, 3, 384, 384 };
    vx_tensor inception_c2_3x1_W;
    inception_c2_3x1_W = vxCreateTensor(context,4, inception_c2_3x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_W, dataFolder + "/weights/inception_c2_3x1.f32"));
    vx_nn_convolution_params_t inception_c2_3x1_params;
    inception_c2_3x1_params.padding_x = 0;
    inception_c2_3x1_params.padding_y = 1;
    inception_c2_3x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x1_params.dilation_x = 0;
    inception_c2_3x1_params.dilation_y = 0;
    vx_node inception_c2_3x1_node;
    inception_c2_3x1_node = vxConvolutionLayer(graph, inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1, inception_c2_3x1_W, NULL, &inception_c2_3x1_params, sizeof(inception_c2_3x1_params ), inception_c2_3x1);
    ERROR_CHECK_OBJECT(inception_c2_3x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_node));

    // inception_c2_3x1_bn Layer
    vx_size inception_c2_3x1_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x1_scale;
    inception_c2_3x1_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_scale);
    vx_size inception_c2_3x1_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_3x1_bn_eps = 0.001;
    vx_tensor inception_c2_3x1_bn_W;
    inception_c2_3x1_bn_W = vxCreateTensor(context,1, inception_c2_3x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_bn_W, dataFolder + "/weights/inception_c2_3x1_bn.f32"));
    vx_size inception_c2_3x1_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_3x1_bn_B;
    inception_c2_3x1_bn_B = vxCreateTensor(context,1, inception_c2_3x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_bn_B, dataFolder + "/bias/inception_c2_3x1_bn.f32"));
    vx_size inception_c2_3x1_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_3x1_scale_W;
    inception_c2_3x1_scale_W = vxCreateTensor(context,1, inception_c2_3x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_scale_W, dataFolder + "/weights/inception_c2_3x1_scale.f32"));
    vx_size inception_c2_3x1_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_3x1_scale_B;
    inception_c2_3x1_scale_B = vxCreateTensor(context,1, inception_c2_3x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_scale_B, dataFolder + "/bias/inception_c2_3x1_scale.f32"));
    vx_node inception_c2_3x1_bn_node;
    inception_c2_3x1_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x1, inception_c2_3x1_bn_W, inception_c2_3x1_bn_B, inception_c2_3x1_scale_W, inception_c2_3x1_scale_B, inception_c2_3x1_bn_eps, inception_c2_3x1_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_bn_node));

    // inception_c2_3x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x1_relu Layer
    vx_size inception_c2_3x1_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x1_relu;
    inception_c2_3x1_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_relu);
    vx_enum inception_c2_3x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x1_relu_param_a = 0;
    vx_float32 inception_c2_3x1_relu_param_b = 0;
    vx_node inception_c2_3x1_relu_node;
    inception_c2_3x1_relu_node = vxActivationLayer(graph, inception_c2_3x1_scale, inception_c2_3x1_relu_mode, inception_c2_3x1_relu_param_a, inception_c2_3x1_relu_param_b, inception_c2_3x1_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_relu_node));

    // inception_c2_3x3_reduce Layer
    vx_size inception_c2_3x3_reduce_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c2_3x3_reduce;
    inception_c2_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_c2_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce);
    vx_size inception_c2_3x3_reduce_W_dims[4] = { 1, 1, 2048, 448 };
    vx_tensor inception_c2_3x3_reduce_W;
    inception_c2_3x3_reduce_W = vxCreateTensor(context,4, inception_c2_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_reduce_W, dataFolder + "/weights/inception_c2_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_c2_3x3_reduce_params;
    inception_c2_3x3_reduce_params.padding_x = 0;
    inception_c2_3x3_reduce_params.padding_y = 0;
    inception_c2_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x3_reduce_params.dilation_x = 0;
    inception_c2_3x3_reduce_params.dilation_y = 0;
    vx_node inception_c2_3x3_reduce_node;
    inception_c2_3x3_reduce_node = vxConvolutionLayer(graph, inception_c1_concat_inception_c1_concat_0_split_2, inception_c2_3x3_reduce_W, NULL, &inception_c2_3x3_reduce_params, sizeof(inception_c2_3x3_reduce_params ), inception_c2_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_reduce_node));

    // inception_c2_3x3_reduce_bn Layer
    vx_size inception_c2_3x3_reduce_scale_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c2_3x3_reduce_scale;
    inception_c2_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_scale);
    vx_size inception_c2_3x3_reduce_bn_W_dims[1] = { 448 };
    vx_float32 inception_c2_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_c2_3x3_reduce_bn_W;
    inception_c2_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_c2_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_reduce_bn_W, dataFolder + "/weights/inception_c2_3x3_reduce_bn.f32"));
    vx_size inception_c2_3x3_reduce_bn_B_dims[1] = { 448 };
    vx_tensor inception_c2_3x3_reduce_bn_B;
    inception_c2_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_c2_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_reduce_bn_B, dataFolder + "/bias/inception_c2_3x3_reduce_bn.f32"));
    vx_size inception_c2_3x3_reduce_scale_W_dims[1] = { 448 };
    vx_tensor inception_c2_3x3_reduce_scale_W;
    inception_c2_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_c2_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_reduce_scale_W, dataFolder + "/weights/inception_c2_3x3_reduce_scale.f32"));
    vx_size inception_c2_3x3_reduce_scale_B_dims[1] = { 448 };
    vx_tensor inception_c2_3x3_reduce_scale_B;
    inception_c2_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_c2_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_reduce_scale_B, dataFolder + "/bias/inception_c2_3x3_reduce_scale.f32"));
    vx_node inception_c2_3x3_reduce_bn_node;
    inception_c2_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x3_reduce, inception_c2_3x3_reduce_bn_W, inception_c2_3x3_reduce_bn_B, inception_c2_3x3_reduce_scale_W, inception_c2_3x3_reduce_scale_B, inception_c2_3x3_reduce_bn_eps, inception_c2_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_reduce_bn_node));

    // inception_c2_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x3_reduce_relu Layer
    vx_size inception_c2_3x3_reduce_relu_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c2_3x3_reduce_relu;
    inception_c2_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_relu);
    vx_enum inception_c2_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_c2_3x3_reduce_relu_param_b = 0;
    vx_node inception_c2_3x3_reduce_relu_node;
    inception_c2_3x3_reduce_relu_node = vxActivationLayer(graph, inception_c2_3x3_reduce_scale, inception_c2_3x3_reduce_relu_mode, inception_c2_3x3_reduce_relu_param_a, inception_c2_3x3_reduce_relu_param_b, inception_c2_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_reduce_relu_node));

    // inception_c2_3x3 Layer
    vx_size inception_c2_3x3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x3;
    inception_c2_3x3 = vxCreateVirtualTensor(graph,4, inception_c2_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3);
    vx_size inception_c2_3x3_W_dims[4] = { 3, 3, 448, 384 };
    vx_tensor inception_c2_3x3_W;
    inception_c2_3x3_W = vxCreateTensor(context,4, inception_c2_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_W, dataFolder + "/weights/inception_c2_3x3.f32"));
    vx_nn_convolution_params_t inception_c2_3x3_params;
    inception_c2_3x3_params.padding_x = 1;
    inception_c2_3x3_params.padding_y = 1;
    inception_c2_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x3_params.dilation_x = 0;
    inception_c2_3x3_params.dilation_y = 0;
    vx_node inception_c2_3x3_node;
    inception_c2_3x3_node = vxConvolutionLayer(graph, inception_c2_3x3_reduce_relu, inception_c2_3x3_W, NULL, &inception_c2_3x3_params, sizeof(inception_c2_3x3_params ), inception_c2_3x3);
    ERROR_CHECK_OBJECT(inception_c2_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_node));

    // inception_c2_3x3_bn Layer
    vx_size inception_c2_3x3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x3_scale;
    inception_c2_3x3_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_scale);
    vx_size inception_c2_3x3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_3x3_bn_eps = 0.001;
    vx_tensor inception_c2_3x3_bn_W;
    inception_c2_3x3_bn_W = vxCreateTensor(context,1, inception_c2_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_bn_W, dataFolder + "/weights/inception_c2_3x3_bn.f32"));
    vx_size inception_c2_3x3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_3x3_bn_B;
    inception_c2_3x3_bn_B = vxCreateTensor(context,1, inception_c2_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_bn_B, dataFolder + "/bias/inception_c2_3x3_bn.f32"));
    vx_size inception_c2_3x3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_3x3_scale_W;
    inception_c2_3x3_scale_W = vxCreateTensor(context,1, inception_c2_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_scale_W, dataFolder + "/weights/inception_c2_3x3_scale.f32"));
    vx_size inception_c2_3x3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_3x3_scale_B;
    inception_c2_3x3_scale_B = vxCreateTensor(context,1, inception_c2_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x3_scale_B, dataFolder + "/bias/inception_c2_3x3_scale.f32"));
    vx_node inception_c2_3x3_bn_node;
    inception_c2_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x3, inception_c2_3x3_bn_W, inception_c2_3x3_bn_B, inception_c2_3x3_scale_W, inception_c2_3x3_scale_B, inception_c2_3x3_bn_eps, inception_c2_3x3_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_bn_node));

    // inception_c2_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x3_relu Layer
    vx_size inception_c2_3x3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x3_relu;
    inception_c2_3x3_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_relu);
    vx_enum inception_c2_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x3_relu_param_a = 0;
    vx_float32 inception_c2_3x3_relu_param_b = 0;
    vx_node inception_c2_3x3_relu_node;
    inception_c2_3x3_relu_node = vxActivationLayer(graph, inception_c2_3x3_scale, inception_c2_3x3_relu_mode, inception_c2_3x3_relu_param_a, inception_c2_3x3_relu_param_b, inception_c2_3x3_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_relu_node));

    // inception_c2_3x3_inception_c2_3x3_relu_0_split_0 Layer
    vx_size inception_c2_3x3_inception_c2_3x3_relu_0_split_0_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x3_inception_c2_3x3_relu_0_split_0;
    inception_c2_3x3_inception_c2_3x3_relu_0_split_0 = vxCreateVirtualTensor(graph,4, inception_c2_3x3_inception_c2_3x3_relu_0_split_0_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_inception_c2_3x3_relu_0_split_0);
    vx_node inception_c2_3x3_inception_c2_3x3_relu_0_split_0_node;
    inception_c2_3x3_inception_c2_3x3_relu_0_split_0_node = vxCopyNode( graph, (vx_reference)inception_c2_3x3_relu, (vx_reference)inception_c2_3x3_inception_c2_3x3_relu_0_split_0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_inception_c2_3x3_relu_0_split_0_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_inception_c2_3x3_relu_0_split_0_node));

    // inception_c2_3x3_inception_c2_3x3_relu_0_split_1 Layer
    vx_size inception_c2_3x3_inception_c2_3x3_relu_0_split_1_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x3_inception_c2_3x3_relu_0_split_1;
    inception_c2_3x3_inception_c2_3x3_relu_0_split_1 = vxCreateVirtualTensor(graph,4, inception_c2_3x3_inception_c2_3x3_relu_0_split_1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x3_inception_c2_3x3_relu_0_split_1);
    vx_node inception_c2_3x3_inception_c2_3x3_relu_0_split_1_node;
    inception_c2_3x3_inception_c2_3x3_relu_0_split_1_node = vxCopyNode( graph, (vx_reference)inception_c2_3x3_relu, (vx_reference)inception_c2_3x3_inception_c2_3x3_relu_0_split_1);
    ERROR_CHECK_OBJECT(inception_c2_3x3_inception_c2_3x3_relu_0_split_1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x3_inception_c2_3x3_relu_0_split_1_node));

    // inception_c2_1x3_2 Layer
    vx_size inception_c2_1x3_2_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_2;
    inception_c2_1x3_2 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2);
    vx_size inception_c2_1x3_2_W_dims[4] = { 3, 1, 384, 384 };
    vx_tensor inception_c2_1x3_2_W;
    inception_c2_1x3_2_W = vxCreateTensor(context,4, inception_c2_1x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_W, dataFolder + "/weights/inception_c2_1x3_2.f32"));
    vx_nn_convolution_params_t inception_c2_1x3_2_params;
    inception_c2_1x3_2_params.padding_x = 1;
    inception_c2_1x3_2_params.padding_y = 0;
    inception_c2_1x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x3_2_params.dilation_x = 0;
    inception_c2_1x3_2_params.dilation_y = 0;
    vx_node inception_c2_1x3_2_node;
    inception_c2_1x3_2_node = vxConvolutionLayer(graph, inception_c2_3x3_inception_c2_3x3_relu_0_split_0, inception_c2_1x3_2_W, NULL, &inception_c2_1x3_2_params, sizeof(inception_c2_1x3_2_params ), inception_c2_1x3_2);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_2_node));

    // inception_c2_1x3_2_bn Layer
    vx_size inception_c2_1x3_2_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_2_scale;
    inception_c2_1x3_2_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_scale);
    vx_size inception_c2_1x3_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_1x3_2_bn_eps = 0.001;
    vx_tensor inception_c2_1x3_2_bn_W;
    inception_c2_1x3_2_bn_W = vxCreateTensor(context,1, inception_c2_1x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_bn_W, dataFolder + "/weights/inception_c2_1x3_2_bn.f32"));
    vx_size inception_c2_1x3_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_2_bn_B;
    inception_c2_1x3_2_bn_B = vxCreateTensor(context,1, inception_c2_1x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_bn_B, dataFolder + "/bias/inception_c2_1x3_2_bn.f32"));
    vx_size inception_c2_1x3_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_2_scale_W;
    inception_c2_1x3_2_scale_W = vxCreateTensor(context,1, inception_c2_1x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_scale_W, dataFolder + "/weights/inception_c2_1x3_2_scale.f32"));
    vx_size inception_c2_1x3_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x3_2_scale_B;
    inception_c2_1x3_2_scale_B = vxCreateTensor(context,1, inception_c2_1x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_scale_B, dataFolder + "/bias/inception_c2_1x3_2_scale.f32"));
    vx_node inception_c2_1x3_2_bn_node;
    inception_c2_1x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x3_2, inception_c2_1x3_2_bn_W, inception_c2_1x3_2_bn_B, inception_c2_1x3_2_scale_W, inception_c2_1x3_2_scale_B, inception_c2_1x3_2_bn_eps, inception_c2_1x3_2_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_2_bn_node));

    // inception_c2_1x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x3_2_relu Layer
    vx_size inception_c2_1x3_2_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x3_2_relu;
    inception_c2_1x3_2_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_relu);
    vx_enum inception_c2_1x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x3_2_relu_param_a = 0;
    vx_float32 inception_c2_1x3_2_relu_param_b = 0;
    vx_node inception_c2_1x3_2_relu_node;
    inception_c2_1x3_2_relu_node = vxActivationLayer(graph, inception_c2_1x3_2_scale, inception_c2_1x3_2_relu_mode, inception_c2_1x3_2_relu_param_a, inception_c2_1x3_2_relu_param_b, inception_c2_1x3_2_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_2_relu_node));

    // inception_c2_3x1_2 Layer
    vx_size inception_c2_3x1_2_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x1_2;
    inception_c2_3x1_2 = vxCreateVirtualTensor(graph,4, inception_c2_3x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2);
    vx_size inception_c2_3x1_2_W_dims[4] = { 1, 3, 384, 384 };
    vx_tensor inception_c2_3x1_2_W;
    inception_c2_3x1_2_W = vxCreateTensor(context,4, inception_c2_3x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_W, dataFolder + "/weights/inception_c2_3x1_2.f32"));
    vx_nn_convolution_params_t inception_c2_3x1_2_params;
    inception_c2_3x1_2_params.padding_x = 0;
    inception_c2_3x1_2_params.padding_y = 1;
    inception_c2_3x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x1_2_params.dilation_x = 0;
    inception_c2_3x1_2_params.dilation_y = 0;
    vx_node inception_c2_3x1_2_node;
    inception_c2_3x1_2_node = vxConvolutionLayer(graph, inception_c2_3x3_inception_c2_3x3_relu_0_split_1, inception_c2_3x1_2_W, NULL, &inception_c2_3x1_2_params, sizeof(inception_c2_3x1_2_params ), inception_c2_3x1_2);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_2_node));

    // inception_c2_3x1_2_bn Layer
    vx_size inception_c2_3x1_2_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x1_2_scale;
    inception_c2_3x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_scale);
    vx_size inception_c2_3x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_3x1_2_bn_eps = 0.001;
    vx_tensor inception_c2_3x1_2_bn_W;
    inception_c2_3x1_2_bn_W = vxCreateTensor(context,1, inception_c2_3x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_bn_W, dataFolder + "/weights/inception_c2_3x1_2_bn.f32"));
    vx_size inception_c2_3x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_3x1_2_bn_B;
    inception_c2_3x1_2_bn_B = vxCreateTensor(context,1, inception_c2_3x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_bn_B, dataFolder + "/bias/inception_c2_3x1_2_bn.f32"));
    vx_size inception_c2_3x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_3x1_2_scale_W;
    inception_c2_3x1_2_scale_W = vxCreateTensor(context,1, inception_c2_3x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_scale_W, dataFolder + "/weights/inception_c2_3x1_2_scale.f32"));
    vx_size inception_c2_3x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_3x1_2_scale_B;
    inception_c2_3x1_2_scale_B = vxCreateTensor(context,1, inception_c2_3x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_scale_B, dataFolder + "/bias/inception_c2_3x1_2_scale.f32"));
    vx_node inception_c2_3x1_2_bn_node;
    inception_c2_3x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x1_2, inception_c2_3x1_2_bn_W, inception_c2_3x1_2_bn_B, inception_c2_3x1_2_scale_W, inception_c2_3x1_2_scale_B, inception_c2_3x1_2_bn_eps, inception_c2_3x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_2_bn_node));

    // inception_c2_3x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x1_2_relu Layer
    vx_size inception_c2_3x1_2_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_3x1_2_relu;
    inception_c2_3x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_relu);
    vx_enum inception_c2_3x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x1_2_relu_param_a = 0;
    vx_float32 inception_c2_3x1_2_relu_param_b = 0;
    vx_node inception_c2_3x1_2_relu_node;
    inception_c2_3x1_2_relu_node = vxActivationLayer(graph, inception_c2_3x1_2_scale, inception_c2_3x1_2_relu_mode, inception_c2_3x1_2_relu_param_a, inception_c2_3x1_2_relu_param_b, inception_c2_3x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_2_relu_node));

    // inception_c2_pool Layer
    vx_size inception_c2_pool_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c2_pool;
    inception_c2_pool = vxCreateVirtualTensor(graph,4, inception_c2_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_pool);
    vx_enum inception_c2_pool_type = VX_NN_POOLING_MAX;
    vx_size inception_c2_pool_kernel_w = 3;
    vx_size inception_c2_pool_kernel_h = 3;
    vx_size inception_c2_pool_pad_w = 1;
    vx_size inception_c2_pool_pad_h = 1;
    vx_enum inception_c2_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_c2_pool_node;
    inception_c2_pool_node = vxPoolingLayer(graph, inception_c1_concat_inception_c1_concat_0_split_3, inception_c2_pool_type, inception_c2_pool_kernel_w, inception_c2_pool_kernel_h, inception_c2_pool_pad_w, inception_c2_pool_pad_h, inception_c2_pool_roundPolicy, inception_c2_pool );
    ERROR_CHECK_OBJECT(inception_c2_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_pool_node));

    // inception_c2_1x1 Layer
    vx_size inception_c2_1x1_dims[4] = { 8, 8, 192, 1 };
    vx_tensor inception_c2_1x1;
    inception_c2_1x1 = vxCreateVirtualTensor(graph,4, inception_c2_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1);
    vx_size inception_c2_1x1_W_dims[4] = { 1, 1, 2048, 192 };
    vx_tensor inception_c2_1x1_W;
    inception_c2_1x1_W = vxCreateTensor(context,4, inception_c2_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_W, dataFolder + "/weights/inception_c2_1x1.f32"));
    vx_nn_convolution_params_t inception_c2_1x1_params;
    inception_c2_1x1_params.padding_x = 0;
    inception_c2_1x1_params.padding_y = 0;
    inception_c2_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x1_params.dilation_x = 0;
    inception_c2_1x1_params.dilation_y = 0;
    vx_node inception_c2_1x1_node;
    inception_c2_1x1_node = vxConvolutionLayer(graph, inception_c2_pool, inception_c2_1x1_W, NULL, &inception_c2_1x1_params, sizeof(inception_c2_1x1_params ), inception_c2_1x1);
    ERROR_CHECK_OBJECT(inception_c2_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_node));

    // inception_c2_1x1_bn Layer
    vx_size inception_c2_1x1_scale_dims[4] = { 8, 8, 192, 1 };
    vx_tensor inception_c2_1x1_scale;
    inception_c2_1x1_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_scale);
    vx_size inception_c2_1x1_bn_W_dims[1] = { 192 };
    vx_float32 inception_c2_1x1_bn_eps = 0.001;
    vx_tensor inception_c2_1x1_bn_W;
    inception_c2_1x1_bn_W = vxCreateTensor(context,1, inception_c2_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_bn_W, dataFolder + "/weights/inception_c2_1x1_bn.f32"));
    vx_size inception_c2_1x1_bn_B_dims[1] = { 192 };
    vx_tensor inception_c2_1x1_bn_B;
    inception_c2_1x1_bn_B = vxCreateTensor(context,1, inception_c2_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_bn_B, dataFolder + "/bias/inception_c2_1x1_bn.f32"));
    vx_size inception_c2_1x1_scale_W_dims[1] = { 192 };
    vx_tensor inception_c2_1x1_scale_W;
    inception_c2_1x1_scale_W = vxCreateTensor(context,1, inception_c2_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_scale_W, dataFolder + "/weights/inception_c2_1x1_scale.f32"));
    vx_size inception_c2_1x1_scale_B_dims[1] = { 192 };
    vx_tensor inception_c2_1x1_scale_B;
    inception_c2_1x1_scale_B = vxCreateTensor(context,1, inception_c2_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_scale_B, dataFolder + "/bias/inception_c2_1x1_scale.f32"));
    vx_node inception_c2_1x1_bn_node;
    inception_c2_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x1, inception_c2_1x1_bn_W, inception_c2_1x1_bn_B, inception_c2_1x1_scale_W, inception_c2_1x1_scale_B, inception_c2_1x1_bn_eps, inception_c2_1x1_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_bn_node));

    // inception_c2_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x1_relu Layer
    vx_size inception_c2_1x1_relu_dims[4] = { 8, 8, 192, 1 };
    vx_tensor inception_c2_1x1_relu;
    inception_c2_1x1_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_relu);
    vx_enum inception_c2_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x1_relu_param_a = 0;
    vx_float32 inception_c2_1x1_relu_param_b = 0;
    vx_node inception_c2_1x1_relu_node;
    inception_c2_1x1_relu_node = vxActivationLayer(graph, inception_c2_1x1_scale, inception_c2_1x1_relu_mode, inception_c2_1x1_relu_param_a, inception_c2_1x1_relu_param_b, inception_c2_1x1_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_relu_node));

    // inception_c2_concat Layer
    vx_size inception_c2_concat_dims[4] = { 8, 8, 2048, 1 };
    vx_tensor inception_c2_concat;
    inception_c2_concat = vxCreateVirtualTensor(graph,4, inception_c2_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_concat);
    vx_node inception_c2_concat_node;
    inception_c2_concat_node = vxConcatLayer(graph, inception_c2_concat, inception_c2_1x1_2_relu, inception_c2_1x3_relu, inception_c2_3x1_relu, inception_c2_1x3_2_relu, inception_c2_3x1_2_relu, inception_c2_1x1_relu, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_c2_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_concat_node));

    // pool_8x8_s1 Layer
    vx_size pool_8x8_s1_dims[4] = { 1, 1, 2048, 1 };
    vx_tensor pool_8x8_s1;
    pool_8x8_s1 = vxCreateVirtualTensor(graph,4, pool_8x8_s1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool_8x8_s1);
    vx_enum pool_8x8_s1_type = VX_NN_POOLING_AVG;
    vx_size pool_8x8_s1_kernel_w = 8;
    vx_size pool_8x8_s1_kernel_h = 8;
    vx_size pool_8x8_s1_pad_w = 0;
    vx_size pool_8x8_s1_pad_h = 0;
    vx_enum pool_8x8_s1_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool_8x8_s1_node;
    pool_8x8_s1_node = vxPoolingLayer(graph, inception_c2_concat, pool_8x8_s1_type, pool_8x8_s1_kernel_w, pool_8x8_s1_kernel_h, pool_8x8_s1_pad_w, pool_8x8_s1_pad_h, pool_8x8_s1_roundPolicy, pool_8x8_s1 );
    ERROR_CHECK_OBJECT(pool_8x8_s1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool_8x8_s1_node));

    // pool_8x8_s1_drop Layer
    vx_size pool_8x8_s1_drop_dims[4] = { 1, 1, 2048, 1 };
    vx_tensor pool_8x8_s1_drop;
    pool_8x8_s1_drop = vxCreateVirtualTensor(graph,4, pool_8x8_s1_drop_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool_8x8_s1_drop);
    vx_node pool_8x8_s1_drop_node;
    pool_8x8_s1_drop_node = vxCopyNode( graph, (vx_reference)pool_8x8_s1, (vx_reference)pool_8x8_s1_drop);
    ERROR_CHECK_OBJECT(pool_8x8_s1_drop_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool_8x8_s1_drop_node));

    // classifier Layer
    vx_size classifier_dims[4] = { 1, 1, 1000, 1 };
    vx_tensor classifier;
    classifier = vxCreateVirtualTensor(graph,4, classifier_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(classifier);
    vx_size classifier_W_dims[4] = { 1, 1, 2048, 1000 };
    vx_tensor classifier_W;
    classifier_W= vxCreateTensor(context,4,classifier_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(classifier_W); 
    ERROR_CHECK_STATUS(copyTensor(classifier_W, dataFolder + "/weights/classifier.f32"));
    vx_size classifier_B_dims[1] = { 1000 };
    vx_tensor classifier_B;
    classifier_B= vxCreateTensor(context,1,classifier_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(classifier_B); 
    ERROR_CHECK_STATUS(copyTensor(classifier_B, dataFolder + "/bias/classifier.f32"));
    vx_enum classifier_convertPolicy = VX_CONVERT_POLICY_SATURATE;
    vx_enum classifier_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node classifier_node;
    classifier_node = vxFullyConnectedLayer( graph, pool_8x8_s1_drop, classifier_W, classifier_B, classifier_convertPolicy, classifier_roundPolicy, classifier);
    ERROR_CHECK_OBJECT(classifier_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&classifier_node));

    // prob Layer
    vx_node prob_node;
    prob_node = vxSoftmaxLayer(graph, classifier, prob);
    ERROR_CHECK_OBJECT(prob_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&prob_node));

    ////
    // release intermediate objects
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool1_3x3_s2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_relu_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_relu_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool2_3x3_s2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool2_3x3_s2_pool2_3x3_s2_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool2_3x3_s2_pool2_3x3_s2_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool2_3x3_s2_pool2_3x3_s2_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool2_3x3_s2_pool2_3x3_s2_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_5x5_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_proj_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_output));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_output_inception_a1_output_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_output_inception_a1_output_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_output_inception_a1_output_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_output_inception_a1_output_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_5x5_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_proj_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_output));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_output_inception_a2_output_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_output_inception_a2_output_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_output_inception_a2_output_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_output_inception_a2_output_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_5x5_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_proj_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_output));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_output_inception_a3_output_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_output_inception_a3_output_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_output_inception_a3_output_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_concat_reduction_a_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_concat_reduction_a_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_concat_reduction_a_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_concat_reduction_a_concat_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_concat_inception_b1_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_concat_inception_b1_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_concat_inception_b1_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_concat_inception_b1_concat_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_concat_inception_b2_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_concat_inception_b2_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_concat_inception_b2_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_concat_inception_b2_concat_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_concat_inception_b3_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_concat_inception_b3_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_concat_inception_b3_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_concat_inception_b3_concat_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_concat_inception_b4_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_concat_inception_b4_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_concat_inception_b4_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_concat_reduction_b_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_concat_reduction_b_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_concat_reduction_b_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_concat_reduction_b_concat_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_reduce_inception_c1_1x3_reduce_relu_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_inception_c1_3x3_relu_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x3_inception_c1_3x3_relu_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_concat_inception_c1_concat_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_concat_inception_c1_concat_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_concat_inception_c1_concat_0_split_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_concat_inception_c1_concat_0_split_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_reduce_inception_c2_1x3_reduce_relu_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_inception_c2_3x3_relu_0_split_0));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x3_inception_c2_3x3_relu_0_split_1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool_8x8_s1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool_8x8_s1_drop));
    ERROR_CHECK_STATUS(vxReleaseTensor(&classifier));
    ERROR_CHECK_STATUS(vxReleaseTensor(&classifier_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&classifier_B));

    ////
    // verify the built graph
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    return graph;
}
