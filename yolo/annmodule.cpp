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
    dimInput[0] = 416;
    dimInput[1] = 416;
    dimInput[2] = 3;
    dimInput[3] = 1;
    dimOutput[0] = 12;
    dimOutput[1] = 12;
    dimOutput[2] = 425;
    dimOutput[3] = 1;
}

VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor data, vx_tensor conv9, const char * dataFolder_)
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
    // conv1 Layer
    vx_size conv1_dims[4] = { 416, 416, 16, 1 };
    vx_tensor conv1;
    conv1 = vxCreateVirtualTensor(graph,4, conv1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1);
    vx_size conv1_W_dims[4] = { 3, 3, 3, 16 };
    vx_tensor conv1_W;
    conv1_W = vxCreateTensor(context,4, conv1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_W, dataFolder + "/weights/conv1.f32"));
    vx_nn_convolution_params_t conv1_params;
    conv1_params.padding_x = 1;
    conv1_params.padding_y = 1;
    conv1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv1_params.dilation_x = 0;
    conv1_params.dilation_y = 0;
    vx_node conv1_node;
    conv1_node = vxConvolutionLayer(graph, data, conv1_W, NULL, &conv1_params, sizeof(conv1_params ), conv1);
    ERROR_CHECK_OBJECT(conv1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_node));

    // conv1_bn Layer
    vx_size conv1_scale_dims[4] = { 416, 416, 16, 1 };
    vx_tensor conv1_scale;
    conv1_scale = vxCreateVirtualTensor(graph,4, conv1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_scale);
    vx_size conv1_bn_W_dims[1] = { 16 };
    vx_float32 conv1_bn_eps = 0.001;
    vx_tensor conv1_bn_W;
    conv1_bn_W = vxCreateTensor(context,1, conv1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_bn_W, dataFolder + "/weights/conv1_bn.f32"));
    vx_size conv1_bn_B_dims[1] = { 16 };
    vx_tensor conv1_bn_B;
    conv1_bn_B = vxCreateTensor(context,1, conv1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv1_bn_B, dataFolder + "/bias/conv1_bn.f32"));
    vx_size conv1_scale_W_dims[1] = { 16 };
    vx_tensor conv1_scale_W;
    conv1_scale_W = vxCreateTensor(context,1, conv1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_scale_W, dataFolder + "/weights/conv1_scale.f32"));
    vx_size conv1_scale_B_dims[1] = { 16 };
    vx_tensor conv1_scale_B;
    conv1_scale_B = vxCreateTensor(context,1, conv1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv1_scale_B, dataFolder + "/bias/conv1_scale.f32"));
    vx_node conv1_bn_node;
    conv1_bn_node = vxBatchNormalizationLayer(graph, conv1, conv1_bn_W, conv1_bn_B, conv1_scale_W, conv1_scale_B, conv1_bn_eps, conv1_scale);
    ERROR_CHECK_OBJECT(conv1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_bn_node));

    // conv1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu1 Layer
    vx_size relu1_dims[4] = { 416, 416, 16, 1 };
    vx_tensor relu1;
    relu1 = vxCreateVirtualTensor(graph,4, relu1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu1);
    vx_enum relu1_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu1_param_a = 0.1;
    vx_float32 relu1_param_b = 0;
    vx_node relu1_node;
    relu1_node = vxActivationLayer(graph, conv1_scale, relu1_mode, relu1_param_a, relu1_param_b, relu1);
    ERROR_CHECK_OBJECT(relu1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu1_node));

    // pool1 Layer
    vx_size pool1_dims[4] = { 208, 208, 16, 1 };
    vx_tensor pool1;
    pool1 = vxCreateVirtualTensor(graph,4, pool1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool1);
    vx_enum pool1_type = VX_NN_POOLING_MAX;
    vx_size pool1_kernel_w = 2;
    vx_size pool1_kernel_h = 2;
    vx_size pool1_pad_w = 0;
    vx_size pool1_pad_h = 0;
    vx_enum pool1_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool1_node;
    pool1_node = vxPoolingLayer(graph, relu1, pool1_type, pool1_kernel_w, pool1_kernel_h, pool1_pad_w, pool1_pad_h, pool1_roundPolicy, pool1 );
    ERROR_CHECK_OBJECT(pool1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool1_node));

    // conv2 Layer
    vx_size conv2_dims[4] = { 208, 208, 32, 1 };
    vx_tensor conv2;
    conv2 = vxCreateVirtualTensor(graph,4, conv2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2);
    vx_size conv2_W_dims[4] = { 3, 3, 16, 32 };
    vx_tensor conv2_W;
    conv2_W = vxCreateTensor(context,4, conv2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_W, dataFolder + "/weights/conv2.f32"));
    vx_nn_convolution_params_t conv2_params;
    conv2_params.padding_x = 1;
    conv2_params.padding_y = 1;
    conv2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv2_params.dilation_x = 0;
    conv2_params.dilation_y = 0;
    vx_node conv2_node;
    conv2_node = vxConvolutionLayer(graph, pool1, conv2_W, NULL, &conv2_params, sizeof(conv2_params ), conv2);
    ERROR_CHECK_OBJECT(conv2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_node));

    // conv2_bn Layer
    vx_size conv2_scale_dims[4] = { 208, 208, 32, 1 };
    vx_tensor conv2_scale;
    conv2_scale = vxCreateVirtualTensor(graph,4, conv2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_scale);
    vx_size conv2_bn_W_dims[1] = { 32 };
    vx_float32 conv2_bn_eps = 0.001;
    vx_tensor conv2_bn_W;
    conv2_bn_W = vxCreateTensor(context,1, conv2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_bn_W, dataFolder + "/weights/conv2_bn.f32"));
    vx_size conv2_bn_B_dims[1] = { 32 };
    vx_tensor conv2_bn_B;
    conv2_bn_B = vxCreateTensor(context,1, conv2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv2_bn_B, dataFolder + "/bias/conv2_bn.f32"));
    vx_size conv2_scale_W_dims[1] = { 32 };
    vx_tensor conv2_scale_W;
    conv2_scale_W = vxCreateTensor(context,1, conv2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_scale_W, dataFolder + "/weights/conv2_scale.f32"));
    vx_size conv2_scale_B_dims[1] = { 32 };
    vx_tensor conv2_scale_B;
    conv2_scale_B = vxCreateTensor(context,1, conv2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv2_scale_B, dataFolder + "/bias/conv2_scale.f32"));
    vx_node conv2_bn_node;
    conv2_bn_node = vxBatchNormalizationLayer(graph, conv2, conv2_bn_W, conv2_bn_B, conv2_scale_W, conv2_scale_B, conv2_bn_eps, conv2_scale);
    ERROR_CHECK_OBJECT(conv2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_bn_node));

    // conv2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu2 Layer
    vx_size relu2_dims[4] = { 208, 208, 32, 1 };
    vx_tensor relu2;
    relu2 = vxCreateVirtualTensor(graph,4, relu2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu2);
    vx_enum relu2_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu2_param_a = 0.1;
    vx_float32 relu2_param_b = 0;
    vx_node relu2_node;
    relu2_node = vxActivationLayer(graph, conv2_scale, relu2_mode, relu2_param_a, relu2_param_b, relu2);
    ERROR_CHECK_OBJECT(relu2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu2_node));

    // pool2 Layer
    vx_size pool2_dims[4] = { 104, 104, 32, 1 };
    vx_tensor pool2;
    pool2 = vxCreateVirtualTensor(graph,4, pool2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool2);
    vx_enum pool2_type = VX_NN_POOLING_MAX;
    vx_size pool2_kernel_w = 2;
    vx_size pool2_kernel_h = 2;
    vx_size pool2_pad_w = 0;
    vx_size pool2_pad_h = 0;
    vx_enum pool2_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool2_node;
    pool2_node = vxPoolingLayer(graph, relu2, pool2_type, pool2_kernel_w, pool2_kernel_h, pool2_pad_w, pool2_pad_h, pool2_roundPolicy, pool2 );
    ERROR_CHECK_OBJECT(pool2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool2_node));

    // conv3 Layer
    vx_size conv3_dims[4] = { 104, 104, 64, 1 };
    vx_tensor conv3;
    conv3 = vxCreateVirtualTensor(graph,4, conv3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3);
    vx_size conv3_W_dims[4] = { 3, 3, 32, 64 };
    vx_tensor conv3_W;
    conv3_W = vxCreateTensor(context,4, conv3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_W, dataFolder + "/weights/conv3.f32"));
    vx_nn_convolution_params_t conv3_params;
    conv3_params.padding_x = 1;
    conv3_params.padding_y = 1;
    conv3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv3_params.dilation_x = 0;
    conv3_params.dilation_y = 0;
    vx_node conv3_node;
    conv3_node = vxConvolutionLayer(graph, pool2, conv3_W, NULL, &conv3_params, sizeof(conv3_params ), conv3);
    ERROR_CHECK_OBJECT(conv3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_node));

    // conv3_bn Layer
    vx_size conv3_scale_dims[4] = { 104, 104, 64, 1 };
    vx_tensor conv3_scale;
    conv3_scale = vxCreateVirtualTensor(graph,4, conv3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_scale);
    vx_size conv3_bn_W_dims[1] = { 64 };
    vx_float32 conv3_bn_eps = 0.001;
    vx_tensor conv3_bn_W;
    conv3_bn_W = vxCreateTensor(context,1, conv3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_bn_W, dataFolder + "/weights/conv3_bn.f32"));
    vx_size conv3_bn_B_dims[1] = { 64 };
    vx_tensor conv3_bn_B;
    conv3_bn_B = vxCreateTensor(context,1, conv3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv3_bn_B, dataFolder + "/bias/conv3_bn.f32"));
    vx_size conv3_scale_W_dims[1] = { 64 };
    vx_tensor conv3_scale_W;
    conv3_scale_W = vxCreateTensor(context,1, conv3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_scale_W, dataFolder + "/weights/conv3_scale.f32"));
    vx_size conv3_scale_B_dims[1] = { 64 };
    vx_tensor conv3_scale_B;
    conv3_scale_B = vxCreateTensor(context,1, conv3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv3_scale_B, dataFolder + "/bias/conv3_scale.f32"));
    vx_node conv3_bn_node;
    conv3_bn_node = vxBatchNormalizationLayer(graph, conv3, conv3_bn_W, conv3_bn_B, conv3_scale_W, conv3_scale_B, conv3_bn_eps, conv3_scale);
    ERROR_CHECK_OBJECT(conv3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_bn_node));

    // conv3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu3 Layer
    vx_size relu3_dims[4] = { 104, 104, 64, 1 };
    vx_tensor relu3;
    relu3 = vxCreateVirtualTensor(graph,4, relu3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu3);
    vx_enum relu3_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu3_param_a = 0.1;
    vx_float32 relu3_param_b = 0;
    vx_node relu3_node;
    relu3_node = vxActivationLayer(graph, conv3_scale, relu3_mode, relu3_param_a, relu3_param_b, relu3);
    ERROR_CHECK_OBJECT(relu3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu3_node));

    // pool3 Layer
    vx_size pool3_dims[4] = { 52, 52, 64, 1 };
    vx_tensor pool3;
    pool3 = vxCreateVirtualTensor(graph,4, pool3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool3);
    vx_enum pool3_type = VX_NN_POOLING_MAX;
    vx_size pool3_kernel_w = 2;
    vx_size pool3_kernel_h = 2;
    vx_size pool3_pad_w = 0;
    vx_size pool3_pad_h = 0;
    vx_enum pool3_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool3_node;
    pool3_node = vxPoolingLayer(graph, relu3, pool3_type, pool3_kernel_w, pool3_kernel_h, pool3_pad_w, pool3_pad_h, pool3_roundPolicy, pool3 );
    ERROR_CHECK_OBJECT(pool3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool3_node));

    // conv4 Layer
    vx_size conv4_dims[4] = { 52, 52, 128, 1 };
    vx_tensor conv4;
    conv4 = vxCreateVirtualTensor(graph,4, conv4_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4);
    vx_size conv4_W_dims[4] = { 3, 3, 64, 128 };
    vx_tensor conv4_W;
    conv4_W = vxCreateTensor(context,4, conv4_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_W, dataFolder + "/weights/conv4.f32"));
    vx_nn_convolution_params_t conv4_params;
    conv4_params.padding_x = 1;
    conv4_params.padding_y = 1;
    conv4_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv4_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv4_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv4_params.dilation_x = 0;
    conv4_params.dilation_y = 0;
    vx_node conv4_node;
    conv4_node = vxConvolutionLayer(graph, pool3, conv4_W, NULL, &conv4_params, sizeof(conv4_params ), conv4);
    ERROR_CHECK_OBJECT(conv4_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_node));

    // conv4_bn Layer
    vx_size conv4_scale_dims[4] = { 52, 52, 128, 1 };
    vx_tensor conv4_scale;
    conv4_scale = vxCreateVirtualTensor(graph,4, conv4_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv4_scale);
    vx_size conv4_bn_W_dims[1] = { 128 };
    vx_float32 conv4_bn_eps = 0.001;
    vx_tensor conv4_bn_W;
    conv4_bn_W = vxCreateTensor(context,1, conv4_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_bn_W, dataFolder + "/weights/conv4_bn.f32"));
    vx_size conv4_bn_B_dims[1] = { 128 };
    vx_tensor conv4_bn_B;
    conv4_bn_B = vxCreateTensor(context,1, conv4_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv4_bn_B, dataFolder + "/bias/conv4_bn.f32"));
    vx_size conv4_scale_W_dims[1] = { 128 };
    vx_tensor conv4_scale_W;
    conv4_scale_W = vxCreateTensor(context,1, conv4_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv4_scale_W, dataFolder + "/weights/conv4_scale.f32"));
    vx_size conv4_scale_B_dims[1] = { 128 };
    vx_tensor conv4_scale_B;
    conv4_scale_B = vxCreateTensor(context,1, conv4_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv4_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv4_scale_B, dataFolder + "/bias/conv4_scale.f32"));
    vx_node conv4_bn_node;
    conv4_bn_node = vxBatchNormalizationLayer(graph, conv4, conv4_bn_W, conv4_bn_B, conv4_scale_W, conv4_scale_B, conv4_bn_eps, conv4_scale);
    ERROR_CHECK_OBJECT(conv4_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv4_bn_node));

    // conv4_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu4 Layer
    vx_size relu4_dims[4] = { 52, 52, 128, 1 };
    vx_tensor relu4;
    relu4 = vxCreateVirtualTensor(graph,4, relu4_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu4);
    vx_enum relu4_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu4_param_a = 0.1;
    vx_float32 relu4_param_b = 0;
    vx_node relu4_node;
    relu4_node = vxActivationLayer(graph, conv4_scale, relu4_mode, relu4_param_a, relu4_param_b, relu4);
    ERROR_CHECK_OBJECT(relu4_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu4_node));

    // pool4 Layer
    vx_size pool4_dims[4] = { 26, 26, 128, 1 };
    vx_tensor pool4;
    pool4 = vxCreateVirtualTensor(graph,4, pool4_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool4);
    vx_enum pool4_type = VX_NN_POOLING_MAX;
    vx_size pool4_kernel_w = 2;
    vx_size pool4_kernel_h = 2;
    vx_size pool4_pad_w = 0;
    vx_size pool4_pad_h = 0;
    vx_enum pool4_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool4_node;
    pool4_node = vxPoolingLayer(graph, relu4, pool4_type, pool4_kernel_w, pool4_kernel_h, pool4_pad_w, pool4_pad_h, pool4_roundPolicy, pool4 );
    ERROR_CHECK_OBJECT(pool4_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool4_node));

    // conv5 Layer
    vx_size conv5_dims[4] = { 26, 26, 256, 1 };
    vx_tensor conv5;
    conv5 = vxCreateVirtualTensor(graph,4, conv5_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv5);
    vx_size conv5_W_dims[4] = { 3, 3, 128, 256 };
    vx_tensor conv5_W;
    conv5_W = vxCreateTensor(context,4, conv5_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv5_W); 
    ERROR_CHECK_STATUS(copyTensor(conv5_W, dataFolder + "/weights/conv5.f32"));
    vx_nn_convolution_params_t conv5_params;
    conv5_params.padding_x = 1;
    conv5_params.padding_y = 1;
    conv5_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv5_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv5_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv5_params.dilation_x = 0;
    conv5_params.dilation_y = 0;
    vx_node conv5_node;
    conv5_node = vxConvolutionLayer(graph, pool4, conv5_W, NULL, &conv5_params, sizeof(conv5_params ), conv5);
    ERROR_CHECK_OBJECT(conv5_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv5_node));

    // conv5_bn Layer
    vx_size conv5_scale_dims[4] = { 26, 26, 256, 1 };
    vx_tensor conv5_scale;
    conv5_scale = vxCreateVirtualTensor(graph,4, conv5_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv5_scale);
    vx_size conv5_bn_W_dims[1] = { 256 };
    vx_float32 conv5_bn_eps = 0.001;
    vx_tensor conv5_bn_W;
    conv5_bn_W = vxCreateTensor(context,1, conv5_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv5_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv5_bn_W, dataFolder + "/weights/conv5_bn.f32"));
    vx_size conv5_bn_B_dims[1] = { 256 };
    vx_tensor conv5_bn_B;
    conv5_bn_B = vxCreateTensor(context,1, conv5_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv5_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv5_bn_B, dataFolder + "/bias/conv5_bn.f32"));
    vx_size conv5_scale_W_dims[1] = { 256 };
    vx_tensor conv5_scale_W;
    conv5_scale_W = vxCreateTensor(context,1, conv5_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv5_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv5_scale_W, dataFolder + "/weights/conv5_scale.f32"));
    vx_size conv5_scale_B_dims[1] = { 256 };
    vx_tensor conv5_scale_B;
    conv5_scale_B = vxCreateTensor(context,1, conv5_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv5_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv5_scale_B, dataFolder + "/bias/conv5_scale.f32"));
    vx_node conv5_bn_node;
    conv5_bn_node = vxBatchNormalizationLayer(graph, conv5, conv5_bn_W, conv5_bn_B, conv5_scale_W, conv5_scale_B, conv5_bn_eps, conv5_scale);
    ERROR_CHECK_OBJECT(conv5_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv5_bn_node));

    // conv5_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu5 Layer
    vx_size relu5_dims[4] = { 26, 26, 256, 1 };
    vx_tensor relu5;
    relu5 = vxCreateVirtualTensor(graph,4, relu5_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu5);
    vx_enum relu5_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu5_param_a = 0.1;
    vx_float32 relu5_param_b = 0;
    vx_node relu5_node;
    relu5_node = vxActivationLayer(graph, conv5_scale, relu5_mode, relu5_param_a, relu5_param_b, relu5);
    ERROR_CHECK_OBJECT(relu5_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu5_node));

    // pool5 Layer
    vx_size pool5_dims[4] = { 13, 13, 256, 1 };
    vx_tensor pool5;
    pool5 = vxCreateVirtualTensor(graph,4, pool5_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool5);
    vx_enum pool5_type = VX_NN_POOLING_MAX;
    vx_size pool5_kernel_w = 2;
    vx_size pool5_kernel_h = 2;
    vx_size pool5_pad_w = 0;
    vx_size pool5_pad_h = 0;
    vx_enum pool5_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool5_node;
    pool5_node = vxPoolingLayer(graph, relu5, pool5_type, pool5_kernel_w, pool5_kernel_h, pool5_pad_w, pool5_pad_h, pool5_roundPolicy, pool5 );
    ERROR_CHECK_OBJECT(pool5_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool5_node));

    // conv6 Layer
    vx_size conv6_dims[4] = { 13, 13, 512, 1 };
    vx_tensor conv6;
    conv6 = vxCreateVirtualTensor(graph,4, conv6_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv6);
    vx_size conv6_W_dims[4] = { 3, 3, 256, 512 };
    vx_tensor conv6_W;
    conv6_W = vxCreateTensor(context,4, conv6_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv6_W); 
    ERROR_CHECK_STATUS(copyTensor(conv6_W, dataFolder + "/weights/conv6.f32"));
    vx_nn_convolution_params_t conv6_params;
    conv6_params.padding_x = 1;
    conv6_params.padding_y = 1;
    conv6_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv6_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv6_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv6_params.dilation_x = 0;
    conv6_params.dilation_y = 0;
    vx_node conv6_node;
    conv6_node = vxConvolutionLayer(graph, pool5, conv6_W, NULL, &conv6_params, sizeof(conv6_params ), conv6);
    ERROR_CHECK_OBJECT(conv6_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv6_node));

    // conv6_bn Layer
    vx_size conv6_scale_dims[4] = { 13, 13, 512, 1 };
    vx_tensor conv6_scale;
    conv6_scale = vxCreateVirtualTensor(graph,4, conv6_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv6_scale);
    vx_size conv6_bn_W_dims[1] = { 512 };
    vx_float32 conv6_bn_eps = 0.001;
    vx_tensor conv6_bn_W;
    conv6_bn_W = vxCreateTensor(context,1, conv6_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv6_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv6_bn_W, dataFolder + "/weights/conv6_bn.f32"));
    vx_size conv6_bn_B_dims[1] = { 512 };
    vx_tensor conv6_bn_B;
    conv6_bn_B = vxCreateTensor(context,1, conv6_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv6_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv6_bn_B, dataFolder + "/bias/conv6_bn.f32"));
    vx_size conv6_scale_W_dims[1] = { 512 };
    vx_tensor conv6_scale_W;
    conv6_scale_W = vxCreateTensor(context,1, conv6_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv6_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv6_scale_W, dataFolder + "/weights/conv6_scale.f32"));
    vx_size conv6_scale_B_dims[1] = { 512 };
    vx_tensor conv6_scale_B;
    conv6_scale_B = vxCreateTensor(context,1, conv6_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv6_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv6_scale_B, dataFolder + "/bias/conv6_scale.f32"));
    vx_node conv6_bn_node;
    conv6_bn_node = vxBatchNormalizationLayer(graph, conv6, conv6_bn_W, conv6_bn_B, conv6_scale_W, conv6_scale_B, conv6_bn_eps, conv6_scale);
    ERROR_CHECK_OBJECT(conv6_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv6_bn_node));

    // conv6_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu6 Layer
    vx_size relu6_dims[4] = { 13, 13, 512, 1 };
    vx_tensor relu6;
    relu6 = vxCreateVirtualTensor(graph,4, relu6_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu6);
    vx_enum relu6_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu6_param_a = 0.1;
    vx_float32 relu6_param_b = 0;
    vx_node relu6_node;
    relu6_node = vxActivationLayer(graph, conv6_scale, relu6_mode, relu6_param_a, relu6_param_b, relu6);
    ERROR_CHECK_OBJECT(relu6_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu6_node));

    // pool6 Layer
    vx_size pool6_dims[4] = { 12, 12, 512, 1 };
    vx_tensor pool6;
    pool6 = vxCreateVirtualTensor(graph,4, pool6_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool6);
    vx_enum pool6_type = VX_NN_POOLING_MAX;
    vx_size pool6_kernel_w = 2;
    vx_size pool6_kernel_h = 2;
    vx_size pool6_pad_w = 0;
    vx_size pool6_pad_h = 0;
    vx_enum pool6_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool6_node;
    pool6_node = vxPoolingLayer(graph, relu6, pool6_type, pool6_kernel_w, pool6_kernel_h, pool6_pad_w, pool6_pad_h, pool6_roundPolicy, pool6 );
    ERROR_CHECK_OBJECT(pool6_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool6_node));

    // conv7 Layer
    vx_size conv7_dims[4] = { 12, 12, 1024, 1 };
    vx_tensor conv7;
    conv7 = vxCreateVirtualTensor(graph,4, conv7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv7);
    vx_size conv7_W_dims[4] = { 3, 3, 512, 1024 };
    vx_tensor conv7_W;
    conv7_W = vxCreateTensor(context,4, conv7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv7_W); 
    ERROR_CHECK_STATUS(copyTensor(conv7_W, dataFolder + "/weights/conv7.f32"));
    vx_nn_convolution_params_t conv7_params;
    conv7_params.padding_x = 1;
    conv7_params.padding_y = 1;
    conv7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv7_params.dilation_x = 0;
    conv7_params.dilation_y = 0;
    vx_node conv7_node;
    conv7_node = vxConvolutionLayer(graph, pool6, conv7_W, NULL, &conv7_params, sizeof(conv7_params ), conv7);
    ERROR_CHECK_OBJECT(conv7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv7_node));

    // conv7_bn Layer
    vx_size conv7_scale_dims[4] = { 12, 12, 1024, 1 };
    vx_tensor conv7_scale;
    conv7_scale = vxCreateVirtualTensor(graph,4, conv7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv7_scale);
    vx_size conv7_bn_W_dims[1] = { 1024 };
    vx_float32 conv7_bn_eps = 0.001;
    vx_tensor conv7_bn_W;
    conv7_bn_W = vxCreateTensor(context,1, conv7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv7_bn_W, dataFolder + "/weights/conv7_bn.f32"));
    vx_size conv7_bn_B_dims[1] = { 1024 };
    vx_tensor conv7_bn_B;
    conv7_bn_B = vxCreateTensor(context,1, conv7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv7_bn_B, dataFolder + "/bias/conv7_bn.f32"));
    vx_size conv7_scale_W_dims[1] = { 1024 };
    vx_tensor conv7_scale_W;
    conv7_scale_W = vxCreateTensor(context,1, conv7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv7_scale_W, dataFolder + "/weights/conv7_scale.f32"));
    vx_size conv7_scale_B_dims[1] = { 1024 };
    vx_tensor conv7_scale_B;
    conv7_scale_B = vxCreateTensor(context,1, conv7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv7_scale_B, dataFolder + "/bias/conv7_scale.f32"));
    vx_node conv7_bn_node;
    conv7_bn_node = vxBatchNormalizationLayer(graph, conv7, conv7_bn_W, conv7_bn_B, conv7_scale_W, conv7_scale_B, conv7_bn_eps, conv7_scale);
    ERROR_CHECK_OBJECT(conv7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv7_bn_node));

    // conv7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu7 Layer
    vx_size relu7_dims[4] = { 12, 12, 1024, 1 };
    vx_tensor relu7;
    relu7 = vxCreateVirtualTensor(graph,4, relu7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu7);
    vx_enum relu7_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu7_param_a = 0.1;
    vx_float32 relu7_param_b = 0;
    vx_node relu7_node;
    relu7_node = vxActivationLayer(graph, conv7_scale, relu7_mode, relu7_param_a, relu7_param_b, relu7);
    ERROR_CHECK_OBJECT(relu7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu7_node));

    // conv8 Layer
    vx_size conv8_dims[4] = { 12, 12, 1024, 1 };
    vx_tensor conv8;
    conv8 = vxCreateVirtualTensor(graph,4, conv8_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv8);
    vx_size conv8_W_dims[4] = { 3, 3, 1024, 1024 };
    vx_tensor conv8_W;
    conv8_W = vxCreateTensor(context,4, conv8_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv8_W); 
    ERROR_CHECK_STATUS(copyTensor(conv8_W, dataFolder + "/weights/conv8.f32"));
    vx_nn_convolution_params_t conv8_params;
    conv8_params.padding_x = 1;
    conv8_params.padding_y = 1;
    conv8_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv8_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv8_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv8_params.dilation_x = 0;
    conv8_params.dilation_y = 0;
    vx_node conv8_node;
    conv8_node = vxConvolutionLayer(graph, relu7, conv8_W, NULL, &conv8_params, sizeof(conv8_params ), conv8);
    ERROR_CHECK_OBJECT(conv8_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv8_node));

    // conv8_bn Layer
    vx_size conv8_scale_dims[4] = { 12, 12, 1024, 1 };
    vx_tensor conv8_scale;
    conv8_scale = vxCreateVirtualTensor(graph,4, conv8_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv8_scale);
    vx_size conv8_bn_W_dims[1] = { 1024 };
    vx_float32 conv8_bn_eps = 0.001;
    vx_tensor conv8_bn_W;
    conv8_bn_W = vxCreateTensor(context,1, conv8_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv8_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv8_bn_W, dataFolder + "/weights/conv8_bn.f32"));
    vx_size conv8_bn_B_dims[1] = { 1024 };
    vx_tensor conv8_bn_B;
    conv8_bn_B = vxCreateTensor(context,1, conv8_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv8_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv8_bn_B, dataFolder + "/bias/conv8_bn.f32"));
    vx_size conv8_scale_W_dims[1] = { 1024 };
    vx_tensor conv8_scale_W;
    conv8_scale_W = vxCreateTensor(context,1, conv8_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv8_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv8_scale_W, dataFolder + "/weights/conv8_scale.f32"));
    vx_size conv8_scale_B_dims[1] = { 1024 };
    vx_tensor conv8_scale_B;
    conv8_scale_B = vxCreateTensor(context,1, conv8_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv8_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv8_scale_B, dataFolder + "/bias/conv8_scale.f32"));
    vx_node conv8_bn_node;
    conv8_bn_node = vxBatchNormalizationLayer(graph, conv8, conv8_bn_W, conv8_bn_B, conv8_scale_W, conv8_scale_B, conv8_bn_eps, conv8_scale);
    ERROR_CHECK_OBJECT(conv8_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv8_bn_node));

    // conv8_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // relu8 Layer
    vx_size relu8_dims[4] = { 12, 12, 1024, 1 };
    vx_tensor relu8;
    relu8 = vxCreateVirtualTensor(graph,4, relu8_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(relu8);
    vx_enum relu8_mode = VX_NN_ACTIVATION_LEAKY_RELU ; 
    vx_float32 relu8_param_a = 0.1;
    vx_float32 relu8_param_b = 0;
    vx_node relu8_node;
    relu8_node = vxActivationLayer(graph, conv8_scale, relu8_mode, relu8_param_a, relu8_param_b, relu8);
    ERROR_CHECK_OBJECT(relu8_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&relu8_node));

    // conv9 Layer
    vx_size conv9_W_dims[4] = { 1, 1, 1024, 425 };
    vx_tensor conv9_W;
    conv9_W = vxCreateTensor(context,4, conv9_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv9_W); 
    ERROR_CHECK_STATUS(copyTensor(conv9_W, dataFolder + "/weights/conv9.f32"));
    vx_size conv9_B_dims[1] = { 425 };
    vx_tensor conv9_B;
    conv9_B = vxCreateTensor(context,1, conv9_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv9_B); 
    ERROR_CHECK_STATUS(copyTensor(conv9_B, dataFolder + "/bias/conv9.f32"));
    vx_nn_convolution_params_t conv9_params;
    conv9_params.padding_x = 0;
    conv9_params.padding_y = 0;
    conv9_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv9_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv9_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv9_params.dilation_x = 0;
    conv9_params.dilation_y = 0;
    vx_node conv9_node;
    conv9_node = vxConvolutionLayer(graph, relu8, conv9_W, conv9_B, &conv9_params, sizeof(conv9_params ), conv9);
    ERROR_CHECK_OBJECT(conv9_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv9_node));

    ////
    // release intermediate objects
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv4_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu4));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool4));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv5_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu5));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool5));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv6_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu6));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool6));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv8_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&relu8));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv9_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv9_B));

    ////
    // verify the built graph
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    return graph;
}
