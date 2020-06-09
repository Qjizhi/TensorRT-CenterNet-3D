1. Replace THC-based DCNv2 with ATen-based DCNv2. 
If it is not replaced, you will get (TypeError: int() not supported on cdata 'struct THLongTensor *') when converting onnx, and I have no idea to solve it.
So I use DCNv2 from mmdetection.
    * copy the dcn to lib/models/netowrks
        ```bash
        cp -r dcn lib/models/netowrks
        ```
    * upgrade pytorch to 1.0-1.1
    * complie Deform Conv
        ```bash
        cd lib/models/netowrks/dcn
        python setup.py build_ext --inplace
        ``` 

2. Add symbolic to DeformConvFunction.
    ```python
    class ModulatedDeformConvFunction(Function):
    
        @staticmethod
        def symbolic(g, input, offset, mask, weight, bias,stride,padding,dilation,groups,deformable_groups):
            return g.op("DCNv2", input, offset, mask, weight, bias,
                        stride_i = stride,padding_i = padding,dilation_i = dilation,
                        groups_i = groups,deformable_group_i = deformable_groups)
        @staticmethod
        def forward(ctx,
                    input,
                    offset,
                    mask,
                    weight,
                    bias=None,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    deformable_groups=1):
                    pass#.......
    ```
3. Change import
   * change (from .DCNv2.dcn_v2 import DCN) to (from .dcn.modules.deform_conv import ModulatedDeformConvPack as DCN) in pose_dla_dcn.py and resnet_dcn.py.
   * Now you can convert the model using Deform Conv to onnx.
   
4. For dla34.  
Convert [ddd_3dop.pth](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md) to `ddd_3dop.onnx`:  
- copy `export_onnx_ddd.py` to the [Official Repo of CenterNet](https://github.com/xingyizhou/CenterNet) under the path `CenterNet/src` 
- change `lib/opts.py` `add_argument('task', default='ctdet'....)` to `add_argument('--task', default='ctdet'....)`
- run `python export_onnx_ddd.py --task ddd`

*   If you get (ValueError: Auto nesting doesn't know how to process an input object of type int. Accepted types: Tensors, or lists/tuples of them)
    You need to change (def _iter_filter) in torch.autograd.function.
    ```python
       def _iter_filter(....):
           ....
           if condition(obj):
                yield obj
           elif isinstance(obj,int):  ## int to tensor
                yield torch.tensor(obj)
           ....

    ```
1. onnx-tensorrt DCNv2 plugin
    * Related code
        * onnx-tensorrt/builtin_op_importers.cpp
        * onnx-tensorrt/builtin_plugins.cpp
        * onnx-tensorrt/DCNv2.hpp
        * onnx-tensorrt/DCNv2.cpp
        * onnx-tensorrt/dcn_v2_im2col_cuda.cu
        * onnx-tensorrt/dcn_v2_im2col_cuda.h
    * Not only support centernet. If you want to convert other model to tensorrt engine, please refer to src/ctdetNet.cpp or contact me.
