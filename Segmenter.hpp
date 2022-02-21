#ifndef SEGMENTER_HPP
#define SEGMENTER_HPP

#include "logger.h"
#include <memory>   //! defines unique_ptr
#include <vector>
#include <fstream>  // fopen
#include <numeric>  // accumulate

#include "NvInfer.h"

class Segmenter {

    public:
    
        struct TensorRTDeleter
        {
            // Arbitrary type
            template <typename T>
            
            // Overloads (), so that if InfDel is an instance of InferDeleter, then InfDel(poo) calls this function w/ obj = poo 
            
            // The cleverness here is that this causes InferDeleter to be CALLABLE, as is required as the deleter of a unique ptr obj.
            // Moreover, InferDeleter is implicitly called with the managed object to delete as the sole argument, hence why this function
            // accepts one pointer of type T, and destroys it
            void operator()(T* obj) const
            {
                if (obj)
                {
                    // All TensorRT objects that UniquePtr's can point to have a destroy() method. 
                    obj->destroy();
                }
            }
        };

        Segmenter(std::string&);
        ~Segmenter();

        std::string mPathToEngineFile = "";


        bool LoadModel();
        bool AllocateMemory();
        size_t ComputeTensorSizeInBytes(nvinfer1::Dims&, int32_t);

    private:

        // bool LoadModel();
        // bool AllocateMemory();

        std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter> mEnginePtr = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext, TensorRTDeleter> mExecutionContext = nullptr;
        int mRequiredImageWidth = -1;
        int mRequiredImageHeight = -1;
        float* mInputCpuBuffer = nullptr;
        std::vector<int> mIoTensorMemorySizesInBytes;
        float* mOutputCpuBuffer = nullptr;
        int mNumClasses = 19;
        void* mGpuMemoryBindings[2] = {nullptr, nullptr};
        cudaStream_t mStream;
};

#endif /* SEGMENTER_HPP */