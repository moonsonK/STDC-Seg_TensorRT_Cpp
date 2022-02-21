#include "Segmenter.hpp"

Segmenter::Segmenter(std::string& PathToEngineFile)
{
    mPathToEngineFile = PathToEngineFile;
}


Segmenter::~Segmenter()
{
    
    
}

bool Segmenter::LoadModel()
{
    nvinferlogs::gLogInfo << "Loading STDC engine file (" << mPathToEngineFile << ")...\n";
    nvinferlogs::gLogInfo.flush();
    
    // Open engine file
    std::ifstream EngineFile(mPathToEngineFile, std::ios::binary);
    
    // Error out if failed to open
    if (EngineFile.fail())
    {
        nvinferlogs::gLogError << "Error: failed to open engine file located at: " << mPathToEngineFile << ".\n"; 
        
        return false;
    }

    // Find file size and return to beginning of file
    EngineFile.seekg(0, std::ifstream::end);
    auto FileSize = EngineFile.tellg();
    EngineFile.seekg(0, std::ifstream::beg);

    // Read binary file data into vector of chars
    std::vector<char> EngineFileContents(FileSize);
    EngineFile.read(EngineFileContents.data(), FileSize);

    // 'Runtime' object allows a serialized engine to be deserialized. 
    // TensorRT’s builder and engine require a logger to capture errors, warnings, and 
    // other information during the build and inference phases
    std::unique_ptr<nvinfer1::IRuntime, TensorRTDeleter> RuntimeObj{nvinfer1::createInferRuntime(nvinferlogs::gLogger.getTRTLogger())};
    
    // Since runtime is a pointer to a IRuntime object, use -> operator instead of . operator
    // reset() destroys the object mEngine used to reference and assigns new responsibility
    // Appeal of unique_ptr's are that the automatically delete the object they manage as soon as they are destroyed
    mEnginePtr.reset(RuntimeObj->deserializeCudaEngine(EngineFileContents.data(), FileSize));
    
    // Error out if mEngine is null ptr
    if(mEnginePtr.get() == nullptr)
    {
        nvinferlogs::gLogError << "Error: failed to deserialize engine file.\n";
        
        return false;
    } 
    else
    {
        nvinferlogs::gLogInfo << "Successfully loaded engine file.\n";
        nvinferlogs::gLogInfo.flush();

        return true;
    }
}


size_t Segmenter::ComputeTensorSizeInBytes(nvinfer1::Dims& TensorDimensions, int32_t SizeOfOneElement)
{
    return std::accumulate(TensorDimensions.d, TensorDimensions.d + TensorDimensions.nbDims, 1, 
                           std::multiplies<int64_t>()) * SizeOfOneElement;
}

bool Segmenter::AllocateMemory()
{
    nvinferlogs::gLogInfo << "Allocating memory for input and output tensors...\n";
    nvinferlogs::gLogInfo.flush();
    
    // A TensorRT 'context' encapsulates model execution 'state'.
    // E.g., contains information about helper tensor sizes used in nn layers between input and output layers. 
    mExecutionContext.reset(mEnginePtr->createExecutionContext());

    if (!mExecutionContext)
    {
        nvinferlogs::gLogError << "Could not create execution context.\n";
        
        return false;
    }
    
    // Model file defines a mapping that assigns tensor names to integer identifiers (indices)
    // Get the identifying index of the input tensor (named input), error out if you can't find it.
    int32_t InputTensorIdx = mEnginePtr->getBindingIndex("input");
    if (InputTensorIdx == -1)
    {
        nvinferlogs::gLogError << "Could not find tensor \'input\'.\n";
        
        return false;
    }
    
    // Batch, channel, height, width format
    nvinfer1::Dims InputTensorDimensions = mExecutionContext->getBindingDimensions(InputTensorIdx);
    
    // Populate member vars based on input size
    mRequiredImageWidth = InputTensorDimensions.d[3];
    mRequiredImageHeight = InputTensorDimensions.d[2];
    mInputCpuBuffer = (float*)malloc(3*mRequiredImageWidth*mRequiredImageHeight*sizeof(float)); 
    
    // Get total bytes needed to store input image
    int InputTensorSizeInBytes = ComputeTensorSizeInBytes(InputTensorDimensions, sizeof(float));
    mIoTensorMemorySizesInBytes.push_back(InputTensorSizeInBytes);
                                   
    // Repeat with output tensor
    int32_t OutputTensorIdx = mEnginePtr->getBindingIndex("output");
    if (OutputTensorIdx == -1)
    {
        return false;
    }
    nvinfer1::Dims OutputTensorDimensions = mExecutionContext->getBindingDimensions(OutputTensorIdx);
    int OutputTensorSizeInBytes = ComputeTensorSizeInBytes(OutputTensorDimensions, sizeof(float));
    mIoTensorMemorySizesInBytes.push_back(OutputTensorSizeInBytes);

    // Output buffer size (1x19xHxW)
    mOutputCpuBuffer = (float*)malloc(mNumClasses*mRequiredImageWidth*mRequiredImageHeight*sizeof(float));

    // cudaMalloc takes address of ptr (void**) that will point at allocated memory, and size of allocated memory 
    if (cudaMalloc(&mGpuMemoryBindings[0], InputTensorSizeInBytes) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: CUDA memory allocation of input tensor failed, size = " << InputTensorSizeInBytes << " bytes" << std::endl;
        return false;
    }
    
    // Allocate outputs memory
    if (cudaMalloc(&mGpuMemoryBindings[1], OutputTensorSizeInBytes) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: CUDA memory allocation of output tensor failed, size = " << OutputTensorSizeInBytes << " bytes" << std::endl;
        return false;
    }
    
    // Construct stream
    if (cudaStreamCreate(&mStream) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }
    
    nvinferlogs::gLogInfo << "Successfully allocated memory.\n";
    nvinferlogs::gLogInfo.flush();


    return true;
}
/*

bool TensorRTDetector::WarmUpModel()
{
    // Populate input buffer with all zeros
    for (int FlatPixelIdx = 0; FlatPixelIdx < 3*mNumInputPixelsWithPadding; FlatPixelIdx++) 
    {
        // HAS TO BE BLACK!!!!!!
        mInputBuffer[FlatPixelIdx] = 0.0;
        // mInputBuffer[FlatPixelIdx] = 127.0;
    }
    
    // Copy image data to input memory
    if (cudaMemcpyAsync(mMemoryBindings[0], mInputBuffer, mMemorySizes[0], cudaMemcpyHostToDevice, mStream) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: CUDA memory copy of input failed, size = " << mMemorySizes[0] << " bytes" << std::endl;
        return false;
    }
    
    // Status messages
    if(!mSilentMode)
    {
        nvinferlogs::gLogInfo << "Performing warmup...\n";
        nvinferlogs::gLogInfo.flush();
    }
      
    // Three rounds inference
    for(int WarmUpIdx = 0; WarmUpIdx < 3; WarmUpIdx++)
    {   
        // Asynchronously execute inference. enqueueV2(array of pts to input and output nn buffers, cuda stream, N/A)
        bool InferenceStatus = mExecutionContext->enqueueV2(mMemoryBindings, mStream, nullptr);
        if (!InferenceStatus)
        {
            nvinferlogs::gLogError << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        } 
        
        cudaStreamSynchronize(mStream);
    }
    
    // Status messages
    if(!mSilentMode)
    {
        nvinferlogs::gLogInfo << "Successfully completed warmup.\n\n";
        nvinferlogs::gLogInfo.flush();
    }
    
    return true;
    
} 

bool TensorRTDetector::LoadAndPrepareModel()
{
    // Load 
    bool LoadSuccessful = true;
    if(mInitializedWithEnginePtr)
    {
        nvinferlogs::gLogInfo << "Preparing yolov5 class instance initialized with engine pointer...\n";
        nvinferlogs::gLogInfo.flush();
    }
    else
    {
        LoadSuccessful = LoadModel();
    }

    // Allocate
    bool AllocateSuccessful = AllocateMemory();

    // NumClasses is determined in allocate memory, which this method relies on
    RegisterClassesToDetect();
    
    // Warm up
    bool WarmupSuccessful = WarmUpModel();

    return (LoadSuccessful && AllocateSuccessful && WarmupSuccessful);
}*/