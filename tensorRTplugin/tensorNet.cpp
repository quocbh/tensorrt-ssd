#include <algorithm>
//#include "common.h"
#include "tensorNet.h"

using namespace nvinfer1;

void TensorNet::caffeToTRTModel(const std::string& deployFile, const std::string& modelFile, const std::vector<std::string>& outputs,
                                unsigned int maxBatchSize)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();
    useFp16 = true;

    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;
    
    std::cout << deployFile.c_str() <<std::endl;
    std::cout << modelFile.c_str() <<std::endl;
    const IBlobNameToTensor* blobNameToTensor =	parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              modelDataType);

    assert(blobNameToTensor != nullptr);
    for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));
  
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 21);

    IRuntime* infer;
    infer = createInferRuntime(gLogger);
    std::cout << "Is it here 2.1" <<std::endl;
    if(useFp16)
    {
        builder->setHalf2Mode(true);
    }
    ICudaEngine* engine = builder->buildCudaEngine( *network );
    assert(engine);

    
    network->destroy();
    parser->destroy();

    std::cout << "This project is finished " << std::endl;
    gieModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
    shutdownProtobufLibrary();

}


void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    context = NULL;
    engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
    context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
	printf("Binding %d \n",engine->getNbBindings());
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize)
{
    assert( engine->getNbBindings()==nbBuffer);
    context->execute(batchSize, buffers);
    //context->destroy();
}

void TensorNet::timeInference(int iteration, int batchSize)
{
    int inputIdx = 0;
    size_t inputSize = 0;
    void* buffers[engine->getNbBindings()];

    for (int b = 0; b < engine->getNbBindings(); b++)
    {
        DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        size_t size = batchSize * dims.c() * dims.h() * dims.w() * sizeof(float);
        CHECK(cudaMalloc(&buffers[b], size));

        if(engine->bindingIsInput(b) == true)
        {
            inputIdx = b;
            inputSize = size;
        }
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    CHECK(cudaMemset(buffers[inputIdx], 0, inputSize));

    for (int i = 0; i < iteration;i++) context->execute(batchSize, buffers);

    context->destroy();
    for (int b = 0; b < engine->getNbBindings(); b++) CHECK(cudaFree(buffers[b]));

}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp( name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}



void TensorNet::printTimes(int iteration)
{
    gProfiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
    pluginFactory.destroyPlugin();
    engine->destroy();
    infer->destroy();
}
