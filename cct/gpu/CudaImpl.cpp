//-----------------------------------------------------------------------------
// Copyright 2016 Chuck Seberino
//
// This file is part of CCT.
//
// CCT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CCT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CCT.  If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------
#include "CudaImpl.hpp"
#include "CudaDevice.hpp"

#include <algorithm>
#include <cuda_runtime.h>

#ifdef CCT_USE_GPU_PROFILING
#   include <nvToolsExt.h>
#   if defined(CCT_MSVC)
#      include <Windows.h>
void CCT::EventTracer::setThreadName(const char* name)
{ nvtxNameOsThread(GetCurrentThreadId(), name); }
#   elif defined(CCT_DARWIN)
void CCT::EventTracer::setThreadName(const char* name)
{ uint64_t tid; pthread_threadid_np(NULL, &tid); nvtxNameOsThread(tid, name); }
#   else // CCT_GCC
void CCT::EventTracer::setThreadName(const char* name)
{ nvtxNameOsThread(pthread_self(), name); }
#   endif

CCT::EventTracer::EventTracer(const char* name, uint32_t color)
{
    if (color)
    {
        nvtxEventAttributes_t eventAttrib = { 0 };
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name;
        nvtxRangePushEx(&eventAttrib);
    }
    else nvtxRangePushA(name);
}
CCT::EventTracer::~EventTracer()
{
    nvtxRangePop();
}
#endif

namespace CCT {

//-----------------------------------------------------------------------------
Impl::Impl(DevicePtr _device)
    : device(_device)
    , thrustCache(NumStreams, ThrustAllocator(this))
    , streamPtr(NumStreams)
    , eventStart(NumEvents)
    , eventStop(NumEvents)
{
    //-------------------------------------------------------------------------
    // Create independent streams of operation and sync/timing events 
    //-------------------------------------------------------------------------
    for (int ii = 0; ii<NumStreams; ++ii)
    {
        CCT_CHECK_GPU(cudaStreamCreateWithFlags(&streamPtr[ii], cudaStreamNonBlocking));
    }

    for (int ii=0; ii<NumEvents; ++ii)
    {
        CCT_CHECK_GPU(cudaEventCreate(&eventStart[ii]));
        CCT_CHECK_GPU(cudaEventCreate(&eventStop[ii]));
    }
}


//-----------------------------------------------------------------------------
Impl::~Impl()
{
    // Device never initialized or no CUDA hardware exists
    if (!device) return;

    thrustCache.clear();

    // Make sure and flush remaining work before exiting.
    for (int ii = 0; ii < NumStreams; ++ii)
    {
        synchronize(ii);
    }

    // Clean up shared references to memory.
    hostPtr.clear();
    gpuPtr.clear();

    for (int ii = 0; ii < NumStreams; ++ii)
    {
        CCT_CHECK_GPU(cudaStreamDestroy(streamPtr[ii]));
    }

    for (int ii = 0; ii < NumEvents; ++ii)
    {
        CCT_CHECK_GPU(cudaEventDestroy(eventStart[ii]));
        CCT_CHECK_GPU(cudaEventDestroy(eventStop[ii]));
    }
    CCT_TRACE("GPU Worker on device " << device->ID() << " complete");
}


//-----------------------------------------------------------------------------
void* Impl::allocate(size_t size, MemType type)
{
    void *ptr = NULL;
    if (GPU == type)
    {
        CCT_CHECK_GPU(cudaMalloc(&ptr, size));
        if (ptr) gpuPtr.push_back(SharedMem(ptr, cudaFree));
    }
    else
    {
        CCT_CHECK_GPU(cudaMallocHost(&ptr, size));
        if (ptr) hostPtr.push_back(SharedMem(ptr, cudaFreeHost));
    }
    return ptr;
}


//-----------------------------------------------------------------------------
void Impl::free(const void* ptr)
{
    MemoryVectorPtr::iterator it;
    // First try in the GPU mem list
    for (it=gpuPtr.begin(); it!=gpuPtr.end(); ++it)
    {
        if (ptr == it->get())
        {
            gpuPtr.erase(it);
            return;
        }
    }
    // Next try host memory
    for (it=hostPtr.begin(); it!=hostPtr.end(); ++it)
    {
        if (ptr == it->get())
        {
            hostPtr.erase(it);
            return;
        }
    }
    // else
    CCT_WARN("Attempt to free unknown data: " << ptr);
}


//-----------------------------------------------------------------------------
void Impl::implcopy(void* to, const void* from, size_t size, MemType type, size_t index)
{
    cudaMemcpyKind kind;
    switch(type)
    {
    case GPU: kind = cudaMemcpyHostToDevice; break;
    case CPU: kind = cudaMemcpyDeviceToHost; break;
    case DEV: kind = cudaMemcpyDeviceToDevice; break;
    }
    CCT_CHECK_GPU(cudaMemcpyAsync(to, from, size, kind, stream(index)));
}


//-----------------------------------------------------------------------------
void Impl::implset(void* to, int value, size_t size, size_t index)
{
    CCT_CHECK_GPU(cudaMemsetAsync(to, value, size, stream(index)));
}


//-----------------------------------------------------------------------------
cudaStream_t Impl::stream(size_t index /*=EventStream*/)
{
    if (index >= streamPtr.size()) index = EventStream;
    return streamPtr[index];
}


//-----------------------------------------------------------------------------
void Impl::streamWait(size_t streamIndex, size_t eventIndex)
{
    CCT_CHECK_GPU(cudaStreamWaitEvent(stream(streamIndex), eventStop[eventIndex], 0));
}


//-----------------------------------------------------------------------------
void Impl::timerStart(size_t eventIndex, size_t streamIndex /*=~0*/)
{
    if (~0 == streamIndex)
    {
        streamIndex = std::min<size_t>(eventIndex, EventStream);
    }
    CCT_CHECK_GPU(cudaEventRecord(eventStart[eventIndex], stream(streamIndex)));
}


//-----------------------------------------------------------------------------
void Impl::timerStop(size_t eventIndex, size_t streamIndex /*=~0*/)
{
    if (~0 == streamIndex)
    {
        streamIndex = std::min<size_t>(eventIndex, EventStream);
    }
    CCT_CHECK_GPU(cudaEventRecord(eventStop[eventIndex], stream(streamIndex)));
}


//-----------------------------------------------------------------------------
float Impl::timerElapsed(size_t eventIndex)
{
    float timeEvent = 0.f;
    // Make sure our stop event has finished, but only if the eventIndex is
    // aligned with the stream index
    if (eventIndex <= EventStream) streamWait(eventIndex, eventIndex);

    CCT_CHECK_GPU(cudaEventElapsedTime(&timeEvent, eventStart[eventIndex], eventStop[eventIndex]));
    return timeEvent;
}


//-----------------------------------------------------------------------------
void Impl::synchronize(size_t streamIndex /*=EventStream*/)
{
    CCT_CHECK_GPU(cudaStreamSynchronize(stream(streamIndex)));
}


//-----------------------------------------------------------------------------
void Impl::deviceSynchronize()
{
    CCT_CHECK_GPU(cudaDeviceSynchronize());
}


//-----------------------------------------------------------------------------
void Impl::getLastError(char const* const func, const char* const file, int const line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        CCT_ERROR("CUDA error at " << file << ":" << line << " rc = "
            << err << "(" << cudaGetErrorString(err) << ") " << func);
    }
}


//-----------------------------------------------------------------------------
ThrustAllocator::ptr_type ThrustAllocator::allocate(std::ptrdiff_t size)
{
    ptr_type result = NULL;
    FreeBlockMap::iterator freeBlock = freeBlockMap.find(size);
    if (freeBlock != freeBlockMap.end())
    {
        result = freeBlock->second;
        freeBlockMap.erase(freeBlock);
    }
    else
    {
        gpuPtr->alloc(result, size);
    }

    allocatedBlockMap.emplace(result, size);
    return result;
}


//-----------------------------------------------------------------------------
void ThrustAllocator::deallocate(ptr_type ptr, size_t)
{
    // Move memory from allocated to free map
    AllocatedBlockMap::iterator it = allocatedBlockMap.find(ptr);
    if (it == allocatedBlockMap.end())
    {
        CCT_ERROR("Attempt to remove unknown memory");
    }
    else
    {
        freeBlockMap.emplace(it->second, it->first);
        allocatedBlockMap.erase(it);
    }
}

} // namespace CCT
