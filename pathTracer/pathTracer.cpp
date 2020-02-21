//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include "pathTracer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


// Camera state
bool             camera_changed = true;
sutil::Camera    camera;

int32_t          samples_per_launch = 16;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct Instance
{
    float transform[12];
};


struct CutoutsState
{
    OptixDeviceContext          context                      = 0;

    OptixTraversableHandle      triangle_gas_handle          = 0;  // Traversable handle for triangle AS
    CUdeviceptr                 d_triangle_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                 d_vertices                   = 0;
    CUdeviceptr                 d_tex_coords                 = 0;

    OptixTraversableHandle      ias_handle                   = 0;  // Traversable handle for instance AS
    CUdeviceptr                 d_ias_output_buffer          = 0;  // Instance AS memory

    OptixModule                 ptx_module                   = 0;

    OptixPipelineCompileOptions pipeline_compile_options     = {};
    OptixPipeline               pipeline                     = 0;

    OptixProgramGroup           raygen_prog_group            = 0;
    OptixProgramGroup           radiance_miss_group          = 0;
    OptixProgramGroup           occlusion_miss_group         = 0;
    OptixProgramGroup           radiance_hit_group           = 0;
    OptixProgramGroup           occlusion_hit_group          = 0;

    CUstream                    stream                       = 0;
    Params                      params;
    Params*                     d_params;

    OptixShaderBindingTable     sbt = {};
};


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t TRIANGLE_COUNT =  32;
const int32_t TRIANGLE_MAT_COUNT = MAT_NUM;

const static std::array<Vertex, TRIANGLE_COUNT*3> g_vertices =
{ {
    // Floor  -- white lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,    0.0f, 0.0f },

    // Ceiling -- white lambert
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    // Back wall -- white lambert
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    // Right wall -- green lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },

    // Left wall -- red lambert
    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },

    // Short block -- white lambert
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },

    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {  240.0f,    0.0f,  272.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  290.0f,    0.0f,  114.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  130.0f,    0.0f,   65.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {   82.0f,    0.0f,  225.0f, 0.0f },
    
    // Tall block -- white lambert
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },

    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  472.0f,    0.0f,  406.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  314.0f,    0.0f,  456.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  265.0f,    0.0f,  296.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  423.0f,    0.0f,  247.0f, 0.0f },

    // Ceiling light -- emmissive
    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },

    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },
    {  343.0f,  548.6f,  332.0f, 0.0f }
} };


static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices =
{ {
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- cutout
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  //4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
} };

/**
const std::array<float3, TRIANGLE_MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f, 0.0f },
    {  0.0f,  0.0f, 0.0f },
    {  0.0f,  0.0f, 0.0f },
    { 15.0f, 15.0f, 5.0f },
    //{  0.0f,  0.0f, 0.0f }
} };


const std::array<float3, TRIANGLE_MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f },
    //{ 0.70f, 0.25f, 0.00f }
} };
*/


// NB: Some UV scaling is baked into the coordinates for the short block, since
//     the coordinates are used for the cutout texture.
const std::array<float2, TRIANGLE_COUNT* 3> g_tex_coords =
{ {
    // Floor
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Ceiling
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Back wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Right wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Left wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Short Block
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },

    // Tall Block
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },

    // Ceiling light
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f }
} };



void initMaterials(CutoutsState& state) {

    state.params.materials[0].base_color = { 0.80f, 0.80f, 0.80f };
    state.params.materials[0].emission = { 0.0f,  0.0f, 0.0f };
    state.params.materials[0].metallic = 0;
    state.params.materials[0].spec_trans = 0;

    state.params.materials[1].base_color = { 0.05f, 0.80f, 0.05f };
    state.params.materials[1].emission = { 0.0f,  0.0f, 0.0f };
    state.params.materials[1].metallic = 0;
    state.params.materials[1].spec_trans = 0;
    
    state.params.materials[2].base_color = { 0.80f, 0.05f, 0.05f };
    state.params.materials[2].emission = { 0.0f,  0.0f, 0.0f };
    state.params.materials[2].metallic = 0;
    state.params.materials[2].spec_trans = 0;

    state.params.materials[3].base_color = { 0.50f, 0.00f, 0.00f };
    state.params.materials[3].emission = { 15.0f, 15.0f, 5.0f };
    state.params.materials[3].metallic = 0;
    state.params.materials[3].spec_trans = 0;

    state.params.materials[4].base_color = { 0.80f, 0.80f, 0.80f };
    state.params.materials[4].emission = { 0.0f,  0.0f, 0.0f };
    state.params.materials[4].ior = 1.f;
    state.params.materials[4].metallic = 80;
    state.params.materials[4].roughness = 15;
    state.params.materials[4].spec_trans = 0;

    state.params.materials[5].base_color = { 0.80f, 0.80f, 0.80f };
    state.params.materials[5].emission = { 0.0f,  0.0f, 0.0f };
    state.params.materials[5].ior = 1.5f;
    state.params.materials[5].metallic = 0;
    state.params.materials[5].roughness = 40;
    state.params.materials[5].spec_trans = 100;
}


void initLaunchParams( CutoutsState& state )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.accum_buffer ),
                            state.params.width*state.params.height*sizeof(float4) ) );
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;

    initMaterials(state);

    state.params.light.emission = make_float3(   15.0f,  15.0f,   5.0f );
    state.params.light.corner   = make_float3(  343.0f, 548.5f, 227.0f );
    state.params.light.v1       = make_float3(    0.0f,   0.0f, 105.0f );
    state.params.light.v2       = make_float3( -130.0f,   0.0f,   0.0f );
    state.params.light.normal   = normalize  ( cross( state.params.light.v1,  state.params.light.v2) );

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

    state.params.handle = state.ias_handle;

}


void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed )
        params.subframe_index = 0;
    handleCameraUpdate( params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, CutoutsState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                &state.params,
                sizeof( Params ),
                cudaMemcpyHostToDevice,
                state.stream
                ) );

    OPTIX_CHECK( optixLaunch(
                 state.pipeline,
                 state.stream,
                 reinterpret_cast<CUdeviceptr>( state.d_params ),
                 sizeof( Params ),
                 &state.sbt,
                 state.params.width,  // launch width
                 state.params.height, // launch height
                 1                    // launch depth
                 ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


void initCameraState()
{
    camera.setEye( make_float3( 278.0f, 273.0f, -900.0f ) );
    camera.setLookat( make_float3( 278.0f, 273.0f, 330.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 35.0f );
    camera_changed = true;
}


void createContext( CutoutsState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}


void buildGeomAccel( CutoutsState& state )
{
    //
    // Build triangle GAS
    //
    {
        const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_vertices ), vertices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_vertices ), g_vertices.data(), vertices_size_in_bytes,
                                cudaMemcpyHostToDevice ) );

        CUdeviceptr  d_mat_indices             = 0;
        const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_mat_indices ), g_mat_indices.data(),
                                mat_indices_size_in_bytes, cudaMemcpyHostToDevice ) );

        const size_t tex_coords_size_in_bytes = g_tex_coords.size() * sizeof( float2 );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_tex_coords ), tex_coords_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_tex_coords ), g_tex_coords.data(),
                                tex_coords_size_in_bytes, cudaMemcpyHostToDevice ) );

        uint32_t triangle_input_flags[TRIANGLE_MAT_COUNT];
        for (int i = 0; i < TRIANGLE_MAT_COUNT; i++) {
            triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        }

        OptixBuildInput triangle_input                           = {};
        triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
        triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() * 3 );
        triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
        triangle_input.triangleArray.flags                       = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords               = TRIANGLE_MAT_COUNT;
        triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &triangle_input,
                                                   1,  // num_build_inputs
                                                   &gas_buffer_sizes ) );

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                    compactedSizeOffset + 8
                    ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild(
                    state.context,
                    0,              // CUDA stream
                    &accel_options,
                    &triangle_input,
                    1,              // num build inputs
                    d_temp_buffer,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_buffer_temp_output_gas_and_compacted_size,
                    gas_buffer_sizes.outputSizeInBytes,
                    &state.triangle_gas_handle,
                    &emitProperty,  // emitted property list
                    1               // num emitted properties
                    ) );

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_mat_indices ) ) );

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy(
                    &compacted_gas_size,
                    (void*)emitProperty.result,
                    sizeof( size_t ),
                    cudaMemcpyDeviceToHost
                    ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_triangle_gas_output_buffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context, 0, state.triangle_gas_handle, state.d_triangle_gas_output_buffer, compacted_gas_size, &state.triangle_gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
        }
        else
        {
            state.d_triangle_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }
}


void buildInstanceAccel( CutoutsState& state )
{
    CUdeviceptr d_instances;
    size_t      instance_size_in_bytes = sizeof( OptixInstance ) ;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), instance_size_in_bytes ) );

    OptixBuildInput instance_input = {};

    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &instance_input,
                                               1,  // num build inputs
                                               &ias_buffer_sizes ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), ias_buffer_sizes.tempSizeInBytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), ias_buffer_sizes.outputSizeInBytes ) );

    // Use the identity matrix for the instance transform
    Instance instance = { { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 } };

    OptixInstance optix_instances[1];
    memset( optix_instances, 0, instance_size_in_bytes );

    optix_instances[0].traversableHandle = state.triangle_gas_handle;
    optix_instances[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[0].instanceId        = 0;
    optix_instances[0].sbtOffset         = 0;
    optix_instances[0].visibilityMask    = 1;
    memcpy( optix_instances[0].transform, instance.transform, sizeof( float ) * 12 );

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_instances ), &optix_instances, instance_size_in_bytes,
                            cudaMemcpyHostToDevice ) );

    OPTIX_CHECK( optixAccelBuild( state.context,
                                  0,  // CUDA stream
                                  &accel_options,
                                  &instance_input,
                                  1,  // num build inputs
                                  d_temp_buffer,
                                  ias_buffer_sizes.tempSizeInBytes,
                                  state.d_ias_output_buffer,
                                  ias_buffer_sizes.outputSizeInBytes,
                                  &state.ias_handle,
                                  nullptr,  // emitted property list
                                  0         // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_instances   ) ) );
}


void createModule( CutoutsState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = 100;

    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur            = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipeline_compile_options.numPayloadValues          = 2;
    state.pipeline_compile_options.numAttributeValues        = 4;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "pathTracer.cu" );
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &state.ptx_module
                ) );
}


void createProgramGroups( CutoutsState& state )
{
    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc    = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context,
                                              &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options,
                                              log, &sizeof_log,
                                              &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
    sizeof_log                                  = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context,
                                              &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options,
                                              log, &sizeof_log,
                                              &state.radiance_miss_group ) );

    memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = nullptr;  // miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    sizeof_log                                  = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.occlusion_miss_group ) );

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context,
                                              &hit_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options,
                                              log,
                                              &sizeof_log,
                                              &state.radiance_hit_group ) );

    memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";


    sizeof_log = sizeof( log );
    OPTIX_CHECK( optixProgramGroupCreate( state.context,
                                          &hit_prog_group_desc,
                                          1,  // num program groups
                                          &program_group_options,
                                          log,
                                          &sizeof_log,
                                          &state.occlusion_hit_group ) );
}


void createPipeline( CutoutsState& state )
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.occlusion_miss_group,
        state.radiance_hit_group,
        state.occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 2;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur   = false;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context,
                                          &state.pipeline_compile_options,
                                          &pipeline_link_options,
                                          program_groups,
                                          sizeof( program_groups ) / sizeof( program_groups[0] ),
                                          log,
                                          &sizeof_log,
                                          &state.pipeline ) );
}


void createSBT( CutoutsState& state )
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    rg_sbt.data = {1.0f, 0.f, 0.f};

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_raygen_record ), &rg_sbt, raygen_record_size,
                            cudaMemcpyHostToDevice ) );

    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size * RAY_TYPE_COUNT ) );

    MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_miss_group, &ms_sbt[0] ) );
    ms_sbt[0].data = {0.0f, 0.0f, 0.0f};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_miss_group, &ms_sbt[1] ) );
    ms_sbt[1].data = {0.0f, 0.0f, 0.0f};

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_miss_records ), ms_sbt, miss_record_size * RAY_TYPE_COUNT,
                            cudaMemcpyHostToDevice ) );

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_records ),
                            hitgroup_record_size * ( RAY_TYPE_COUNT *  TRIANGLE_MAT_COUNT  ) ) );

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT *  TRIANGLE_MAT_COUNT];

    // Set up the HitGroupRecords for the triangle materials
    for( int i = 0; i < TRIANGLE_MAT_COUNT; ++i )
    {
        {
            const int sbt_idx = i*RAY_TYPE_COUNT+0; // SBT for radiance ray-type for ith material

            OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[ sbt_idx ].data.vertices       = reinterpret_cast<float4*>(state.d_vertices);
            hitgroup_records[ sbt_idx ].data.tex_coords     = reinterpret_cast<float2*>(state.d_tex_coords);
            hitgroup_records[sbt_idx].data.mat_id = i;
        }

        {
            const int sbt_idx = i*RAY_TYPE_COUNT+1; // SBT for occlusion ray-type for ith material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[ sbt_idx ].data.vertices   = reinterpret_cast<float4*>(state.d_vertices);
            hitgroup_records[ sbt_idx ].data.tex_coords = reinterpret_cast<float2*>(state.d_tex_coords);
            hitgroup_records[sbt_idx].data.mat_id = i;
        }
    }

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_hitgroup_records ), hitgroup_records,
                            hitgroup_record_size * ( RAY_TYPE_COUNT *  TRIANGLE_MAT_COUNT  ),
                            cudaMemcpyHostToDevice ) );
    

    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_records;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * TRIANGLE_MAT_COUNT ;
}


void cleanupState( CutoutsState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_miss_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_vertices ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_tex_coords ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_triangle_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_ias_output_buffer ) ) );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    CutoutsState state;
    state.params.width  = 400;
    state.params.height = 400;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;


    try {
        initCameraState();
        //
        // Set up OptiX state
        //
        createContext      ( state );
        std::cout << "suc context\n";
        buildGeomAccel     ( state );
        std::cout << "suc geo accel\n";
        buildInstanceAccel ( state );
        std::cout << "suc ins geo accel\n";
        createModule       ( state );
        std::cout << "suc module \n";
        createProgramGroups( state );
        std::cout << "suc group\n";
        createPipeline     ( state );
        std::cout << "suc pipeline\n";
        createSBT          ( state );
        std::cout << "suc SBT \n";
        initLaunchParams( state );
        std::cout << "suc init params\n";


            GLFWwindow* window = sutil::initUI( "pathTracer", state.params.width, state.params.height );
            glfwSetWindowUserPointer  ( window, &state.params );
            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.width, state.params.height );
                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                do
                {
                    glfwPollEvents();
                    updateState(output_buffer, state.params);
                    launchSubframe(output_buffer, state);
                    displaySubframe(output_buffer, gl_display, window);
                    glfwSwapBuffers(window); 

                    ++state.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }
            sutil::cleanupUI( window );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
