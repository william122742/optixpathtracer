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
#include "pathTracer.h"
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <stdio.h>



extern "C" {
__constant__ Params params;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    float3   emitted;
    float3   radiance;
    float3   curbrdf;
    float3   origin;
    float3   direction;
    uint32_t seed;
    int32_t  countEmitted;
    int32_t  done;
    uint32_t  inmat;
};

struct Onb
{
    __forceinline__ __device__ Onb( const float3& normal )
    {
        m_normal = normal;

        if( fabs( m_normal.x ) > fabs( m_normal.z ) )
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize( m_binormal );
        m_tangent  = cross( m_binormal, m_normal );
    }

    __forceinline__ __device__ void inverse_transform( float3& p ) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

#define print_x 512
#define print_y 384

#define print_pixel(...)                                                       \
{                                                                              \
    const uint3  idx__ = optixGetLaunchIndex();                                \
    if( idx__.x == print_y && idx__.y == print_x )                             \
        printf( __VA_ARGS__ );                                                 \
}


static __forceinline__ __device__ void* unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}


static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<uint32_t>( occluded ) );
}

// pdf = cos_theta/(pi*r^2) ?
static __forceinline__ __device__ void cosine_sample_hemisphere( const float u1, const float u2, float3& p )
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f * M_PIf * u2;
    p.x             = r * cosf( phi );
    p.y             = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x * p.x - p.y * p.y ) );
}


static __forceinline__ __device__ bool same_hemisphere(const float3& l, const float3& v, const float3& n) {
    return dot(l, n) * dot(v, n) > 0.f;
}


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    // TODO: deduce stride from num ray-types passed in params

    uint32_t u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    uint32_t u0 = false;

        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0);
    
    return u0;
}


__forceinline__ __device__ uchar4 make_color( const float3&  c )
{
    const float gamma = 2.2f;
    return make_uchar4(
            static_cast<uint8_t>( powf( clamp( c.x, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<uint8_t>( powf( clamp( c.y, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<uint8_t>( powf( clamp( c.z, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            255u
            );
}

__forceinline__ __device__ float SchlickFresnel(float n1, float n2, float cos_h) {
    float RF_0 = (n1 - n2) / (n1 + n2);
    RF_0 = RF_0 * RF_0;
    float m = cos_h;
    float m2 = m * m;
    return RF_0 + (1.0f - RF_0) * (m2 * m2 * m);
}

__forceinline__ __device__ float Fresnel(float cos_i,float cos_t,  float eta) {

    if (cos_i*cos_i <= 1.f-eta*eta){
        return 0.f;
    }
    float a = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    float b = (cos_t - eta * cos_i) / (cos_t + eta * cos_i);
    float ret = (a * a + b * b) * 0.5f;
    return ret;
    
}
__forceinline__ __device__ float X_plus(float m) {
    return (m > 0.f) ? 1.f : 0.f;
}
__forceinline__ __device__ float Smith_GGX(float m, float alpha) {
    float a = alpha * alpha;
    float b = m * m;
    return 1.f / (m + sqrtf(a + b - a * b));

}
__forceinline__ __device__ float D_GXX(float cos_m, float alpha) {
    float D_TR = alpha / ((alpha * alpha - 1.f) * cos_m * cos_m + 1.f);
    return D_TR * D_TR;
}

__forceinline__ __device__ float3 calcBRDF(BRDFMaterial* cur_mat, float3 l, float3 v, float3 n) {
    // are all pre-divide by M_PI!!!
    float3 f_diff = make_float3(0.f);
    if (same_hemisphere(l, v, n)) {
        f_diff = cur_mat->base_color;
    }
    if (cur_mat->metallic != 0 || cur_mat->spec_trans != 0) {
        float f_spec = 0.f;
        float froughness = float(cur_mat->roughness) / 100.f;
        float alpha = froughness * froughness;
        float fmetallic = float(cur_mat->metallic) / 100.f;
        
        float3 h = normalize(l + v);

        float cos_l = dot(l, n);
        float cos_v = dot(v, n);
        float cos_h = dot(h, n);      

        if (same_hemisphere(l, v, n) && cur_mat->metallic != 0) {
            float Gs = Smith_GGX(cos_v, alpha) * Smith_GGX(cos_l, alpha);
            float D_TR = D_GXX(cos_h, alpha);
            float RF_h = SchlickFresnel(1.0f, cur_mat->ior, dot(v,h)); // interaction with air
            float X_vh = X_plus(dot(v, h));
            float X_lh = X_plus(dot(l, h));
            f_spec = D_TR * Gs * RF_h*(X_vh * X_lh);
        } 

        if (cur_mat->spec_trans != 0){
            float fspec_trans = float(cur_mat->spec_trans)/100.f;
            float f_trans;
            if (cos_l == 0.f || cos_v == 0.f || same_hemisphere(l,v,n)) {
                f_trans = 0.f;
            }
            else {
                bool exiting = (cos_l > 0.f);
                float eta_l = exiting ? 1.f : cur_mat->ior;
                float eta_v = exiting ? cur_mat->ior: 1.f;
                float eta = eta_l / eta_v;

                float3 ht = -normalize(l*eta + v);
                
                float cos_ht = dot(ht, n);
                float D_TRt = D_GXX(cos_ht, alpha);
                float Gst = Smith_GGX(fabs(cos_v), alpha) * Smith_GGX(fabs(cos_l), alpha);

                float v_dot_ht = dot(v, ht);
                float l_dot_ht = dot(l, ht);
                float X_vht = X_plus(v_dot_ht);
                float X_lht = X_plus(l_dot_ht);
                

                float RF_ht = Fresnel(cos_v, cos_ht, eta);
                //float RF_ht = SchlickFresnel(eta_v, eta_l, dot(v, ht));
                
                float tmp = eta / (v_dot_ht + l_dot_ht * eta);
                tmp = fabs(l_dot_ht) * fabs(v_dot_ht) * (tmp * tmp);
                f_trans = tmp * (1.0f - RF_ht) * D_TRt * Gst;// *(X_vht * X_lht);
                
            }
            return (f_diff * (1.f - fspec_trans) + f_trans * fspec_trans) * (1.f - fmetallic) + f_spec * fmetallic;
            
        }
        return f_diff * (1 - fmetallic) + f_spec *fmetallic; // linear interp of diffuse and specularity
    }
    return f_diff;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    uint32_t seed = tea<4>( idx.y*w + idx.x, subframe_index );

    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;
    do
    {
        const float2 subpixel_jitter = make_float2( rnd( seed )-0.5f, rnd( seed )-0.5f );

        const float2 d = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                ) - 1.0f;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.emitted      = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.curbrdf  = make_float3(1.f);
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;
        prd.direction = ray_direction;
        prd.origin = ray_origin;
        prd.inmat = false;

        float3 throughput = make_float3(1.f);
        
        int depth = 0;
        for( ;; )
        {
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd );
            result += prd.emitted*throughput;  // L_e term 
            result += prd.radiance * throughput;//prd.throughput;

            throughput *= prd.curbrdf;

            if( prd.done  || depth >= 6)//3 ) // TODO RR, variable for depth
                break;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    }
    while( --i );

    const uint3    launch_index = optixGetLaunchIndex();
    const uint32_t image_index  = launch_index.y * params.width + launch_index.x;
    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3( rt_data->r, rt_data->g, rt_data->b );
    prd->done     = true;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );   
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    RadiancePRD*  prd     = getPRD();
    
    const int          prim_idx        = optixGetPrimitiveIndex();
    const float3       ray_dir         = optixGetWorldRayDirection();
    const int          vert_idx_offset = prim_idx*3;
    const unsigned int hit_kind        = optixGetHitKind();

    float3 N; 
    
    // get hit point's setting
    if( optixIsTriangleHit() )
    {
        const float3 v0  = make_float3( rt_data->vertices[vert_idx_offset + 0] );
        const float3 v1  = make_float3( rt_data->vertices[vert_idx_offset + 1] );
        const float3 v2  = make_float3( rt_data->vertices[vert_idx_offset + 2] );
        const float3 N_0 = normalize( cross( v1 - v0, v2 - v0 ) );
        if (prd->inmat) {
            N = faceforward(N_0, ray_dir, N_0);
        }
        else {
            N = faceforward(N_0, -ray_dir, N_0);
        }
    }
    else
    {
        N = make_float3(int_as_float( optixGetAttribute_0() ),
                        int_as_float( optixGetAttribute_1() ),
                        int_as_float( optixGetAttribute_2() ));
    } 

    BRDFMaterial* cur_mat = &(params.materials[rt_data->mat_id]);
    
    float oldnDl = 2.f*dot(-prd->direction,N);
    // sampling emission only once to avoid noise
    prd->emitted = ( prd->countEmitted ) ? cur_mat->emission*oldnDl : make_float3( 0.0f ); // shuold mult by cos_theta'?
    
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    uint32_t seed = prd->seed;


    float3 w_in;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        
        cosine_sample_hemisphere( z1, z2, w_in );
        Onb onb(N);
        onb.inverse_transform(w_in);
        
        if (prd->inmat == false && cur_mat->spec_trans != 0){// && rnd(seed) > 0.5f) {   
           w_in = w_in*-N*2.0f*dot(N,w_in);
            prd->inmat = true;
        } else {      
            prd->inmat = false;
        }

        prd->curbrdf = calcBRDF(cur_mat, normalize(w_in), -normalize(prd->direction), normalize(N));

        prd->countEmitted = false;
    }
   
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);
        prd->seed = seed;

        ParallelogramLight light = params.light;
        const float3       light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // approximate current L(pn->pn-1)
        // calculate Geo term G(p_n<->p_n-1)
        const float  Ldist = length(light_pos - P);
        const float3 L = normalize(light_pos - P);
        const float  nDl = dot(N, L);
        const float  LnDl = -dot(light.normal, L);
        

        float weight = 0.0f;
        prd->radiance = make_float3(0.f);
        if (nDl > 0.0f && LnDl > 0.0f && (prd->inmat == false|| rnd(seed)>0.5f ))
        {
            const bool occluded = traceOcclusion(
                params.handle,
                P,
                L,
                0.01f,         // tmin
                Ldist - 0.01f  // tmax
            );

            if (!occluded)
            {
                const float dA = length(cross(light.v1, light.v2));
                weight = nDl * LnDl * dA / max(M_PIf * Ldist * Ldist, 0.001f);
                prd->radiance = (calcBRDF(cur_mat, normalize(L), -normalize(prd->direction), normalize(N)) * light.emission) * weight*oldnDl; //shuold add cos_theta' ?
            }
            
        }
        
        

        prd->direction = w_in;
        prd->origin = P;
    
}