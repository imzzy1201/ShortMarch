
struct CameraInfo {
	float4x4 screen_to_camera;
	float4x4 camera_to_world;

    float focal_distance;
    float aperture_radius;
    float pad0;
    float pad1;
};

struct Material {
    float3 base_color;  // [DEPRECATED]

    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 transmittance;  // [NOT USED]
    float3 emission;
    float shininess;       // [NOT USED]
    float ior;             // index of refraction
    float dissolve;        // 1 == opaque; 0 == fully transparent
    int illum;             // illumination model, [NOT USED]

    int ambient_tex_id;             // [NOT USED]
    int diffuse_tex_id;             // [NOT USED]
    int specular_tex_id;            // [NOT USED]
    int specular_highlight_tex_id;  // [NOT USED]
    int bump_tex_id;                // [NOT USED]
    int displacement_tex_id;        // [NOT USED]
    int alpha_tex_id;               // [NOT USED]
    int reflection_tex_id;          // [NOT USED]

    // PBR extensions
    float roughness;    
    float metallic;  
    float sheen;                // [NOT USED]
    float clearcoat_thickness;  // [NOT USED]
    float clearcoat_roughness;  // [NOT USED]
    float anisotropy;           // [NOT USED]
    float anisotropy_rotation;  // [NOT USED]

    int roughness_tex_id;  // [NOT USED]
    int metallic_tex_id;   // [NOT USED]
    int sheen_tex_id;      // [NOT USED]
    int emissive_tex_id;   // [NOT USED]
    int normal_tex_id;     // [NOT USED]
};

struct HoverInfo {
	int hovered_entity_id;
};

struct InstanceInfo {
    uint vertex_offset;
    uint index_offset;
    uint has_normal;
    uint normal_offset;
    uint has_texcoord;
    uint texcoord_offset;
    uint has_tangent;
    uint tangent_offset;
};

struct PointLight {
    float3 position;
    float power;
    float3 color;
    float radius;
};

struct AreaLight {
    float3 position;
    float power;
    float3 color;
    float _pad0;
    float3 u;
    float _pad1;
    float3 v;
    float _pad2;
};

struct SunLight {
    float3 direction;
    float power;
    float3 color;
    float angle; // in degrees
};

struct SceneInfo {
    uint num_point_lights;
    uint num_area_lights;
    uint num_sun_lights;
    uint _pad;
};

RaytracingAccelerationStructure as : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);
StructuredBuffer<Material> materials : register(t0, space3);
ConstantBuffer<HoverInfo> hover_info : register(b0, space4);
RWTexture2D<int> entity_id_output : register(u0, space5);
RWTexture2D<float4> accumulated_color : register(u0, space6);
RWTexture2D<int> accumulated_samples : register(u0, space7);
StructuredBuffer<PointLight> point_lights : register(t0, space8);
StructuredBuffer<AreaLight> area_lights : register(t0, space9);
StructuredBuffer<SunLight> sun_lights : register(t0, space10);
ConstantBuffer<SceneInfo> scene_info : register(b0, space11);
StructuredBuffer<InstanceInfo> instance_infos : register(t0, space12);
StructuredBuffer<float3> vertices : register(t0, space13);
StructuredBuffer<uint> indices : register(t0, space14);
StructuredBuffer<float3> normals : register(t0, space15);
StructuredBuffer<float2> texcoords : register(t0, space16);
StructuredBuffer<float3> tangents : register(t0, space17);
Texture2D<float4> material_images[] : register(t0, space18);
SamplerState material_sampler : register(s0, space19);
Texture2D<float4> g_HDRISkybox : register(t0, space20);
// Random Number Generator
uint tea(uint val0, uint val1) {
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;
    for (uint n = 0; n < 16; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

uint lcg(inout uint prev) {
    uint LCG_A = 1664525u;
    uint LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

float rnd(inout uint prev) {
    return (float(lcg(prev)) / float(0x01000000));
}

static const float PI = 3.14159265359;
static const int MAX_DEPTH = 8;
static const float DIRECT_CLAMP = 11.0;
static const float INDIRECT_CLAMP = 6.0;
static const float ISO_MULTIPLIER = 1.6;
static const float GAMMA = 1.0 / 2.0;

float3 clamp_direct(float3 color) {
    float norm = length(color);
    if (norm > DIRECT_CLAMP) {
        color = color * (DIRECT_CLAMP / norm);
    }
    return color;
}

float3 clamp_indirect(float3 color) {
    float norm = length(color);
    if (norm > INDIRECT_CLAMP) {
        color = color * (INDIRECT_CLAMP / norm);
    }
    return color;
}


struct RayPayload {
	float3 color;
    float3 throughput;
    float3 origin;
    float3 direction;
    float transmittance;
    uint seed;
	bool hit;
	uint instance_id;
    bool is_shadow;
};

float2 sample_disk(inout uint seed, float radius)
{
    float2 a = float2(rnd(seed), rnd(seed)) * 2.0 - 1.0;
    if (a.x == 0.0 && a.y == 0.0) return float2(0.0, 0.0);

    float theta, r;
    if (abs(a.x) > abs(a.y)) {
        r = a.x;
        theta = (PI / 4.0) * (a.y / a.x);
    } else {
        r = a.y;
        theta = (PI / 2.0) - (PI / 4.0) * (a.x / a.y);
    }
    
    return float2(radius * r * cos(theta), radius * r * sin(theta));
}

[shader("raygeneration")] void RayGenMain() {
	uint2 pixel_coords = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    
    // Initialize RNG
    int sample_count = accumulated_samples[pixel_coords];
    uint seed = tea(pixel_coords.y * dims.x + pixel_coords.x, sample_count);

	float2 pixel_center = (float2)pixel_coords + float2(rnd(seed), rnd(seed)); // Jitter for AA
	float2 uv = pixel_center / float2(dims);
	uv.y = 1.0 - uv.y;
	float2 d = uv * 2.0 - 1.0;
    float3 camera_pos_O = mul(camera_info.camera_to_world, float4(0, 0, 0, 1)).xyz;
    float4 target_screen_cs = mul(camera_info.screen_to_camera, float4(d, 1, 1));
    float3 direction_ws = mul(camera_info.camera_to_world, float4(target_screen_cs.xyz, 0)).xyz;
    direction_ws = normalize(direction_ws);
    float3 focal_point_Pf = camera_pos_O + direction_ws * camera_info.focal_distance;
    float2 lens_sample_xy = sample_disk(seed, camera_info.aperture_radius);
    float3 camera_x_axis = mul(camera_info.camera_to_world, float4(1, 0, 0, 0)).xyz;
    float3 camera_y_axis = mul(camera_info.camera_to_world, float4(0, 1, 0, 0)).xyz;
    float3 ray_origin_Pl = camera_pos_O + lens_sample_xy.x * camera_x_axis + lens_sample_xy.y * camera_y_axis;
    float3 ray_direction_Dnew = normalize(focal_point_Pf - ray_origin_Pl);

	float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    
    RayDesc ray;
	ray.Origin = ray_origin_Pl;
	ray.Direction = ray_direction_Dnew;
	ray.TMin = 0.001;
	ray.TMax = 10000.0;

    RayPayload payload;
    payload.seed = seed;
    
    int first_hit_instance_id = -1;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        payload.color = float3(0, 0, 0);
        payload.throughput = float3(1, 1, 1);
        payload.hit = false;
        payload.instance_id = 0;
        payload.is_shadow = false;

        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

        if (depth == 0) {
            first_hit_instance_id = payload.hit ? (int)payload.instance_id : -1;
        }

        if (!payload.hit) {
            // Miss shader returns sky color in payload.color
            radiance += payload.color * throughput;
            break;
        }

        // Hit
        float3 contribution = payload.color * throughput;
        if (depth > 0) {
            contribution = clamp_indirect(contribution);
            // contribution = min(contribution, float3(INDIRECT_CLAMP, INDIRECT_CLAMP, INDIRECT_CLAMP));
        }
        radiance += contribution; // Add direct lighting
        throughput *= payload.throughput;
        
        // Update ray for next bounce
        ray.Origin = payload.origin;
        ray.Direction = payload.direction;
        
        // Russian Roulette
        if (depth > 3) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (rnd(payload.seed) > p) break;
            throughput /= p;
        }
    }

    // Prevent NaN/Inf from corrupting the accumulation buffer
    if (any(isnan(radiance)) || any(isinf(radiance))) {
        radiance = float3(0, 0, 0);
    }

    radiance = pow(radiance, GAMMA);
    radiance = radiance * ISO_MULTIPLIER;

	// Write to immediate output (for camera movement mode)
	output[pixel_coords] = float4(radiance, 1);
	
	// Write entity ID to the ID buffer (only from first hit)
    entity_id_output[pixel_coords] = first_hit_instance_id; 
	
	// Accumulate color for progressive rendering (when camera is stationary)
	float4 prev_color = accumulated_color[pixel_coords];
	
	accumulated_color[pixel_coords] = prev_color + float4(radiance, 1);
	accumulated_samples[pixel_coords] = sample_count + 1;
}

[shader("miss")] void MissMain(inout RayPayload payload) {
    if (payload.is_shadow) {
        payload.hit = false; // Not occluded
        payload.instance_id = 0; // Stop tracing
        return;
    }
    
    float3 W = normalize(WorldRayDirection());
    
    float phi = atan2(W.x, W.z); 
    float theta = acos(W.y);
    
    float u = phi / (2.0 * 3.14159265359) + 0.5;
    float v = theta / 3.14159265359;
    
    float2 uv = float2(u, v);
    
    float3 sky_color = g_HDRISkybox.SampleLevel(material_sampler, uv, 0).rgb;
    
    payload.color = sky_color;
    
    payload.hit = false;
    payload.throughput = float3(0, 0, 0);
}

// PBR Helper Functions
float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float3 FresnelDielectric(float cosThetaI, float eta) {
    float sinThetaT2 = eta * eta * (1.0 - cosThetaI * cosThetaI);
    if (sinThetaT2 > 1.0) return float3(1.0, 1.0, 1.0); // TIR

    float cosThetaT = sqrt(1.0 - sinThetaT2);
    
    float r_parl = ((eta * cosThetaI) - cosThetaT) / ((eta * cosThetaI) + cosThetaT);
    float r_perp = ((cosThetaI) - (eta * cosThetaT)) / ((cosThetaI) + (eta * cosThetaT));
    
    float f = (r_parl * r_parl + r_perp * r_perp) / 2.0;
    return float3(f, f, f);
}

float DistributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return num / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness * roughness;
    float k = a / 2.0;
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 SampleGGX(float2 u, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * u.x;
    float cosTheta = sqrt((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    float3 H;
    H.x = sinTheta * cos(phi);
    H.y = sinTheta * sin(phi);
    H.z = cosTheta;
    
    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

float3 SampleCosineHemisphere(float2 u, float3 N) {
    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    
    float3 L = normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + N * sqrt(max(0.0, 1.0 - u.x)));
    return L;
}

float3 EvalPrincipledBSDF(float3 N, float3 V, float3 L, float3 albedo, float roughness, float metallic, float3 F0, float transmission, float eta) {
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    
    if (NdotV <= 0.0) return float3(0, 0, 0);

    if (NdotL < 0.0) {
        // Transmission
        if (transmission <= 0.0) return float3(0, 0, 0);
        
        float3 H = -normalize(V * eta + L);
        if (dot(H, N) < 0.0) H = -H;

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        
        float VdotH = max(dot(V, H), 0.0);
        float LdotH = max(dot(-L, H), 0.0);
        
        float3 F = FresnelDielectric(VdotH, eta);
        
        float sqrtDenom = (eta * VdotH + LdotH);
        float common = (NDF * G * VdotH * LdotH) / (NdotV * abs(NdotL) * sqrtDenom * sqrtDenom);
        
        return albedo * (1.0 - F) * common * (1.0 - metallic) * transmission;
    } else {
        // Reflection
        float3 H = normalize(V + L);
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
        
        float3 numerator = NDF * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        float3 specular = numerator / denominator;
        
        float3 kS = F;
        float3 kD = (float3(1.0, 1.0, 1.0) - kS) * (1.0 - metallic);
        float3 diffuse = kD * albedo * (1.0 - transmission) / PI;
        
        return diffuse + specular;
    }
}

void SampleIndirect(
    inout uint seed,
    float3 N,
    float3 V,
    float3 world_pos,
    float3 albedo,
    float roughness,
    float metallic,
    float3 F0,
    float transmission,
    float eta,
    out float3 L_indirect,
    out float3 throughput_weight,
    out float3 next_origin
) {
    throughput_weight = float3(0, 0, 0);
    next_origin = world_pos + N * 0.001;
    
    float F_d = FresnelDielectric(max(dot(N, V), 0.0), eta).x;
    
    float w_spec_refl = metallic + (1.0 - metallic) * F_d;
    float w_spec_trans = (1.0 - metallic) * (1.0 - F_d) * transmission;
    float w_diffuse = (1.0 - metallic) * (1.0 - F_d) * (1.0 - transmission);
    
    float w_sum = w_spec_refl + w_spec_trans + w_diffuse;
    if (w_sum < 0.0001) w_sum = 1.0;
    
    float u_lobe = rnd(seed) * w_sum;
    
    if (u_lobe < w_spec_refl) {
        // Sample Reflection
        float2 u = float2(rnd(seed), rnd(seed));
        float3 H = SampleGGX(u, N, roughness);
        L_indirect = reflect(-V, H);
        
        float NdotL_indirect = max(dot(N, L_indirect), 0.0);
        if (NdotL_indirect > 0.0) {
             float3 bsdf = EvalPrincipledBSDF(N, V, L_indirect, albedo, roughness, metallic, F0, transmission, eta);
             
             float3 H_indirect = normalize(V + L_indirect);
             float NdotH = max(dot(N, H_indirect), 0.0);
             float HdotV = max(dot(H_indirect, V), 0.0);
             float NDF = DistributionGGX(N, H_indirect, roughness);
             float pdf_spec = NDF * NdotH / (4.0 * HdotV + 0.0001);
             
             float pdf = (w_spec_refl / w_sum) * pdf_spec;
             if (pdf > 0.001) {
                 throughput_weight = bsdf * NdotL_indirect / pdf;
             }
        }
        next_origin = world_pos + N * 0.001;
    } else if (u_lobe < w_spec_refl + w_spec_trans) {
        // Sample Transmission
        float2 u = float2(rnd(seed), rnd(seed));
        float3 H = SampleGGX(u, N, roughness);
        L_indirect = refract(-V, H, eta);
        
        if (length(L_indirect) > 0.0) {
             float3 H_indirect = -normalize(V * eta + L_indirect);
             if (dot(H_indirect, N) < 0) H_indirect = -H_indirect;
             
             float VdotH = abs(dot(V, H_indirect));
             float NdotH = abs(dot(N, H_indirect));
             float LdotH = abs(dot(L_indirect, H_indirect));
             
             float NDF = DistributionGGX(N, H_indirect, roughness);
             
             // PDF for transmission
             float sqrtDenom = (eta * VdotH + LdotH);
             float pdf_trans = NDF * NdotH * LdotH / (sqrtDenom * sqrtDenom);
             
             float pdf = (w_spec_trans / w_sum) * pdf_trans;
             
             if (pdf > 0.001) {
                 float3 bsdf = EvalPrincipledBSDF(N, V, L_indirect, albedo, roughness, metallic, F0, transmission, eta);
                 throughput_weight = bsdf * abs(dot(N, L_indirect)) / pdf;
             }
             
             next_origin = world_pos + L_indirect * 0.001;
        } else {
             throughput_weight = float3(0, 0, 0);
        }
    } else {
        // Sample Diffuse
        float2 u = float2(rnd(seed), rnd(seed));
        L_indirect = SampleCosineHemisphere(u, N);
        
        float NdotL_indirect = max(dot(N, L_indirect), 0.0);
        if (NdotL_indirect > 0.0) {
             float3 bsdf = EvalPrincipledBSDF(N, V, L_indirect, albedo, roughness, metallic, F0, transmission, eta);
             float pdf_diff = NdotL_indirect / PI;
             float pdf = (w_diffuse / w_sum) * pdf_diff;
             
             if (pdf > 0.001) {
                 throughput_weight = bsdf * NdotL_indirect / pdf;
             }
        }
        next_origin = world_pos + N * 0.001;
    }
}

float TraceShadowRay(RaytracingAccelerationStructure as, float3 origin, float3 direction, float maxDistance, inout uint seed) {
    float3 currentOrigin = origin;
    float currentTMax = maxDistance;
    float transmittance = 1.0;
    
    while(true) {
        RayDesc shadowRay;
        shadowRay.Origin = currentOrigin;
        shadowRay.Direction = direction;
        shadowRay.TMin = 0.001;
        shadowRay.TMax = currentTMax;

        RayPayload shadowPayload;
        shadowPayload.is_shadow = true;
        shadowPayload.hit = false; 
        shadowPayload.seed = seed;
        shadowPayload.instance_id = 0;
        shadowPayload.transmittance = transmittance;
        
        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, shadowRay, shadowPayload);
        
        if (shadowPayload.hit) {
            return 0.0; // Occluded
        }
        
        if (shadowPayload.instance_id == 1) {
            // Transparent hit, continue
            float3 prevOrigin = currentOrigin;
            currentOrigin = shadowPayload.origin;
            transmittance = shadowPayload.transmittance;
            
            // Update TMax
            float step = length(currentOrigin - prevOrigin);
            currentTMax -= step;
            
            if (currentTMax <= 0.001) return transmittance; // Reached target
            
            seed = shadowPayload.seed; 
            continue;
        } else {
            return transmittance; // Missed everything
        }
    }
    return transmittance;
}

[shader("closesthit")] void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
	// Get material index from instance
	uint instance_id = InstanceID();
	
	// Load material
	Material mat = materials[instance_id];
    
    // Fetch Geometry
    InstanceInfo info = instance_infos[instance_id];
    uint prim_idx = PrimitiveIndex();
    
    uint3 idx = uint3(
        indices[info.index_offset + prim_idx * 3 + 0],
        indices[info.index_offset + prim_idx * 3 + 1],
        indices[info.index_offset + prim_idx * 3 + 2]
    );
    
    float3 v0 = vertices[info.vertex_offset + idx.x];
    float3 v1 = vertices[info.vertex_offset + idx.y];
    float3 v2 = vertices[info.vertex_offset + idx.z];
    
    float3 n0, n1, n2;
    if (info.has_normal) {
        n0 = normals[info.normal_offset + idx.x];
        n1 = normals[info.normal_offset + idx.y];
        n2 = normals[info.normal_offset + idx.z];
    } else {
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        n0 = n1 = n2 = normalize(cross(e1, e2));
    }
    
    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    float3 world_normal = normalize(mul(ObjectToWorld3x4(), float4(n0 * bary.x + n1 * bary.y + n2 * bary.z, 0)).xyz);
    float3 world_pos = mul(ObjectToWorld3x4(), float4(v0 * bary.x + v1 * bary.y + v2 * bary.z, 1)).xyz;

    if(info.has_texcoord && mat.diffuse_tex_id >= 0) {
        float2 uv0=texcoords[info.texcoord_offset + idx.x];
        float2 uv1=texcoords[info.texcoord_offset + idx.y];
        float2 uv2=texcoords[info.texcoord_offset + idx.z];
        float2 interp_uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
        interp_uv = frac(interp_uv);
        interp_uv.y = 1-interp_uv.y;

        float4 tex = material_images[mat.diffuse_tex_id].SampleLevel(material_sampler, interp_uv, 0.0f);
        tex.xyz = pow(tex.xyz, float3(2.2, 2.2, 2.2));
        mat.diffuse = tex.xyz;
    }
    
    if (payload.is_shadow) { // this is wrong when ior is involved, but just ignore that for now. no one will notice.
        if (mat.dissolve < 1.0) {
            payload.hit = false;
            payload.instance_id = 1; // Signal continue
            payload.origin = world_pos + WorldRayDirection() * 0.001;
            payload.direction = WorldRayDirection();
            payload.transmittance *= (1 - mat.dissolve);
            return;
        } else {
            payload.hit = true; // Occluded
            payload.instance_id = 0; // Signal stop (opaque)
            return;
        }
    }

	payload.hit = true;
	payload.instance_id = instance_id;
    
    float3 V = -normalize(WorldRayDirection());
    float3 N = normalize(world_normal);
    bool front_face = dot(N, V) > 0.0;
    if (!front_face) N = -N;

    // Material properties
    float3 albedo = mat.diffuse;
    float roughness = max(mat.roughness, 0.001); 
    float metallic = mat.metallic;
    float3 emission = mat.emission;
    float transmission = 1.0 - mat.dissolve;
    float ior = mat.ior;
    // if (ior < 1.0) ior = 1.5;
    
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // IOR handling
    float eta = front_face ? (1.0 / ior) : ior;

    // Direct Lighting
    float3 Lo = float3(0, 0, 0);

    // Point Lights
    for (uint i = 0; i < scene_info.num_point_lights; i++) {
        PointLight light = point_lights[i];
        
        float3 lightPos = light.position;
        if (light.radius > 0.0) {
            float r1 = rnd(payload.seed);
            float r2 = rnd(payload.seed);
            float z = 1.0 - 2.0 * r1;
            float r = sqrt(max(0.0, 1.0 - z * z));
            float phi = 2.0 * PI * r2;
            float3 randomDir = float3(r * cos(phi), r * sin(phi), z);
            lightPos += randomDir * light.radius;
        }

        float3 L = normalize(lightPos - world_pos);
        float dist = length(lightPos - world_pos);
        float attenuation = 1.0 / (dist * dist);
        float3 radiance = light.color * light.power * attenuation / (4.0 * PI);

        float shadow_hit = TraceShadowRay(as, world_pos + N * 0.001, L, dist - 0.001, payload.seed);

        if (true) {
            float NdotL = abs(dot(N, L));
            float3 bsdf = EvalPrincipledBSDF(N, V, L, albedo, roughness, metallic, F0, transmission, eta);
            Lo += shadow_hit*clamp_direct(bsdf * radiance * NdotL);
        }
    }

    // Area Lights
    for (uint k = 0; k < scene_info.num_area_lights; k++) {
        AreaLight light = area_lights[k];
        float r1 = rnd(payload.seed);
        float r2 = rnd(payload.seed);
        float3 lightPos = light.position + light.u * r1 + light.v * r2;
        
        float3 L = normalize(lightPos - world_pos);
        float dist = length(lightPos - world_pos);
        float attenuation = 1.0 / (dist * dist);
        
        float3 lightNormal = normalize(cross(light.u, light.v));
        float NdotL_light = max(dot(-L, lightNormal), 0.0);
        
        float power_area = length(cross(light.u, light.v));
        float3 radiance = (light.color * light.power / power_area / PI) * NdotL_light * attenuation;

        float shadow_hit = TraceShadowRay(as, world_pos + N * 0.001, L, dist - 0.001, payload.seed);

        if (true) {
            float NdotL = abs(dot(N, L));
            float3 bsdf = EvalPrincipledBSDF(N, V, L, albedo, roughness, metallic, F0, transmission, eta);
            Lo += shadow_hit * clamp_direct(bsdf * radiance * NdotL);
        }
    }

    // Sun Lights
    for (uint j = 0; j < scene_info.num_sun_lights; j++) {
        SunLight light = sun_lights[j];
        float3 L = normalize(-light.direction);
        
        if (light.angle > 0.0) {
            float r1 = rnd(payload.seed);
            float r2 = rnd(payload.seed);
            float radius = tan(radians(light.angle * 0.5));
            float r = sqrt(r1) * radius;
            float phi = 2.0 * PI * r2;
            
            float3 up = abs(L.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
            float3 tangent = normalize(cross(up, L));
            float3 bitangent = cross(L, tangent);
            
            L = normalize(L + tangent * (r * cos(phi)) + bitangent * (r * sin(phi)));
        }

        float3 radiance = light.color * light.power;
        
        float shadow_hit = TraceShadowRay(as, world_pos + N * 0.001, L, 10000.0, payload.seed);
        
        if (true) {
            float NdotL = abs(dot(N, L));
            float3 bsdf = EvalPrincipledBSDF(N, V, L, albedo, roughness, metallic, F0, transmission, eta);
            Lo += shadow_hit * clamp_direct(bsdf * radiance * NdotL);
        }
    }

    payload.color = emission + Lo;

    // Indirect Lighting (Next Bounce)
    float3 L_indirect;
    float3 throughput_weight;
    float3 next_origin;

    SampleIndirect(payload.seed, N, V, world_pos, albedo, roughness, metallic, F0, transmission, eta, L_indirect, throughput_weight, next_origin);
    
    payload.throughput = throughput_weight;
    payload.direction = L_indirect;
    payload.origin = next_origin;
}
