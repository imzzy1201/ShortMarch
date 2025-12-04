
struct CameraInfo {
	float4x4 screen_to_camera;
	float4x4 camera_to_world;
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
static const int MAX_DEPTH = 10;
static const float DIRECT_CLAMP = 0.550;
static const float INDIRECT_CLAMP = 0.300;
static const float SENSITIVITY = 2.0;


struct RayPayload {
	float3 color;
    float3 throughput;
    float3 origin;
    float3 direction;
    uint seed;
	bool hit;
	uint instance_id;
    bool is_shadow;
};

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
	float4 origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
	float4 target = mul(camera_info.screen_to_camera, float4(d, 1, 1));
	float4 direction = mul(camera_info.camera_to_world, float4(target.xyz, 0));

	float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    
    RayDesc ray;
	ray.Origin = origin.xyz;
	ray.Direction = normalize(direction.xyz);
	ray.TMin = 0.001;
	ray.TMax = 10000.0;

    RayPayload payload;
    payload.seed = seed;
    
    int first_hit_instance_id = -1;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        payload.color = float3(0, 0, 0);
        payload.throughput = float3(0, 0, 0);
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
            float contribution_norm = length(contribution);
            if (contribution_norm > INDIRECT_CLAMP) {
                contribution = contribution * (INDIRECT_CLAMP / contribution_norm);
            }
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

    radiance = radiance * SENSITIVITY;

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
    
	// Sky gradient
	float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
	payload.color = lerp(float3(1.0, 1.0, 1.0), float3(0.5, 0.7, 1.0), t);
	payload.hit = false;
    payload.throughput = float3(0, 0, 0);
}

// PBR Helper Functions
float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
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

float3 EvalPrincipledBSDF(float3 N, float3 V, float3 L, float3 albedo, float roughness, float metallic, float3 F0) {
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    
    if (NdotL <= 0.0 || NdotV <= 0.0) return float3(0, 0, 0);

    float3 H = normalize(V + L);
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    
    float3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    float3 specular = numerator / denominator;
    
    float3 kS = F;
    float3 kD = (float3(1.0, 1.0, 1.0) - kS) * (1.0 - metallic);
    float3 diffuse = kD * albedo / PI;
    
    return diffuse + specular;
}

bool TraceShadowRay(RaytracingAccelerationStructure as, float3 origin, float3 direction, float maxDistance, inout uint seed) {
    float3 currentOrigin = origin;
    float currentTMax = maxDistance;
    
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
        
        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, shadowRay, shadowPayload);
        
        if (shadowPayload.hit) {
            return true; // Occluded
        }
        
        if (shadowPayload.instance_id == 1) {
            // Transparent hit, continue
            float3 prevOrigin = currentOrigin;
            currentOrigin = shadowPayload.origin;
            
            // Update TMax
            float step = length(currentOrigin - prevOrigin);
            currentTMax -= step;
            
            if (currentTMax <= 0.001) return false; // Reached target
            
            seed = shadowPayload.seed; 
            continue;
        } else {
            return false; // Missed everything
        }
    }
    return false;
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
    
    // Transparency Check
    bool is_transparent = false;
    if (mat.dissolve < 1.0) {
        if (rnd(payload.seed) >= mat.dissolve) {
            is_transparent = true;
        }
    }

    if (payload.is_shadow) {
        if (is_transparent) {
            payload.hit = false;
            payload.instance_id = 1; // Signal continue
            payload.origin = world_pos + WorldRayDirection() * 0.001;
            payload.direction = WorldRayDirection();
            return;
        } else {
            payload.hit = true; // Occluded
            payload.instance_id = 0; // Signal stop (opaque)
            return;
        }
    }

	payload.hit = true;
	payload.instance_id = instance_id;
    
    // Transparency (Regular Rays)
    if (is_transparent) {
        payload.color = float3(0, 0, 0);
        payload.throughput = float3(1, 1, 1);
        payload.origin = world_pos + WorldRayDirection() * 0.001;
        payload.direction = WorldRayDirection();
        return;
    }
    
    float3 V = -normalize(WorldRayDirection());
    float3 N = normalize(world_normal);
    if (dot(N, V) < 0) N = -N;

    // Material properties
    float3 albedo = mat.diffuse;
    float roughness = max(mat.roughness, 0.001); // Clamp roughness to prevent NaN in GGX
    float metallic = mat.metallic;
    float3 emission = mat.emission;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

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

        bool shadow_hit = TraceShadowRay(as, world_pos + N * 0.001, L, dist - 0.001, payload.seed);

        if (!shadow_hit) {
            float NdotL = max(dot(N, L), 0.0);
            float3 bsdf = EvalPrincipledBSDF(N, V, L, albedo, roughness, metallic, F0);
            Lo += bsdf * radiance * NdotL;
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
        
        float3 radiance = (light.color * light.power / PI) * NdotL_light * attenuation;

        bool shadow_hit = TraceShadowRay(as, world_pos + N * 0.001, L, dist - 0.001, payload.seed);

        if (!shadow_hit) {
            float NdotL = max(dot(N, L), 0.0);
            float3 bsdf = EvalPrincipledBSDF(N, V, L, albedo, roughness, metallic, F0);
            Lo += bsdf * radiance * NdotL;
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
        
        bool shadow_hit = TraceShadowRay(as, world_pos + N * 0.001, L, 10000.0, payload.seed);
        
        if (!shadow_hit) {
            float NdotL = max(dot(N, L), 0.0);
            float3 bsdf = EvalPrincipledBSDF(N, V, L, albedo, roughness, metallic, F0);
            Lo += bsdf * radiance * NdotL;
        }
    }

    float Lo_norm = length(Lo);
    if (Lo_norm > DIRECT_CLAMP) {
        Lo = Lo * (DIRECT_CLAMP / Lo_norm);
    }
    // Lo = min(Lo, float3(DIRECT_CLAMP, DIRECT_CLAMP, DIRECT_CLAMP));

    payload.color = emission + Lo;

    // Indirect Lighting (Next Bounce)
    float2 u = float2(rnd(payload.seed), rnd(payload.seed));
    
    // Probability to sample specular
    float probSpecular = lerp(0.5, 1.0, metallic);
    float3 L_indirect;
    
    if (rnd(payload.seed) < probSpecular) {
        float3 H = SampleGGX(u, N, roughness);
        L_indirect = reflect(-V, H);
    } else {
        L_indirect = SampleCosineHemisphere(u, N);
    }
    
    float NdotL_indirect = max(dot(N, L_indirect), 0.0);
    
    if (NdotL_indirect > 0.0) {
        float3 bsdf = EvalPrincipledBSDF(N, V, L_indirect, albedo, roughness, metallic, F0);
        
        float3 H_indirect = normalize(V + L_indirect);
        float NdotH = max(dot(N, H_indirect), 0.0);
        float HdotV = max(dot(H_indirect, V), 0.0);
        
        float NDF = DistributionGGX(N, H_indirect, roughness);
        
        float pdf_spec = NDF * NdotH / (4.0 * HdotV + 0.0001);
        float pdf_diff = NdotL_indirect / PI;
        
        float pdf = probSpecular * pdf_spec + (1.0 - probSpecular) * pdf_diff;
        
        if (pdf > 0.001) {
            payload.throughput = bsdf * NdotL_indirect / pdf;
        } else {
            payload.throughput = float3(0, 0, 0);
        }
    } else {
        payload.throughput = float3(0, 0, 0);
    }
    
    payload.origin = world_pos + N * 0.001;
    payload.direction = L_indirect;
}
