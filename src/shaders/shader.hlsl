
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
    float _pad;
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
    float _pad;
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
        radiance += payload.color * throughput; // Add direct lighting
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
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
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

float3 SampleCosineHemisphere(float2 u, float3 N) {
    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    
    float3 L = normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + N * sqrt(max(0.0, 1.0 - u.x)));
    return L;
}

[shader("closesthit")] void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    if (payload.is_shadow) {
        payload.hit = true; // Occluded
        return;
    }

	payload.hit = true;
	
	// Get material index from instance
	uint instance_id = InstanceID();
	payload.instance_id = instance_id;
	
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
    
    // Transparency
    if (mat.dissolve < 1.0) {
        if (rnd(payload.seed) >= mat.dissolve) {
            payload.color = float3(0, 0, 0);
            payload.throughput = float3(1, 1, 1);
            payload.origin = world_pos + WorldRayDirection() * 0.001;
            payload.direction = WorldRayDirection();
            return;
        }
    }
    
    float3 V = -normalize(WorldRayDirection());
    float3 N = normalize(world_normal);
    if (dot(N, V) < 0) N = -N;

    // Material properties
    float3 albedo = mat.diffuse;
    float roughness = mat.roughness;
    float metallic = mat.metallic;
    float3 emission = mat.emission;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // Direct Lighting
    float3 Lo = float3(0, 0, 0);

    // Point Lights
    for (uint i = 0; i < scene_info.num_point_lights; i++) {
        PointLight light = point_lights[i];
        float3 L = normalize(light.position - world_pos);
        float dist = length(light.position - world_pos);
        float attenuation = 1.0 / (dist * dist);
        float3 radiance = light.color * light.power * attenuation / (4.0 * PI);

        RayDesc shadowRay;
        shadowRay.Origin = world_pos + N * 0.001;
        shadowRay.Direction = L;
        shadowRay.TMin = 0.001;
        shadowRay.TMax = dist - 0.001;

        RayPayload shadowPayload;
        shadowPayload.is_shadow = true;
        shadowPayload.hit = true; 
        
        TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, shadowRay, shadowPayload);

        if (!shadowPayload.hit) {
            float3 H = normalize(V + L);
            float NdotL = max(dot(N, L), 0.0);
            
            float NDF = DistributionGGX(N, H, roughness);
            float G = GeometrySmith(N, V, L, roughness);
            float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
            
            float3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
            float3 specular = numerator / denominator;
            
            float3 kS = F;
            float3 kD = float3(1.0, 1.0, 1.0) - kS;
            kD *= 1.0 - metallic;
            
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
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

        RayDesc shadowRay;
        shadowRay.Origin = world_pos + N * 0.001;
        shadowRay.Direction = L;
        shadowRay.TMin = 0.001;
        shadowRay.TMax = dist - 0.001;

        RayPayload shadowPayload;
        shadowPayload.is_shadow = true;
        shadowPayload.hit = true; 
        
        TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, shadowRay, shadowPayload);

        if (!shadowPayload.hit) {
            float3 H = normalize(V + L);
            float NdotL = max(dot(N, L), 0.0);
            
            float NDF = DistributionGGX(N, H, roughness);
            float G = GeometrySmith(N, V, L, roughness);
            float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
            
            float3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
            float3 specular = numerator / denominator;
            
            float3 kS = F;
            float3 kD = float3(1.0, 1.0, 1.0) - kS;
            kD *= 1.0 - metallic;
            
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
        }
    }

    // Sun Lights
    for (uint j = 0; j < scene_info.num_sun_lights; j++) {
        SunLight light = sun_lights[j];
        float3 L = normalize(-light.direction);
        float3 radiance = light.color * light.power;
        
        RayDesc shadowRay;
        shadowRay.Origin = world_pos + N * 0.001;
        shadowRay.Direction = L;
        shadowRay.TMin = 0.001;
        shadowRay.TMax = 10000.0;
        
        RayPayload shadowPayload;
        shadowPayload.is_shadow = true;
        shadowPayload.hit = true;
        
        TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, shadowRay, shadowPayload);
        
        if (!shadowPayload.hit) {
            float3 H = normalize(V + L);
            float NdotL = max(dot(N, L), 0.0);
            
            float NDF = DistributionGGX(N, H, roughness);
            float G = GeometrySmith(N, V, L, roughness);
            float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
            
            float3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
            float3 specular = numerator / denominator;
            
            float3 kS = F;
            float3 kD = float3(1.0, 1.0, 1.0) - kS;
            kD *= 1.0 - metallic;
            
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
        }
    }

    payload.color = emission + Lo;

    // Indirect Lighting (Next Bounce)
    float2 u = float2(rnd(payload.seed), rnd(payload.seed));
    float3 L_indirect = SampleCosineHemisphere(u, N);
    float3 H_indirect = normalize(V + L_indirect);
    float NdotL_indirect = max(dot(N, L_indirect), 0.0);
    
    if (NdotL_indirect > 0.0) {
        float NDF = DistributionGGX(N, H_indirect, roughness);
        float G = GeometrySmith(N, V, L_indirect, roughness);
        float3 F = FresnelSchlick(max(dot(H_indirect, V), 0.0), F0);
        
        float3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL_indirect + 0.0001;
        float3 specular = numerator / denominator;
        
        float3 kS = F;
        float3 kD = float3(1.0, 1.0, 1.0) - kS;
        kD *= 1.0 - metallic;
        
        float pdf = NdotL_indirect / PI;
        
        payload.throughput = (kD * albedo / PI + specular) * NdotL_indirect / max(pdf, 0.001);
    } else {
        payload.throughput = float3(0, 0, 0);
    }
    
    payload.origin = world_pos + N * 0.001;
    payload.direction = L_indirect;
}
