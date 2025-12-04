
struct CameraInfo {
	float4x4 screen_to_camera;
	float4x4 camera_to_world;
};

struct Material {
    float3 base_color;  // DEPRECATED
    float roughness;    // DEPRECATED
    float metallic;     // DEPRECATED

    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 transmittance;
    float3 emission;
    float shininess;
    float ior;       // index of refraction
    float dissolve;  // 1 == opaque; 0 == fully transparent
    int illum;       // illumination model

    int ambient_tex_id;
    int diffuse_tex_id;
    int specular_tex_id;
    int specular_highlight_tex_id;
    int bump_tex_id;
    int displacement_tex_id;
    int alpha_tex_id;
    int reflection_tex_id;
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
    float intensity;
    float3 color;
    float _pad;
};

struct AreaLight {
    float3 position;
    float intensity;
    float3 color;
    float _pad0;
    float3 u;
    float _pad1;
    float3 v;
    float _pad2;
};

struct SunLight {
    float3 direction;
    float intensity;
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
static const int MAX_DEPTH = 8;


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
        if (depth > 2) {
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

float3 SampleHemisphereCosine(float3 n, inout uint seed) {
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float phi = 2.0 * PI * r1;
    float sqrtr2 = sqrt(r2);
    float3 local_dir = float3(sqrtr2 * cos(phi), sqrtr2 * sin(phi), sqrt(1.0 - r2));
    
    // Build orthonormal basis
    float3 up = abs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    
    return normalize(tangent * local_dir.x + bitangent * local_dir.y + n * local_dir.z);
}

float3 EvalBRDF(Material mat, float3 N, float3 L, float3 V) {
    float3 result = float3(0, 0, 0);
    
    // Diffuse
    if (mat.illum >= 1) {
        result += mat.diffuse / PI;
    }
    
    // Specular (Blinn-Phong)
    if (mat.illum >= 2) {
        float3 H = normalize(L + V);
        float ndoth = max(0.0, dot(N, H));
        if (ndoth > 0.0) {
            float spec_power = max(1.0, mat.shininess);
            float norm = (spec_power + 2.0) / (8.0 * PI);
            float spec = pow(ndoth, spec_power) * norm;
            result += mat.specular * spec;
        }
    }
    return result;
}

void SampleBRDF(Material mat, float3 N, float3 V, inout uint seed, out float3 next_dir, out float3 throughput) {
    // Default to absorption
    next_dir = float3(0, 1, 0);
    throughput = float3(0, 0, 0);
    
    if (mat.illum == 5) {
        // Perfect Reflection
        next_dir = reflect(-V, N);
        throughput = mat.specular; 
        return;
    }
    
    // For Illum 2 (Phong), mix diffuse and specular
    float lum_diff = dot(mat.diffuse, float3(0.2126, 0.7152, 0.0722));
    float lum_spec = dot(mat.specular, float3(0.2126, 0.7152, 0.0722));
    
    // If illum < 2, force diffuse
    float prob_spec = 0.0;
    if (mat.illum >= 2 && (lum_diff + lum_spec) > 1e-6) {
        prob_spec = lum_spec / (lum_diff + lum_spec);
    }
    
    if (rnd(seed) < prob_spec) {
        // Sample Specular (Blinn-Phong)
        float spec_power = max(1.0, mat.shininess);
        float alpha = acos(pow(rnd(seed), 1.0 / (spec_power + 1.0)));
        float phi = 2.0 * PI * rnd(seed);
        
        float sin_alpha = sin(alpha);
        float cos_alpha = cos(alpha);
        
        float3 H_local = float3(sin_alpha * cos(phi), sin_alpha * sin(phi), cos_alpha);
        
        // Transform H to world space
        float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
        float3 tangent = normalize(cross(up, N));
        float3 bitangent = cross(N, tangent);
        float3 H = normalize(tangent * H_local.x + bitangent * H_local.y + N * H_local.z);
        
        next_dir = reflect(-V, H);
        
        if (dot(next_dir, N) <= 0.0) {
            throughput = float3(0, 0, 0);
            return;
        }
        
        throughput = mat.specular / prob_spec;
    } else {
        // Sample Diffuse
        next_dir = SampleHemisphereCosine(N, seed);
        throughput = mat.diffuse / (1.0 - prob_spec);
    }
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
    
    // Flip normal if backfacing
    if (dot(world_normal, WorldRayDirection()) > 0) {
        world_normal = -world_normal;
    }
    
    // Direct Lighting (Next Event Estimation)
    float3 direct_light = float3(0, 0, 0);
    
    // Point Lights
    for (uint i = 0; i < scene_info.num_point_lights; i++) {
        PointLight light = point_lights[i];
        float3 L = light.position - world_pos;
        float dist = length(L);
        L = normalize(L);
        
        float ndotl = max(0.0, dot(world_normal, L));
        if (ndotl > 0) {
            // Shadow ray
            RayDesc shadow_ray;
            shadow_ray.Origin = world_pos + world_normal * 0.001;
            shadow_ray.Direction = L;
            shadow_ray.TMin = 0.001;
            shadow_ray.TMax = dist - 0.001;
            
            RayPayload shadow_payload;
            shadow_payload.is_shadow = true;
            shadow_payload.hit = true; // Assume occluded
            
            TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, shadow_ray, shadow_payload);
            
            if (!shadow_payload.hit) {
                // Visible
                float atten = light.intensity / (dist * dist);
                float3 brdf = EvalBRDF(mat, world_normal, L, -WorldRayDirection());
                direct_light += light.color * atten * ndotl * brdf;
            }
        }
    }
    
    // Sun Lights
    for (uint j = 0; j < scene_info.num_sun_lights; j++) {
        SunLight light = sun_lights[j];
        float3 L = -normalize(light.direction);
        
        float ndotl = max(0.0, dot(world_normal, L));
        if (ndotl > 0) {
            RayDesc shadow_ray;
            shadow_ray.Origin = world_pos + world_normal * 0.001;
            shadow_ray.Direction = L;
            shadow_ray.TMin = 0.001;
            shadow_ray.TMax = 10000.0;
            
            RayPayload shadow_payload;
            shadow_payload.is_shadow = true;
            shadow_payload.hit = true;
            
            TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, shadow_ray, shadow_payload);
            
            if (!shadow_payload.hit) {
                float3 brdf = EvalBRDF(mat, world_normal, L, -WorldRayDirection());
                direct_light += light.color * light.intensity * ndotl * brdf;
            }
        }
    }

    // Area Lights
    for (uint k = 0; k < scene_info.num_area_lights; k++) {
        AreaLight light = area_lights[k];
        
        // Sample point on light
        float r1 = rnd(payload.seed);
        float r2 = rnd(payload.seed);
        float3 light_pos = light.position + light.u * r1 + light.v * r2;
        
        float3 L_vec = light_pos - world_pos;
        float dist_sq = dot(L_vec, L_vec);
        float dist = sqrt(dist_sq);
        float3 L = L_vec / dist;
        
        float ndotl = max(0.0, dot(world_normal, L));
        
        if (ndotl > 0) {
            float3 light_normal = normalize(cross(light.u, light.v));
            float ldotn = max(0.0, dot(-L, light_normal));
            
            if (ldotn > 0) {
                RayDesc shadow_ray;
                shadow_ray.Origin = world_pos + world_normal * 0.001;
                shadow_ray.Direction = L;
                shadow_ray.TMin = 0.001;
                shadow_ray.TMax = dist - 0.001;
                
                RayPayload shadow_payload;
                shadow_payload.is_shadow = true;
                shadow_payload.hit = true;
                
                TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 1, 0, shadow_ray, shadow_payload);
                
                if (!shadow_payload.hit) {
                    float3 brdf = EvalBRDF(mat, world_normal, L, -WorldRayDirection());
                    float area = length(cross(light.u, light.v));
                    direct_light += light.color * light.intensity * brdf * ndotl * ldotn * area / dist_sq;
                }
            }
        }
    }
    
    payload.color = direct_light + mat.emission;
    
    // Indirect Bounce
    float3 next_dir;
    float3 throughput_val;
    
    SampleBRDF(mat, world_normal, -WorldRayDirection(), payload.seed, next_dir, throughput_val);
    
    payload.throughput = throughput_val;
    payload.direction = next_dir;
    payload.origin = world_pos + world_normal * 0.001;
}
