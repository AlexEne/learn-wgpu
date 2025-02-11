#version 450
#extension GL_GOOGLE_include_directive : enable

#include "types.glsl"

layout(location = 0) in vec2 v_tex_coords;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_world_position;
layout(location = 3) in vec3 v_camera_pos;

layout(set = 0, binding = 1) uniform texture2D t_texture;
layout(set = 0, binding = 2) uniform sampler s_diffuse;

layout(set = 0, binding = 3) uniform texture2D t_metalic_roughness;
layout(set = 0, binding = 4) uniform sampler s_metalic_roughness;

layout(set = 2, binding = 0) uniform Light {
    vec3 position;
    vec3 color;
} light;

layout(set = 3, binding = 0) uniform PBR {
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
} pbr;

layout(location = 0) out vec4 out_color;


vec4 linear_to_srgb(vec4 color) {
    return vec4(pow(color.rgb, vec3(2.2)), color.a);
}

// void main()
// {
//     vec3 light_dir = normalize(light.position - v_world_position);
//     float diffuse_strength = max(dot(v_normal, light_dir), 0.0);
//     vec3 diffuse_color = light.color * diffuse_strength;
    
//     vec3 light_ambient_strength = vec3(0.1);
//     vec3 ambient_color = light_ambient_strength * light.color;
    
//     vec4 object_color = texture(sampler2D(t_texture, s_diffuse), v_tex_coords);
    
//     out_color = vec4((ambient_color + diffuse_color) * object_color.rgb, object_color.a);
// }


float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    vec3 N = normalize(v_normal);
    vec3 V = normalize(v_camera_pos - v_world_position);
    vec3 L = normalize(light.position - v_world_position);
    float NdotV1 = dot(N, V);
    float NdotL = max(dot(N, L), 0.0);     
    float NdotV = max(NdotV1, 0.0);
    
    vec3 albedo = texture(sampler2D(t_texture, s_diffuse), v_tex_coords).rgb * pbr.base_color_factor.rgb;
    albedo = pow(albedo, vec3(2.2));
    
    vec4 metallic_roughness = texture(sampler2D(t_metalic_roughness, s_metalic_roughness), v_tex_coords);
    float metallic = metallic_roughness.b; //* pbr.metallic_factor;
    float roughness = metallic_roughness.g;// * pbr.roughness_factor;

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // calculate per-light radiance
    vec3 H = normalize(V + L);
    float distance = length(light.position - v_world_position);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = light.color.rgb * attenuation;        
    
    // cook-torrance brdf
    float NDF = DistributionGGX(N, H, roughness);        
    float G = GeometrySmith(NdotV, NdotL, roughness);      
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);       
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;	  
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;  
        
    // add to outgoing radiance Lo
               
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL; 
  
    vec3 ambient = vec3(0.03) * albedo * 1.0; // * ao
    vec3 color = ambient + Lo;
    
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));  

    out_color = vec4(color, 1.0);
    // out_color = vec4(vec3(v_world_position.z), 1.0);
}

