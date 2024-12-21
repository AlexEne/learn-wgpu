#version 450
#extension GL_GOOGLE_include_directive : enable

#include "types.glsl"

layout(location = 0) in vec2 v_tex_coords;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_world_position;

layout(set = 0, binding = 0) uniform texture2D t_texture;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

layout(set = 2, binding = 0) uniform Light {
    vec3 position;
    vec3 color;
} light;

layout(location = 0) out vec4 out_color;

void main()
{
    vec3 light_dir = normalize(light.position - v_world_position);
    float diffuse_strength = max(dot(v_normal, light_dir), 0.0);
    diffuse_strength = min(diffuse_strength, 0.9);
    vec3 diffuse_color = light.color * diffuse_strength;
    
    vec3 light_ambient_strength = vec3(0.1);
    vec3 ambient_color = light_ambient_strength * light.color;
    
    vec4 object_color = texture(sampler2D(t_texture, s_diffuse), v_tex_coords);
    
    out_color = vec4((ambient_color + diffuse_color) * object_color.rgb, object_color.a);
}
