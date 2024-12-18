struct VertexInput
{
    float3 position : POSITION;
    float2 tex_coords: TEXCOORD0;
};

struct VertexOutput
{
    float4 clip_position : SV_POSITION;
    [[vk::location(0)]] float2 tex_coords: TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D t_diffuse;
[[vk::binding(0, 1)]] SamplerState s_diffuse;

VertexOutput vs_main(VertexInput input)
{
    VertexOutput output;

    output.clip_position = float4(input.position, 1.0);
    output.tex_coords = input.tex_coords;
    return output;
}

float4 fs_main(VertexOutput input) : SV_Target
{
    float4 color_rgb = t_diffuse.Sample(s_diffuse, input.tex_coords);
    
    // return float4(pow(color_rgb.rgb, 2.2), 1.0);
    return color_rgb;
}