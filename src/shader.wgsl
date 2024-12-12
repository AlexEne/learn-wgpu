struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) vertex_pos: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput = VertexOutput();
    
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4f(x, y, 0.0, 1.0);
    out.vertex_pos = out.clip_position.xyz;
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(0.3, 0.2, 0.1, 1.0);
}