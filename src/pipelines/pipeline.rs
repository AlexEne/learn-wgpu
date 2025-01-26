pub trait Pipeline {
    fn draw_instanced(
        render_pass: &mut wgpu::RenderPass,
        material_instance: PBRMaterialInstance,
        model_gpu_instanced: &ModelGPUDataInstanced,
    );
}
