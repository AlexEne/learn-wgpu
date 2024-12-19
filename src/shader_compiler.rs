use shaderc::{CompileOptions, IncludeType, ResolvedInclude};

struct MyIncludeResolver;

impl MyIncludeResolver {
    fn resolve_include(
        requested_source: &str,
        include_type: IncludeType,
        requesting_source: &str,
        _include_depth: usize,
    ) -> Result<ResolvedInclude, String> {
        // Example: Simple resolver that loads files from the local directory
        // Adjust this logic to match your project's structure
        let include_path = match include_type {
            IncludeType::Standard => requested_source.to_string(),
            IncludeType::Relative => {
                // Assuming `requesting_source` is a path, resolve relative paths
                let base_path = std::path::Path::new(requesting_source)
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."));
                base_path
                    .join(requested_source)
                    .to_str()
                    .unwrap()
                    .to_string()
            }
        };

        println!("{:?}", std::env::current_dir());

        // Read the file's contents
        let include_path = "./src/shaders/".to_owned() + &include_path;
        let source_code = std::fs::read_to_string(&include_path)
            .map_err(|e| format!("Failed to load include file {}: {}", include_path, e))?;

        Ok(ResolvedInclude {
            resolved_name: include_path,
            content: source_code,
        })
    }
}

pub fn compile_shaders() -> (shaderc::CompilationArtifact, shaderc::CompilationArtifact) {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut compile_options = CompileOptions::new().unwrap();
    compile_options.set_include_callback(MyIncludeResolver::resolve_include);

    let vertex_shader = compiler
        .compile_into_spirv(
            include_str!("shaders/vertex_shader.vert"),
            shaderc::ShaderKind::Vertex,
            "shader.vert",
            "main",
            Some(&compile_options),
        )
        .unwrap();

    let fragment_shader = compiler
        .compile_into_spirv(
            include_str!("shaders/fragment.frag"),
            shaderc::ShaderKind::Fragment,
            "fragment.frag",
            "main",
            Some(&compile_options),
        )
        .unwrap();

    (vertex_shader, fragment_shader)
}
