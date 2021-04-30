#version 330 core

in vec3 frag_position;
out vec4 frag_color;

varying vec4 v_color;
varying vec2 v_texCoord0;
uniform sampler2D u_texture;

void frag_shader1() {
    frag_color = v_color * texture2D(u_texture, v_texCoord0);

    if (frag_color.a == 0.0) {
        discard;
    }
}

void frag_shader2() {
    frag_color = vec4(0.5, 0.5, 0.5, 0.25) * texture2D(u_texture, v_texCoord0).a;
}

void main() {
    
    frag_shader1();
    frag_shader2();

}