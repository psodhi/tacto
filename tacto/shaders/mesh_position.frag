#version 330 core

in vec3 frag_position;
uniform mat3 color;

out vec4 frag_color;

// Doing some funny stuff here - passing the object bounds to the shader using the 'color' uniform that PyRender supports for segmentation maps.

void main()
{
//    frag_color = vec4(vec3(.25,.25,.25) + .75 * color[0] * (frag_position - color[1]), 1.0);
    frag_color = vec4((frag_position - color[1]) * color[0], 1);
}
