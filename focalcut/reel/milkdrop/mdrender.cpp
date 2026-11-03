// Render MilkDrop presets natively (projectM 4, off-screen EGL) frame by frame from a song, as raw RGB24 on stdout.
// Each frame gets the song's samples since the previous frame and a fixed frame time, so the layer is
// deterministic in time and frame-exact against the cut.
//
//   mdrender W H FPS FRAMES START song.s16 schedule.tsv > frames.rgb
//     song.s16: the whole song, interleaved stereo 16-bit 44.1 kHz; START: the cut's first frame in the song (s)
//     schedule.tsv: "frame<TAB>blend seconds<TAB>preset.milk" per preset change, in frame order
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <projectM-4/projectM.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

typedef void (*GenFB)(GLsizei, GLuint*); typedef void (*BindFB)(GLenum, GLuint);
typedef void (*FBTex)(GLenum, GLenum, GLenum, GLuint, GLint);

int main(int argc, char** argv) {
    if (argc != 8) { fprintf(stderr, "usage: mdrender W H FPS FRAMES START song.s16 schedule.tsv\n"); return 2; }
    const int W = atoi(argv[1]), H = atoi(argv[2]), fps = atoi(argv[3]), frames = atoi(argv[4]);
    const double start = atof(argv[5]);
    const int SR = 44100;
    std::vector<int16_t> pcm;
    { std::ifstream b(argv[6], std::ios::binary | std::ios::ate); size_t n = b.tellg(); b.seekg(0);
      pcm.assign(n / 2, 0); b.read(reinterpret_cast<char*>(pcm.data()), n & ~size_t(1)); }
    const long total = pcm.size() / 2;
    std::map<int, std::pair<double, std::string>> sched;
    { std::ifstream s(argv[7]); std::string line;
      while (std::getline(s, line)) { std::istringstream l(line); int f; double bl; std::string p;
        if (l >> f >> bl && std::getline(l >> std::ws, p)) sched[f] = {bl, p}; } }

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(dpy, nullptr, nullptr)) { fprintf(stderr, "mdrender: no EGL display\n"); return 1; }
    const EGLint ca[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE, 8,
                         EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
    EGLConfig cfg; EGLint nc;
    eglChooseConfig(dpy, ca, &cfg, 1, &nc);
    const EGLint pa[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_NONE};
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pa);
    eglBindAPI(EGL_OPENGL_API);
    const EGLint xa[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                         EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xa);
    if (!ctx || !eglMakeCurrent(dpy, surf, surf, ctx)) { fprintf(stderr, "mdrender: no GL 3.3 context\n"); return 1; }
    fprintf(stderr, "[mdrender] %s\n", glGetString(GL_RENDERER));

    auto genFB = (GenFB)eglGetProcAddress("glGenFramebuffers"); auto bindFB = (BindFB)eglGetProcAddress("glBindFramebuffer");
    auto fbTex = (FBTex)eglGetProcAddress("glFramebufferTexture2D");
    GLuint tex, fbo;
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    genFB(1, &fbo); bindFB(GL_FRAMEBUFFER, fbo); fbTex(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    projectm_handle pm = projectm_create();
    if (!pm) { fprintf(stderr, "mdrender: projectM failed to start\n"); return 1; }
    projectm_set_window_size(pm, W, H);
    projectm_set_fps(pm, fps);
    projectm_set_mesh_size(pm, 48, 32);
    projectm_set_preset_locked(pm, true);
    projectm_set_hard_cut_enabled(pm, false);
    projectm_set_preset_duration(pm, 1e9);

    std::vector<unsigned char> rgba((size_t)W * H * 4), rgb((size_t)W * H * 3);
    std::vector<float> buf;
    long prev = std::lround(start * SR) - 1024;
    for (int k = 0; k < frames; k++) {
        auto it = sched.find(k);
        if (it != sched.end()) {
            projectm_set_soft_cut_duration(pm, it->second.first > 0 ? it->second.first : 0.001);
            projectm_load_preset_file(pm, it->second.second.c_str(), it->second.first > 0);
        }
        long end = std::lround((start + (double)k / fps) * SR);
        buf.clear();
        for (long j = prev; j < end; j++)
            for (int c = 0; c < 2; c++) buf.push_back(j >= 0 && j < total ? pcm[2 * j + c] / 32768.f : 0.f);
        if (end > prev) projectm_pcm_add_float(pm, buf.data(), end - prev, PROJECTM_STEREO);
        prev = end;
        projectm_set_frame_time(pm, (double)k / fps);
        projectm_opengl_render_frame_fbo(pm, fbo);
        bindFB(GL_FRAMEBUFFER, fbo);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        for (int y = 0; y < H; y++) {                        // GL rows run bottom up
            const unsigned char* s = &rgba[(size_t)(H - 1 - y) * W * 4];
            unsigned char* d = &rgb[(size_t)y * W * 3];
            for (int x = 0; x < W; x++) { d[3 * x] = s[4 * x]; d[3 * x + 1] = s[4 * x + 1]; d[3 * x + 2] = s[4 * x + 2]; }
        }
        if (fwrite(rgb.data(), 1, rgb.size(), stdout) != rgb.size()) return 1;
    }
    projectm_destroy(pm);
    eglTerminate(dpy);
    return 0;
}
