#include "napi.h"
#include "common.h"

#include <windows.h>
#include "whisper.h"

#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>

HINSTANCE hDLL;

typedef WHISPER_API int (*Twhisper_full_n_segments)(struct whisper_context *ctx);
typedef int64_t (*Twhisper_full_get_segment_t0)(whisper_context *ctx, int i_segment);
typedef int64_t (*Twhisper_full_get_segment_t1)(whisper_context *ctx, int i_segment);
typedef const char *(*Twhisper_full_get_segment_text)(whisper_context *ctx, int i_segment);

typedef int (*Twhisper_lang_id)(const char *lang);
typedef whisper_context_params (*Twhisper_context_default_params)();
typedef whisper_context *(*Twhisper_init_from_file_with_params)(const char *path_model, whisper_context_params params);
typedef const char *(*Twhisper_print_system_info)();
typedef int (*Twhisper_is_multilingual)(whisper_context *ctx);
typedef whisper_full_params (*Twhisper_full_default_params)(whisper_sampling_strategy strategy);
typedef void (*Twhisper_print_timings)(whisper_context *ctx);
typedef void (*Twhisper_free)(whisper_context *ctx);
typedef int (*Twhisper_full_parallel)(whisper_context *ctx, whisper_full_params params, const float *samples, int n_samples, int n_processors);

struct whisper_params
{
    int32_t n_threads = min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 5;
    int32_t beam_size = -1;
    int32_t audio_ctx = 0;

    float word_thold = 0.01f;
    float entropy_thold = 2.4f;
    float logprob_thold = -1.0f;

    bool translate = false;
    bool diarize = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool no_prints = false;
    bool use_gpu = true;
    bool flash_attn = false;
    bool comma_in_time = true;

    std::string language = "en";
    std::string prompt;
    std::string dll_location;
    std::string model = "../../ggml-large.bin";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    std::vector<float> pcmf32 = {}; // mono-channel F32 PCM

    std::vector<uint8_t> buffer;

    std::string openvino_encode_device = "CPU";
};

struct whisper_print_user_data
{
    const whisper_params *params;

    const std::vector<std::vector<float>> *pcmf32s;
};

void whisper_print_segment_callback(struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data)
{
    const auto &params = *((whisper_print_user_data *)user_data)->params;
    const auto &pcmf32s = *((whisper_print_user_data *)user_data)->pcmf32s;

    Twhisper_full_n_segments whisper_full_n_segments = (Twhisper_full_n_segments)GetProcAddress(hDLL, "whisper_full_n_segments");
    Twhisper_full_get_segment_t0 whisper_full_get_segment_t0 = (Twhisper_full_get_segment_t0)GetProcAddress(hDLL, "whisper_full_get_segment_t0");
    Twhisper_full_get_segment_t1 whisper_full_get_segment_t1 = (Twhisper_full_get_segment_t1)GetProcAddress(hDLL, "whisper_full_get_segment_t1");
    Twhisper_full_get_segment_text whisper_full_get_segment_text = (Twhisper_full_get_segment_text)GetProcAddress(hDLL, "whisper_full_get_segment_text");

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0;
    int64_t t1;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0)
    {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++)
    {
        if (!params.no_timestamps || params.diarize)
        {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps)
        {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2)
        {
            const int64_t n_samples = pcmf32s[0].size();

            const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
            const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

            double energy0 = 0.0f;
            double energy1 = 0.0f;

            for (int64_t j = is0; j < is1; j++)
            {
                energy0 += fabs(pcmf32s[0][j]);
                energy1 += fabs(pcmf32s[1][j]);
            }

            if (energy0 > 1.1 * energy1)
            {
                speaker = "(speaker 0)";
            }
            else if (energy1 > 1.1 * energy0)
            {
                speaker = "(speaker 1)";
            }
            else
            {
                speaker = "(speaker ?)";
            }

            // printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, %s\n", is0, is1, energy0, energy1, speaker.c_str());
        }

        // colorful print bug
        //
        const char *text = whisper_full_get_segment_text(ctx, i);
        printf("%s%s", speaker.c_str(), text);

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize)
        {
            printf("\n");
        }

        fflush(stdout);
    }
}

void cb_log_disable(enum ggml_log_level, const char *, void *) {}

int run(whisper_params &params, std::vector<std::vector<std::string>> &result)
{

    // std::string dll_location_first = "whisper.dll";
    // HINSTANCE hDLL_first = LoadLibraryA(dll_location_first.c_str());
    fprintf(stderr, "dll_location: %s\n", params.dll_location.c_str());
    // hDLL = LoadLibraryA(params.dll_location.c_str());
    hDLL = LoadLibraryExA(params.dll_location.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);

    if (hDLL == NULL)
    {
        // call GetLastError
        DWORD dw = GetLastError();

        // format message
        LPVOID lpMsgBuf;

        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
                FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            dw,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);

        // display message
        fprintf(stderr, "error: failed to load library: %s\n", (char *)lpMsgBuf);

        return 1;
    }

    Twhisper_lang_id whisper_lang_id = (Twhisper_lang_id)GetProcAddress(hDLL, "whisper_lang_id");
    Twhisper_context_default_params whisper_context_default_params = (Twhisper_context_default_params)GetProcAddress(hDLL, "whisper_context_default_params");
    Twhisper_init_from_file_with_params whisper_init_from_file_with_params = (Twhisper_init_from_file_with_params)GetProcAddress(hDLL, "whisper_init_from_file_with_params");
    Twhisper_print_system_info whisper_print_system_info = (Twhisper_print_system_info)GetProcAddress(hDLL, "whisper_print_system_info");
    Twhisper_is_multilingual whisper_is_multilingual = (Twhisper_is_multilingual)GetProcAddress(hDLL, "whisper_is_multilingual");
    Twhisper_full_default_params whisper_full_default_params = (Twhisper_full_default_params)GetProcAddress(hDLL, "whisper_full_default_params");

    Twhisper_full_n_segments whisper_full_n_segments = (Twhisper_full_n_segments)GetProcAddress(hDLL, "whisper_full_n_segments");
    Twhisper_full_get_segment_t0 whisper_full_get_segment_t0 = (Twhisper_full_get_segment_t0)GetProcAddress(hDLL, "whisper_full_get_segment_t0");
    Twhisper_full_get_segment_t1 whisper_full_get_segment_t1 = (Twhisper_full_get_segment_t1)GetProcAddress(hDLL, "whisper_full_get_segment_t1");
    Twhisper_full_get_segment_text whisper_full_get_segment_text = (Twhisper_full_get_segment_text)GetProcAddress(hDLL, "whisper_full_get_segment_text");
    Twhisper_print_timings whisper_print_timings = (Twhisper_print_timings)GetProcAddress(hDLL, "whisper_print_timings");
    Twhisper_free whisper_free = (Twhisper_free)GetProcAddress(hDLL, "whisper_free");
    Twhisper_full_parallel whisper_full_parallel = (Twhisper_full_parallel)GetProcAddress(hDLL, "whisper_full_parallel");

    if (params.no_prints)
    {
        whisper_log_set(cb_log_disable, NULL);
    }

    if (params.fname_inp.empty() && params.pcmf32.empty())
    {
        fprintf(stderr, "error: no input files or audio buffer specified\n");
        return 2;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1)
    {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        exit(0);
    }

    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    struct whisper_context *ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr)
    {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 3;
    }

    // whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    // if params.pcmf32 is provided, set params.fname_inp to "buffer"
    // this is simpler than further modifications in the code
    // if (!params.pcmf32.empty())
    // {
    //     fprintf(stderr, "info: using audio buffer as input\n");
    //     params.fname_inp.clear();
    //     params.fname_inp.emplace_back("buffer");
    // }

    if (!::read_wav_from_buffer(params.buffer, pcmf32, pcmf32s, params.diarize))
    {
        fprintf(stderr, "error: failed to read audio from buffer\n");
        // continue;
    }

    // print system information
    if (!params.no_prints)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads * params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
    }

    // print some info about the processing
    if (!params.no_prints)
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx))
        {
            if (params.language != "en" || params.translate)
            {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing  (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, timestamps = %d ...\n",
                __func__, int(pcmf32.size()), float(pcmf32.size()) / WHISPER_SAMPLE_RATE,
                params.n_threads, params.n_processors,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime = false;
        wparams.print_progress = params.print_progress;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.print_special = params.print_special;
        wparams.translate = params.translate;
        wparams.language = params.language.c_str();
        wparams.n_threads = params.n_threads;
        wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms = params.offset_t_ms;
        wparams.duration_ms = params.duration_ms;

        wparams.token_timestamps = params.output_wts || params.max_len > 0;
        wparams.thold_pt = params.word_thold;
        wparams.entropy_thold = params.entropy_thold;
        wparams.logprob_thold = params.logprob_thold;
        wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
        wparams.audio_ctx = params.audio_ctx;

        wparams.greedy.best_of = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.initial_prompt = params.prompt.c_str();

        wparams.no_timestamps = params.no_timestamps;

        whisper_print_user_data user_data = {&params, &pcmf32s};

        // this callback is called on each new segment
        if (!wparams.print_realtime)
        {
            wparams.new_segment_callback = whisper_print_segment_callback;
            wparams.new_segment_callback_user_data = &user_data;
        }

        // example for abort mechanism
        // in this example, we do not abort the processing, but we could if the flag is set to true
        // the callback is called before every encoder run - if it returns false, the processing is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void *user_data)
            {
                bool is_aborted = *(bool *)user_data;
                return !is_aborted;
            };
            wparams.encoder_begin_callback_user_data = &is_aborted;
        }

        if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0)
        {
            fprintf(stderr, "failed to process audio\n");
            return 10;
        }
    }

    const int n_segments = whisper_full_n_segments(ctx);
    result.resize(n_segments);
    for (int i = 0; i < n_segments; ++i)
    {
        const char *text = whisper_full_get_segment_text(ctx, i);
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

        result[i].emplace_back(to_timestamp(t0, params.comma_in_time));
        result[i].emplace_back(to_timestamp(t1, params.comma_in_time));
        result[i].emplace_back(text);
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}

class Worker : public Napi::AsyncWorker
{
public:
    Worker(Napi::Function &callback, whisper_params params)
        : Napi::AsyncWorker(callback), params(params) {}

    void Execute() override
    {
        run(params, result);
    }

    void OnOK() override
    {
        Napi::HandleScope scope(Env());
        Napi::Object res = Napi::Array::New(Env(), result.size());
        for (uint64_t i = 0; i < result.size(); ++i)
        {
            Napi::Object tmp = Napi::Array::New(Env(), 3);
            for (uint64_t j = 0; j < 3; ++j)
            {
                tmp[j] = Napi::String::New(Env(), result[i][j]);
            }
            res[i] = tmp;
        }
        Callback().Call({Env().Null(), res});
    }

private:
    whisper_params params;
    std::vector<std::vector<std::string>> result;
};

Napi::Value whisper(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (info.Length() <= 0 || !info[0].IsObject())
    {
        Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
    }
    whisper_params params;

    Napi::Object whisper_params = info[0].As<Napi::Object>();

    std::string log_location = whisper_params.Get("log_location").As<Napi::String>();

    freopen(log_location.c_str(), "w", stderr); // Redirect stderr to a file

    std::string language = whisper_params.Get("language").As<Napi::String>();
    std::string model = whisper_params.Get("model").As<Napi::String>();
    std::string input = whisper_params.Get("fname_inp").As<Napi::String>();
    bool use_gpu = whisper_params.Get("use_gpu").As<Napi::Boolean>();
    bool flash_attn = whisper_params.Get("flash_attn").As<Napi::Boolean>();
    bool no_prints = whisper_params.Get("no_prints").As<Napi::Boolean>();
    bool no_timestamps = whisper_params.Get("no_timestamps").As<Napi::Boolean>();
    std::int32_t max_len = whisper_params.Get("max_len").As<Napi::Number>();
    int32_t audio_ctx = whisper_params.Get("audio_ctx").As<Napi::Number>();
    std::string oved = whisper_params.Get("oved").As<Napi::String>();
    bool comma_in_time = whisper_params.Get("comma_in_time").As<Napi::Boolean>();

    std::string dll_location = whisper_params.Get("dll_location").As<Napi::String>();
    Napi::ArrayBuffer arrayBuffer = whisper_params.Get("array_buffer").As<Napi::ArrayBuffer>();

    Napi::Value pcmf32Value = whisper_params.Get("pcmf32");
    std::vector<float> pcmf32_vec;

    std::int32_t n_threads = whisper_params.Get("n_threads").As<Napi::Number>();

    std::vector<uint8_t> buffer(static_cast<uint8_t *>(arrayBuffer.Data()), static_cast<uint8_t *>(arrayBuffer.Data()) + arrayBuffer.ByteLength());

    if (pcmf32Value.IsTypedArray())
    {
        Napi::Float32Array pcmf32 = pcmf32Value.As<Napi::Float32Array>();
        size_t length = pcmf32.ElementLength();
        pcmf32_vec.reserve(length);
        for (size_t i = 0; i < length; i++)
        {
            pcmf32_vec.push_back(pcmf32[i]);
        }
    }

    params.language = language;
    params.model = model;
    params.fname_inp.emplace_back(input);
    params.use_gpu = use_gpu;
    params.flash_attn = flash_attn;
    params.no_prints = no_prints;
    params.no_timestamps = no_timestamps;
    params.audio_ctx = audio_ctx;
    params.max_len = max_len;
    params.pcmf32 = pcmf32_vec;
    params.comma_in_time = comma_in_time;

    params.buffer = buffer;
    params.dll_location = dll_location;

    params.openvino_encode_device = oved;

    Napi::Function callback = info[1].As<Napi::Function>();
    Worker *worker = new Worker(callback, params);
    worker->Queue();
    return env.Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(
        Napi::String::New(env, "whisper"),
        Napi::Function::New(env, whisper));
    return exports;
}

NODE_API_MODULE(whisper, Init);
