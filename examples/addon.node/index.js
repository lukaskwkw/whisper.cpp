const path = require("path");
// const { whisper } = require(path.join(
//   __dirname,
//   "../../build/Release/addon.node"
// ));

const wav = require('wav-decoder');

const { whisper } = require(path.join(
  __dirname,
  "../../build/bin/Release/addon.node"
));
const { promisify } = require("util");
const fs = require("fs");

const whisperAsync = promisify(whisper);

const fname_inp = "../../samples/jfk.wav";
// load fname_inp to ArrayBuffer
const buffer = fs.readFileSync(fname_inp);




const whisperParams = {
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  // fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
  dll_location: path.join(__dirname, '../../build/bin/Release/whisper.dll'),
  use_gpu: true,
  flash_attn: false,
  no_prints: true,
  comma_in_time: false,
  translate: false,
  no_timestamps: false,
  n_threads: 4,
  max_len: 0,
  log_location: path.join(__dirname, './log.txt'),
  // audio_ctx: 750,
  audio_ctx: 0,
  oved: 'CPU'
};



// const arguments = process.argv.slice(2);
// const params = Object.fromEntries(
//   arguments.reduce((pre, item) => {
//     if (item.startsWith("--")) {
//       const [key, value] = item.slice(2).split("=");
//       if (key === "audio_ctx") {
//         whisperParams[key] = parseInt(value);
//       } else {
//         whisperParams[key] = value;
//       }
//       return pre;
//     }
//     return pre;
//   }, [])
// );

// for (const key in params) {
//   if (whisperParams.hasOwnProperty(key)) {
//     whisperParams[key] = params[key];
//   }
// }


console.log("whisperParams =", whisperParams);

const byUsingWavFilePath = () => {
  const startTime = process.hrtime()

  const whisperParamsExtra = { ...whisperParams };

  whisperParamsExtra.fname_inp = path.join(__dirname, "../../samples/jfk.wav");

  whisperAsync(whisperParamsExtra).then((result) => {
    const endTime = process.hrtime(startTime)
    console.log();
    console.log(`Result from whisper: ${result}`);
    const elapsedTime = endTime[0] + endTime[1] / 1e9
    console.log(
      `Transcription time ${elapsedTime.toFixed(2)}s`
    );
  });
}

const byUsingVectorArray = () => {
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  );

  const startTime = process.hrtime()

  const whisperParamsExtra = { ...whisperParams };

  whisperParamsExtra.array_buffer = arrayBuffer;

  whisperAsync(whisperParamsExtra).then((result) => {
    const endTime = process.hrtime(startTime)
    console.log();
    console.log(`Result from whisper: ${result}`);
    const elapsedTime = endTime[0] + endTime[1] / 1e9
    console.log(
      `Transcription time ${elapsedTime.toFixed(2)}s`
    );
  });
}


// const startTime = process.hrtime()

const byUsingWavDecoder = () => {
  wav.decode(buffer).then((audioData) => {
    const startTime = process.hrtime()

    const floatArray = audioData.channelData[0]; // Assuming mono audio for simplicity

    // console.log('Float32Array:', floatArray.slice(0, 10)); // Check the first few elements

    const whisperParamsExtra = { ...whisperParams };

    whisperParamsExtra.pcmf32 = new Float32Array(floatArray),

      whisperAsync(whisperParamsExtra).then((result) => {
        const endTime = process.hrtime(startTime)
        console.log();
        console.log(`Result from whisper: ${result}`);
        const elapsedTime = endTime[0] + endTime[1] / 1e9
        console.log(
          `Transcription time ${elapsedTime.toFixed(2)}s`
        );
      });
  }).catch((err) => {
    console.error('Error decoding WAV file:', err);
  });
}

// byUsingWavDecoder();
byUsingVectorArray();
// byUsingWavFilePath();