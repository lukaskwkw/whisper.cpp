const path = require("path");
const { whisper } = require(path.join(
  __dirname,
  "../../build/bin/Release/addon.node.node"
));
const { promisify } = require("util");
const fs = require("fs");

const whisperAsync = promisify(whisper);

const fname_inp = "../../samples/jfk.wav";

const buffer = fs.readFileSync(fname_inp);
const arrayBuffer = buffer.buffer.slice(
  buffer.byteOffset,
  buffer.byteOffset + buffer.byteLength
);

const whisperParams = {
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  fname_inp: "dupa",
  use_gpu: true,
  n_threads: 4,
  array_buffer: arrayBuffer,
  dll_location: path.join(__dirname,
    "../../build/bin/Release/whisper.dll"
  ),
  // dll_location: 'whisper.dll',
  no_timestamps: false
};

const arguments = process.argv.slice(2);
const params = Object.fromEntries(
  arguments.reduce((pre, item) => {
    if (item.startsWith("--")) {
      return [...pre, item.slice(2).split("=")];
    }
    return pre;
  }, [])
);

for (const key in params) {
  if (whisperParams.hasOwnProperty(key)) {
    whisperParams[key] = params[key];
  }
}

console.log("whisperParams =", whisperParams);

whisperAsync(whisperParams).then((result) => {
  console.log(`Result from whisper: ${result}`);
});
