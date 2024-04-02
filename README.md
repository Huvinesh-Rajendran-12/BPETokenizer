# BPETokenizer

This is a bpetokenizer inspired from the GPT4 model which merges the most popular tokens to create a new token to add to the vocabulary.

# Usage
```bash
usage: bpetokenizer.jl [-t TRAIN] [-i INPUT] [-e ENCODE] [-d DECODE]
                       [-v VOCAB-SIZE] [-h]
optional arguments:
  -t, --train TRAIN     Boolean value to turn on or off the training
                        (type: Bool, default: true)
  -i, --input INPUT     Path of the input file. Only set the path if
                        --train is true (default: "")
  -e, --encode ENCODE   Encode text
  -d, --decode DECODE   Decode a set of unicode tokens into strings
                        (type: Vector{CodeUnits{UInt8, String}})
  -v, --vocab-size VOCAB-SIZE
                        Vocabulary size (type: Int64, default: 300)
  -h, --help            show this help message and exit
```
