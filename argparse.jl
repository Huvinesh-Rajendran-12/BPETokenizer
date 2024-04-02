using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--train", "-t"
        help="Boolean value to turn on or off the training"
        arg_type=Bool
        default=true

        "--input", "-i"
        help="Path of the input file. Only set the path if --train is true"
        arg_type=String
        default=""

        "--encode", "-e"
        help="Encode text"
        arg_type=String

        "--decode", "-d"
        help="Decode a set of unicode tokens into strings"
        arg_type=Vector{byteVec}

        "--vocab-size", "-v"
        help="Vocabulary size"
        arg_type=Int
        default=300
    end
    return parse_args(s)
end
          
