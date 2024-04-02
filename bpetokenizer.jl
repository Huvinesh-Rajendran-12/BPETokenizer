using DataStructures
using StringEncodings
include("types.jl")
include("argparse.jl")

mutable struct BPETokenizer
    mergeDict::mergeDict
    vocabDict::OrderedDict{Int, byteVec}

    function BPETokenizer()
        return new(OrderedDict{Tuple{byteVec, byteVec}, Tuple{byteVec, Int}}(), OrderedDict{Int, byteVec}())
    end
end

mutable struct TrainingParams
   vocab_size::Int
   n_merges::Int
   null_token::String
   null_byte::byteVec

   function TrainingParams(vocab_size::Int, n_merges::Int, null_token)
       null_byte = transcode(UInt8, null_token)
       return new(vocab_size, n_merges, null_token, null_byte)
    end
end

function encode_without_merging(input::String)::Vector{byteVec}
    vec_byteVec = Vector{byteVec}()
    text_tokens = input |> collect
    for token in text_tokens
        push!(vec_byteVec, transcode(UInt8, string(token))) 
    end
    return vec_byteVec
end

function get_stats(ids::Vector{byteVec})::OrderedDict{Tuple{byteVec, byteVec}, Int}
    stats = OrderedDict{Tuple{byteVec,byteVec}, Int}()
    for i in 1:(length(ids) - 1)
        pair = (ids[i], ids[i+1])
        stats[pair] = get(stats, pair, 0) + 1
    end
    return stats
end

function get_top_pair(stats::OrderedDict{Tuple{byteVec, byteVec}, Int})::Tuple{byteVec, byteVec}
    top_val = maximum(values(stats))
    if top_val > 1
        for (key, val) in stats
            if val == top_val
                return key
            end
        end
    else 
        null_code = transcode(UInt8, string("<NULL>"))
        return (null_code, null_code)
    end
end

function merge_pair(tokens::Vector{byteVec}, pair::Tuple{byteVec, byteVec}, new_token::byteVec)::Vector{byteVec}
    newTokens = byteVec[]
    i = 1
    while i < length(tokens) - 1
        if i < length(tokens) && pair[1] == tokens[i] && pair[2] == tokens[i+1]
            push!(newTokens, new_token)
            i += 2
        else
            push!(newTokens, tokens[i])
            i += 1
        end
    end
    return newTokens
end

function build_vocab(merge_dict::OrderedDict{Tuple{byteVec, byteVec}, Tuple{byteVec, Int}})::OrderedDict{Int, byteVec}
    vocab = OrderedDict{Int, byteVec}()
    for idx in 0:255
        vocab[idx] = transcode(UInt8, string(Char(idx))) 
    end
    for ((p0, p1), (merged_token, idx)) in merge_dict
        vocab[idx] = merged_token
    end
    return vocab
end

function train(tokenizer::BPETokenizer, params::TrainingParams, tokens::Vector{byteVec})
    merges = OrderedDict{Tuple{byteVec, byteVec}, Tuple{byteVec, Int}}()
    n_merges = params.n_merges
    for i in 1:n_merges
        stats = get_stats(tokens)
        top_pair = get_top_pair(stats)
        if top_pair[1] == params.null_byte && top_pair[2] == params.null_byte
            break
        end
        idx = 256 + (i - 1)
        new_token = transcode(UInt8, String(vcat(top_pair[1], top_pair[2])))
        tokens = merge_pair(tokens, top_pair, new_token)
        merges[top_pair] = (new_token, idx)
        @info "Merging $top_pair into a new token $new_token"  
    end
    vocab = build_vocab(merges)
    tokenizer.mergeDict = merges
    tokenizer.vocabDict = vocab
end

function encode_tokens(tokenizer::BPETokenizer, input::String)::Vector{byteVec}
    tokens = encode_without_merging(input)
    merges = tokenizer.mergeDict
    while length(tokens) >= 2
        stats = get_stats(tokens)
        newDict = OrderedDict{Tuple{byteVec, byteVec}, Int}(p => get(tokenizer.mergeDict, p, Inf)[2] 
            for p in keys(stats) if get(tokenizer.mergeDict, p, Inf) != Inf)
        if length(newDict) == 0
            break
        end
        pair = findmin(sort(newDict, by=last))[2]
        (new_token, idx) = merges[pair]
        tokens = merge_pair(tokens, pair, new_token)
    end
    return tokens
end

function decode_tokens(ids::Vector{byteVec})::String
    tokens = join([transcode(String, idx) for idx in ids])
    return String(tokens)
end

function main()
    parse_args = parse_commandline()
    tokenizer = BPETokenizer()
    vocab_size = parse_args["vocab-size"]
    training_params = TrainingParams(vocab_size, (vocab_size - 256), "<NULL>")
    if parse_args["train"] && parse_args["input"] == ""
        @error "Must specify training text file path if --train is set to true"
    end
    if parse_args["train"]
        file_path = parse_args["input"]
        file = open(file_path, "r")
        content = read(file, String)
        tokens = encode_without_merging(content)
        train(tokenizer, training_params, tokens)
        return true
    end
    if parse_args["encode"]
        if length(tokenizer.mergeDict) == 0
            @error "Must train tokenizer before encoding"
        end
        input = parse_args["encode"]
        return encode_tokens(tokenizer, input)
    end
    if parse_args["decode"]
        if length(tokenizer.mergeDict) == 0
            @error "Must train tokenizer before decoding"
        end
        input = parse_args["decode"]
        return decode_tokens(input)
    end
end

main()



