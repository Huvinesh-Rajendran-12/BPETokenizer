using DataStructures
byteVec = Base.CodeUnits{UInt8, String}
mergeDict = Union{OrderedDict{Tuple{byteVec, byteVec}, Tuple{byteVec, Int}}}
